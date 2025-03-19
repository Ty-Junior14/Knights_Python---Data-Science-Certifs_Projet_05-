"""
Application de Prédiction du Niveau de la Mer
-------------------------------------------
Auteur: [Votre nom]
Date: [Date]

Description:
Cette application analyse et visualise l'évolution du niveau de la mer
en utilisant des données historiques et des méthodes de prédiction statistiques.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, norm
import numpy as np
import io
from typing import Tuple
from src.analysis import add_advanced_statistics, create_advanced_plots, seasonal_decomposition
from src.utils import validate_data_quality

# Configuration générale de l'application
st.set_page_config(
    page_title="Prédicteur du Niveau de la Mer",
    page_icon="🌊",
    layout="wide"  # Mise en page large pour meilleure visualisation
)

# Style CSS personnalisé pour améliorer l'interface
st.markdown("""
    <style>
    /* Style principal de l'application */
    .main {
        padding: 2rem;
    }
    /* Style du titre */
    .stTitle {
        color: #2c3e50;
        font-weight: bold;
    }
    /* Style des métriques */
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """
    Charge et prétraite les données du niveau de la mer
    
    Fonctionnalités:
    - Lecture du fichier CSV source
    - Validation des données d'entrée
    - Conversion des dates
    - Gestion des valeurs manquantes
    
    Retourne:
    DataFrame pandas contenant les données nettoyées
    """
    df = pd.read_csv("epa-sea-level.csv")
    # Conversion des années en format datetime pour meilleure manipulation
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-01-01')
    return df

def create_prediction_data(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    use_recent: bool = False
) -> Tuple[pd.Series, pd.Series, float, float, float]:
    """
    Génère les prédictions du niveau de la mer
    
    Paramètres:
    - df: DataFrame contenant les données historiques
    - start_year: Année de début pour les prédictions
    - end_year: Année de fin pour les prédictions
    - use_recent: Si True, utilise uniquement les données depuis 2000
    
    Retourne:
    - years: Série des années de prédiction
    - predictions: Valeurs prédites
    - slope: Pente de la régression (taux de changement)
    - intercept: Ordonnée à l'origine
    - r2: Coefficient de détermination (qualité du modèle)
    """
    # Filtrage des données récentes si demandé
    if use_recent:
        df = df[df["Year"] >= 2000]
    
    # Calcul de la régression linéaire
    slope, intercept, r_value, p_value, std_err = linregress(
        df["Year"], 
        df["CSIRO Adjusted Sea Level"]
    )
    
    # Génération des prédictions
    years = pd.Series(range(start_year, end_year + 1))
    predictions = slope * years + intercept
    
    return years, predictions, slope, intercept, r_value**2

def add_configuration_section():
    """
    Configure les paramètres de l'analyse dans la barre latérale
    
    Paramètres configurables:
    - Niveau de confiance des prédictions
    - Type de régression
    - Paramètres visuels
    - Options d'affichage
    """
    st.sidebar.markdown("## Configuration")
    
    # Section des paramètres du modèle
    st.sidebar.subheader("Paramètres du Modèle")
    confidence_level = st.sidebar.slider(
        "Niveau de Confiance (%)",
        min_value=80,
        max_value=99,
        value=95,
        step=1,
        help="Définit la précision des intervalles de prédiction"
    )
    
    # Choix du type de régression
    regression_type = st.sidebar.selectbox(
        "Type de Régression",
        ["Linéaire", "Polynomiale", "Moyenne Mobile"],
        help="Sélectionnez la méthode de prédiction"
    )
    
    # Options supplémentaires pour régression polynomiale
    if regression_type == "Polynomiale":
        degree = st.sidebar.slider(
            "Degré du Polynôme",
            min_value=2,
            max_value=5,
            value=2,
            help="Complexité du modèle polynomial"
        )
    
    # Paramètres visuels
    st.sidebar.subheader("Paramètres de Visualisation")
    theme = st.sidebar.selectbox(
        "Thème du Graphique",
        ["plotly_white", "plotly_dark", "seaborn", "simple_white"],
        help="Apparence générale du graphique"
    )
    
    # Options d'affichage
    show_uncertainty = st.sidebar.checkbox(
        "Afficher les Bandes d'Incertitude",
        value=True,
        help="Visualiser la marge d'erreur des prédictions"
    )
    
    return {
        "confidence_level": confidence_level,
        "regression_type": regression_type,
        "theme": theme,
        "show_uncertainty": show_uncertainty,
        "degree": degree if regression_type == "Polynomiale" else None
    }

def calculate_moving_average(df, window=5):
    """Calcule la moyenne mobile des données du niveau de la mer"""
    return df["CSIRO Adjusted Sea Level"].rolling(window=window).mean()

def add_confidence_bands(fig, x, y, confidence_level, std_err):
    """
    Ajoute les bandes de confiance au graphique
    
    Paramètres:
    - fig: Figure Plotly
    - x: Données de l'axe X
    - y: Données de l'axe Y
    - confidence_level: Niveau de confiance
    - std_err: Erreur standard
    """
    z_value = norm.ppf(1 - (1 - confidence_level/100)/2)
    std_dev = np.std(y)
    
    upper = y + z_value * std_dev
    lower = y - z_value * std_dev
    
    fig.add_trace(go.Scatter(
        x=x,
        y=upper,
        fill=None,
        mode='lines',
        line_color='rgba(68, 68, 68, 0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=x,
        y=lower,
        fill='tonexty',
        mode='lines',
        line_color='rgba(68, 68, 68, 0)',
        name=f'Intervalle de Confiance {confidence_level}%'
    ))

@handle_errors
def main():
    """Fonction principale de l'application"""
    # En-tête
    st.title("🌊 Analyse de la Montée du Niveau de la Mer")
    st.markdown("---")

    # Chargement des données
    with st.spinner("Chargement des données..."):
        df = load_data()

    # Add data quality warnings
    warnings = validate_data_quality(df)
    if warnings:
        with st.expander("Avertissements sur la Qualité des Données"):
            for warning in warnings:
                st.warning(warning)

    # Sidebar controls
    st.sidebar.header("Contrôles du Tableau de Bord")
    
    # Add more control options
    analysis_type = st.sidebar.selectbox(
        "Type d'Analyse",
        ["De Base", "Avancé"],
        help="Choisissez le niveau d'analyse statistique à afficher"
    )
    
    chart_type = st.sidebar.selectbox(
        "Type de Graphique",
        ["Ligne + Nuage", "Zone", "Bandes de Confiance"],
        help="Sélectionnez le style de visualisation"
    )
    
    # Existing controls
    prediction_end_year = st.sidebar.slider(
        "Prédire jusqu'en année:", 
        min_value=2020,
        max_value=2100,
        value=2050,
        step=5
    )
    
    show_confidence = st.sidebar.checkbox("Afficher l'Intervalle de Confiance", value=True)
    show_trend_analysis = st.sidebar.checkbox("Afficher l'Analyse de Tendance", value=True)

    # Add export options
    st.sidebar.markdown("### Options d'Exportation")
    export_format = st.sidebar.selectbox(
        "Format d'Exportation",
        ["CSV", "Excel", "JSON"]
    )
    
    if st.sidebar.button("Exporter les Données"):
        if export_format == "CSV":
            csv = df.to_csv(index=False)
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name="sea_level_data.csv",
                mime="text/csv"
            )
        elif export_format == "Excel":
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            st.download_button(
                label="Télécharger Excel",
                data=buffer,
                file_name="sea_level_data.xlsx",
                mime="application/vnd.ms-excel"
            )
        else:
            json_str = df.to_json(orient="records")
            st.download_button(
                label="Télécharger JSON",
                data=json_str,
                file_name="sea_level_data.json",
                mime="application/json"
            )

    # Add configuration section
    config = add_configuration_section()

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Visualisation des Prédictions du Niveau de la Mer")
        
        # Sélecteur de plage de dates
        min_date = pd.to_datetime(str(int(df["Year"].min())) + '-01-01')
        max_date = pd.to_datetime(str(int(df["Year"].max())) + '-01-01')
        
        date_range = st.date_input(
            "Sélectionner la Plage de Dates",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            help="Filtrer les données par plage de dates"
        )
        
        # Filter data based on selection using Year
        start_year = date_range[0].year
        end_year = date_range[1].year
        mask = (df["Year"] >= start_year) & (df["Year"] <= end_year)
        filtered_df = df[mask]
        
        # Create Plotly figure
        fig = go.Figure()

        # Add actual data points
        fig.add_trace(go.Scatter(
            x=filtered_df["Year"],
            y=filtered_df["CSIRO Adjusted Sea Level"],
            mode='markers',
            name='Données Historiques',
            marker=dict(color='blue', size=8)
        ))

        # Add predictions
        years_full, predictions_full, slope_full, intercept_full, r2_full = create_prediction_data(
            filtered_df, 1880, prediction_end_year
        )
        
        years_recent, predictions_recent, slope_recent, intercept_recent, r2_recent = create_prediction_data(
            filtered_df, 2000, prediction_end_year, use_recent=True
        )

        # Add full period prediction line
        fig.add_trace(go.Scatter(
            x=years_full,
            y=predictions_full,
            mode='lines',
            name='Tendance sur Toute la Période',
            line=dict(color='red', width=2)
        ))

        # Add recent period prediction line
        fig.add_trace(go.Scatter(
            x=years_recent,
            y=predictions_recent,
            mode='lines',
            name='Tendance Récente (2000+)',
            line=dict(color='green', width=2)
        ))

        # Update layout
        fig.update_layout(
            title="Montée du Niveau de la Mer au Fil du Temps",
            xaxis_title="Année",
            yaxis_title="Niveau de la Mer (pouces)",
            hovermode='x unified',
            template="plotly_white"
        )

        # Update visualization based on configuration
        if config["regression_type"] == "Polynomiale":
            # Add polynomial regression
            poly_features = np.polynomial.polynomial.polyfit(
                filtered_df["Year"],
                filtered_df["CSIRO Adjusted Sea Level"],
                deg=config["degree"]
            )
            # Add polynomial prediction line
            years_range = np.linspace(
                filtered_df["Year"].min(),
                prediction_end_year,
                num=500
            )
            predictions = np.polynomial.polynomial.polyval(years_range, poly_features)
            
            fig.add_trace(go.Scatter(
                x=years_range,
                y=predictions,
                mode='lines',
                name=f'Polynomial (degree={config["degree"]})',
                line=dict(color='purple', width=2)
            ))
        
        # Add uncertainty bands if enabled
        if config["show_uncertainty"]:
            add_confidence_bands(fig, years_full, predictions_full, config["confidence_level"], np.sqrt(std_err**2 * (1 + 1/len(filtered_df))))

        # Update layout with selected theme
        fig.update_layout(template=config["theme"])

        # Add advanced analysis if selected
        if analysis_type == "Avancé":
            # Calculate advanced statistics
            advanced_stats = add_advanced_statistics(filtered_df)
            
            # Create additional plots
            advanced_plots = create_advanced_plots(filtered_df, advanced_stats)
            
            # Display additional plots in new tabs
            tabs = st.tabs(["Main Plot", "Taux de Variation", "Tendances Décennales"])
            
            with tabs[0]:
                st.plotly_chart(fig, use_container_width=True)
            
            with tabs[1]:
                st.plotly_chart(advanced_plots['rate_of_change'], use_container_width=True)
            
            with tabs[2]:
                st.plotly_chart(advanced_plots['decadal_trends'], use_container_width=True)
                
            # Add seasonal decomposition if requested
            if st.checkbox("Afficher la Décomposition Saisonnière"):
                decomp = seasonal_decomposition(filtered_df)
                st.line_chart(decomp['trend'])
                st.line_chart(decomp['seasonal'])
                st.line_chart(decomp['residual'])
        else:
            # Display only the main plot for basic analysis
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Analyse Statistique")
        
        # Add metric cards
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric(
                "Niveau de Montée de la Mer Moyen",
                f"{df['CSIRO Adjusted Sea Level'].mean():.2f} pouces",
                f"{slope_full:.4f} pouces/année"
            )
        with col2_2:
            st.metric(
                "Montée Prévue par 2050",
                f"{predictions_full.iloc[-1]:.2f} pouces",
                f"{predictions_full.iloc[-1] - df['CSIRO Adjusted Sea Level'].iloc[-1]:.2f} pouces"
            )

        # Display trend analysis
        if show_trend_analysis:
            st.markdown("### Analyse de Tendance")
            
            # Full period statistics
            st.markdown("**Période Complexe (1880-présent)**")
            st.write(f"- Pente: {slope_full:.4f} pouces/année")
            st.write(f"- Score R²: {r2_full:.4f}")
            st.write(f"- Montée prévue par {prediction_end_year}: "
                    f"{predictions_full.iloc[-1]-predictions_full.iloc[0]:.2f} pouces")
            
            # Recent period statistics
            st.markdown("**Période Récente (2000-présent)**")
            st.write(f"- Pente: {slope_recent:.4f} pouces/année")
            st.write(f"- Score R²: {r2_recent:.4f}")
            st.write(f"- Montée prévue par {prediction_end_year}: "
                    f"{predictions_recent.iloc[-1]-predictions_recent.iloc[0]:.2f} pouces")

        # Summary statistics
        st.markdown("### Statistiques de Résumé")
        summary_stats = df["CSIRO Adjusted Sea Level"].describe()
        st.dataframe(summary_stats)

        # Add data exploration tab
        if st.checkbox("Afficher l'Explorateur de Données Brutes"):
            st.subheader("Explorateur de Données Brutes")
            
            # Add column selector
            columns_to_show = st.multiselect(
                "Sélectionner les Colonnes",
                df.columns.tolist(),
                default=["Year", "CSIRO Adjusted Sea Level"]
            )
            
            # Show interactive table
            st.dataframe(
                df[columns_to_show],
                use_container_width=True,
                height=400
            )
            
            # Add basic statistics
            if st.checkbox("Afficher les Statistiques de Base"):
                st.write(df[columns_to_show].describe())

if __name__ == "__main__":
    main() 