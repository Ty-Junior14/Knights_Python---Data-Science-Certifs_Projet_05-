# This entrypoint file to be used in development. Start by reading README.md
import sea_level_predictor
from unittest import main
import plotly.graph_objects as go
import streamlit as st

# Test your function by calling it here
sea_level_predictor.draw_plot()

# Run unit tests automatically
main(module='test_module', exit=False)

def create_dynamic_plot(df, config, filtered_df, predictions):
    """
    Crée un graphique dynamique avec coloration adaptative
    
    Paramètres:
    - df: DataFrame complet
    - config: Configuration utilisateur
    - filtered_df: DataFrame filtré
    - predictions: Données de prédiction
    """
    # Création de la figure
    fig = go.Figure()

    # Palette de couleurs dynamique basée sur le thème
    if config['theme'] == 'plotly_dark':
        color_scale = 'Viridis'
        base_color = 'rgb(255, 255, 255)'
    else:
        color_scale = 'RdBu'
        base_color = 'rgb(0, 0, 0)'

    # Ajout des données historiques avec coloration basée sur la variation
    variation = filtered_df["CSIRO Adjusted Sea Level"].diff()
    fig.add_trace(go.Scatter(
        x=filtered_df["Year"],
        y=filtered_df["CSIRO Adjusted Sea Level"],
        mode='markers',
        name='Données Historiques',
        marker=dict(
            size=8,
            color=variation,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                title="Variation Annuelle (pouces)"
            )
        ),
        hovertemplate="Année: %{x}<br>Niveau: %{y:.2f} pouces<br>Variation: %{marker.color:.2f}<extra></extra>"
    ))

    # Ajout des lignes de prédiction avec opacité dynamique
    years_full, predictions_full = predictions['full']
    years_recent, predictions_recent = predictions['recent']

    # Ligne de tendance complète
    fig.add_trace(go.Scatter(
        x=years_full,
        y=predictions_full,
        mode='lines',
        name='Tendance Globale',
        line=dict(
            color='rgba(255, 0, 0, 0.8)',
            width=2,
            dash='solid'
        ),
        hovertemplate="Année: %{x}<br>Prédiction: %{y:.2f} pouces<extra></extra>"
    ))

    # Ligne de tendance récente
    fig.add_trace(go.Scatter(
        x=years_recent,
        y=predictions_recent,
        mode='lines',
        name='Tendance Récente',
        line=dict(
            color='rgba(0, 255, 0, 0.8)',
            width=2,
            dash='dash'
        ),
        hovertemplate="Année: %{x}<br>Prédiction: %{y:.2f} pouces<extra></extra>"
    ))

    # Mise à jour du layout avec des éléments interactifs
    fig.update_layout(
        title={
            'text': "Évolution du Niveau de la Mer",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Année",
        yaxis_title="Niveau de la Mer (pouces)",
        hovermode='x unified',
        template=config['theme'],
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Réinitialiser",
                        method="relayout",
                        args=[{"xaxis.range": [df["Year"].min(), df["Year"].max()]}]
                    )
                ],
                x=0.05,
                y=1.15,
            )
        ],
        annotations=[
            dict(
                text=f"Niveau de confiance: {config['confidence_level']}%",
                showarrow=False,
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper"
            )
        ]
    )

    # Ajout d'interactivité supplémentaire
    fig.update_xaxes(rangeslider_visible=True)
    
    return fig

def main():
    # ... code existant ...

    with col1:
        st.subheader("Visualisation des Prédictions du Niveau de la Mer")
        
        # Création des prédictions
        predictions = {
            'full': create_prediction_data(filtered_df, 1880, prediction_end_year),
            'recent': create_prediction_data(filtered_df, 2000, prediction_end_year, use_recent=True)
        }
        
        # Création du graphique dynamique
        fig = create_dynamic_plot(df, config, filtered_df, predictions)
        
        # Affichage du graphique
        st.plotly_chart(fig, use_container_width=True)

        # Ajout de contrôles supplémentaires
        if st.checkbox("Afficher les Options Avancées"):
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                marker_size = st.slider(
                    "Taille des Points",
                    min_value=5,
                    max_value=15,
                    value=8
                )
            with col1_2:
                line_width = st.slider(
                    "Épaisseur des Lignes",
                    min_value=1,
                    max_value=5,
                    value=2
                )