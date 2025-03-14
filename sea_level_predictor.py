import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress

def draw_plot():
    # Read data from file
    df = pd.read_csv("epa-sea-level.csv")


    # Create scatter plot
    plt.figure(figsize=(10, 6))


    # Create first line of best fit
    plt.scatter(df["Year"], df["CSIRO Adjusted Sea Level"], label="Données réelles")



    # Create second line of best fit
    slope, intercept, _, _, _ = stats.linregress(df["Year"], df["CSIRO Adjusted Sea Level"])
    years = pd.Series(range(1880, 2051))
    plt.plot(years, slope * years + intercept, 'r', label="Régression linéaire (1880-2050)")


    # Add labels and title
    df_recent = df[df["Year"] >= 2000]
    slope_recent, intercept_recent, _, _, _ = stats.linregress(df_recent["Year"], df_recent["CSIRO Adjusted Sea Level"])
    years_recent = pd.Series(range(2000, 2051))
    plt.plot(years_recent, slope_recent * years_recent + intercept_recent, 'g', label="Régression linéaire (2000-2050)")

    plt.xlabel("Year")
    plt.ylabel("Sea Level (inches)")
    plt.title("Rise in Sea Level")
    plt.legend()
    plt.grid()


    
    # Save plot and return data for testing (DO NOT MODIFY)
    plt.savefig('sea_level_plot.png')
    return plt.gca()