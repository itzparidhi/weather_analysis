import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def load_cleaned_data():
    """Load the cleaned weather data"""
    try:
        df = pd.read_csv('cleaned_weather_data.csv')
        print("Successfully loaded cleaned weather data")
        return df
    except FileNotFoundError:
        print("Error: 'cleaned_weather_data.csv' not found. Please run preprocessing first.")
        return None

def plot_temperature_distribution(df):
    """Plot temperature distributions by city"""
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='TAVG', hue='City', kde=True, palette='viridis', element='step')
    plt.title('Average Temperature Distribution by City')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.show()

# def plot_correlation_matrix(df):
#     """Plot correlation matrix of weather features"""
#     plt.figure(figsize=(10, 8))
#     corr_cols = ['TAVG', 'TMAX', 'TMIN', 'Humidity (%)', 'Wind (km/h)', 'Feels Like']
#     corr = df[corr_cols].corr()
#     sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
#     plt.title('Weather Parameters Correlation Matrix')
#     plt.tight_layout()
#     plt.show()

def analyze_probability_distributions(df):
    """Analyze temperature probability distributions by city"""
    plt.figure(figsize=(12, 6))
    
    for city in df['City'].unique():
        city_temp = df[df['City'] == city]['TAVG']
        mu, std = stats.norm.fit(city_temp)
        
        # Plot histogram
        sns.histplot(city_temp, kde=False, stat='density', alpha=0.5, label=f'{city} Data')
        
        # Plot fitted normal distribution
        x = np.linspace(city_temp.min(), city_temp.max(), 100)
        pdf = stats.norm.pdf(x, mu, std)
        plt.plot(x, pdf, label=f'{city} Normal Fit')
    
    plt.title('Temperature Distributions with Normal Fits')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def perform_hypothesis_tests(df):
    """Perform statistical hypothesis tests"""
    print("\n=== Hypothesis Testing Results ===")
    
    # Compare Delhi and Bangalore temperatures
    delhi_temp = df[df['City'] == 'Delhi']['TAVG']
    bangalore_temp = df[df['City'] == 'Bangalore']['TAVG']
    t_stat, p_val = stats.ttest_ind(delhi_temp, bangalore_temp)
    print(f"\nDelhi vs Bangalore Temperatures (t-test):")
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")
    
    # Compare coastal vs inland cities
    coastal = df[df['City'].isin(['Mumbai', 'Chennai'])]['TAVG']
    inland = df[df['City'].isin(['Delhi', 'Bangalore'])]['TAVG']
    t_stat, p_val = stats.ttest_ind(coastal, inland)
    print(f"\nCoastal vs Inland Cities Temperatures (t-test):")
    print(f"T-statistic: {t_stat:.4f}, p-value: {p_val:.4f}")

def perform_anova(df):
    """Perform ANOVA analysis"""
    print("\n=== ANOVA Results ===")
    model = ols('TAVG ~ C(City)', data=df).fit()
    anova_results = anova_lm(model, typ=2)
    print(anova_results)
    
    # Plot city-wise temperature comparisons
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='City', y='TAVG', palette='viridis')
    plt.title('Temperature Distribution by City (ANOVA)')
    plt.xlabel('City')
    plt.ylabel('Average Temperature (°C)')
    plt.show()

def perform_regression_analysis(df):
    """Perform regression analysis"""
    print("\n=== Regression Analysis ===")
    
    # Simple linear regression: TAVG vs Feels Like
    X = df['TAVG']
    y = df['Feels Like']
    X = sm.add_constant(X)
    model_simple = sm.OLS(y, X).fit()
    print("\nSimple Linear Regression (TAVG → Feels Like):")
    print(model_simple.summary())
    
    # Multiple regression: TAVG + Humidity + Wind → Feels Like
    X_multi = df[['TAVG', 'Humidity (%)', 'Wind (km/h)']]
    X_multi = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi).fit()
    print("\nMultiple Regression (TAVG + Humidity + Wind → Feels Like):")
    print(model_multi.summary())
    
    # Plot regression results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.regplot(x='TAVG', y='Feels Like', data=df, line_kws={'color': 'red'})
    plt.title('Simple Linear Regression')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=model_multi.predict(), y=y, hue=df['City'])
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Predicted Feels Like Temperature')
    plt.ylabel('Actual Feels Like Temperature')
    plt.title('Multiple Regression Fit')
    
    plt.tight_layout()
    plt.show()


def plot_monthly_temperature_trends(df):
    """Line plot of monthly average temperatures per city"""
    plt.figure(figsize=(12, 6))
    
    # Calculate monthly averages
    monthly_avg = df.groupby(['City', 'Month'])['TAVG'].mean().reset_index()
    
    # Plot with Seaborn
    sns.lineplot(data=monthly_avg, x='Month', y='TAVG', hue='City', 
                 palette='viridis', marker='o', linewidth=2.5)
    
    plt.title('Monthly Average Temperature Trends by City')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.xticks(range(1, 13), 
               ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    """Main analysis workflow"""
    # Load cleaned data
    df = load_cleaned_data()
    if df is None:
        return
    
    # Perform EDA and statistical analysis
    plot_temperature_distribution(df)
    plot_correlation_matrix(df)
    analyze_probability_distributions(df)
    perform_hypothesis_tests(df)
    perform_anova(df)
    perform_regression_analysis(df)
    plot_monthly_temperature_trends(df)
    
    print("\nAnalysis complete. Close the plot windows to exit.")

if __name__ == "__main__":
    main()