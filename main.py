import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

season_names = {1: 'Summer', 2: 'Monsoon', 3: 'Winter'}
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']  # All available cities

def preprocess_data():
    """Data preprocessing function"""
    # 1.) Load the dataset(excel file)
    excel_file = "multi_city_weather_2019_2023.xlsx"
    excel_data = pd.read_excel(excel_file, sheet_name="Sheet1")
    
    # 2.) Data Preprocessing (with if-else checks)
    # 1. Convert 'Date' to datetime format
    if 'Date' in excel_data.columns:
        excel_data['Date'] = pd.to_datetime(excel_data['Date'], errors='coerce')
        # Check if any dates failed to parse
        if excel_data['Date'].isnull().any():
            print("Warning: Some dates couldn't be parsed and were set to NaT")
        else:
            print("All dates converted successfully")
    else:
        print("Error: 'Date' column not found in dataset")

    # 2. Check for missing values (with if-else)
    if excel_data.isnull().sum().sum() == 0:
        print("No missing values found in the dataset")
    else:
        print("Missing values detected:")
        print(excel_data.isnull().sum())
        # Add handling here if you later find missing values

    # 3. Check for duplicates
    if excel_data.duplicated().sum() > 0:
        print(f"Found {excel_data.duplicated().sum()} duplicate rows - removing them")
        excel_data = excel_data.drop_duplicates()
    else:
        print("No duplicate rows found")

    # 4. Validate temperature relationships (TMAX ≥ TAVG ≥ TMIN)
    if all(col in excel_data.columns for col in ['TMAX', 'TAVG', 'TMIN']):
        invalid_temp = excel_data[(excel_data['TMAX'] < excel_data['TAVG']) | 
                            (excel_data['TAVG'] < excel_data['TMIN'])]
        if len(invalid_temp) > 0:
            print(f"Warning: {len(invalid_temp)} rows have invalid temperature relationships")
            # Option 1: Correct the values (TMAX = max, TMIN = min)
            excel_data['TMAX'] = excel_data[['TMAX', 'TAVG', 'TMIN']].max(axis=1)
            excel_data['TMIN'] = excel_data[['TMAX', 'TAVG', 'TMIN']].min(axis=1)
            print("Temperatures auto-corrected to maintain TMAX ≥ TAVG ≥ TMIN")
        else:
            print("All temperature relationships are valid (TMAX ≥ TAVG ≥ TMIN)")
    else:
        print("Warning: Temperature columns missing")

    # 5. Validate humidity range (0-100%)
    if 'Humidity (%)' in excel_data.columns:
        if (excel_data['Humidity (%)'] < 0).any() or (excel_data['Humidity (%)'] > 100).any():
            print("Adjusting humidity values to 0-100% range")
            excel_data['Humidity (%)'] = excel_data['Humidity (%)'].clip(0, 100)
        else:
            print("All humidity values are within valid range (0-100%)")
    else:
        print("Warning: 'Humidity (%)' column missing")

    # 6. Create time-based features with Indian season classification
    if 'Date' in excel_data.columns and pd.api.types.is_datetime64_any_dtype(excel_data['Date']):
        excel_data['Year'] = excel_data['Date'].dt.year
        excel_data['Month'] = excel_data['Date'].dt.month
    
        # Indian season classification
        def get_indian_season(month):
            if 3 <= month <= 5:   # March-May = Summer
                return 1
            elif 6 <= month <= 9: # June-Sept = Monsoon
                return 2
            else:                 # Oct-Feb = Winter
                return 3
    
        excel_data['Season'] = excel_data['Month'].apply(get_indian_season)
    
        # Verify season distribution
        print("\nSeason distribution:")
        print(excel_data['Season'].map(season_names).value_counts())
    
        print("Added time-based features with Indian season classification")
    else:
        print("Could not create time-based features - Date column missing or invalid")

    # 7. Final validation
    print("\nFinal dataset summary:")
    print(f"Rows: {len(excel_data)}, Columns: {len(excel_data.columns)}")
    print("\nData types:")
    print(excel_data.dtypes)

    # Save cleaned data
    excel_data.to_csv('cleaned_weather_data.csv', index=False)
    print("\nCleaned data saved to 'cleaned_weather_data.csv'")
    return excel_data

def train_and_evaluate_models(df):
    """Train and evaluate weather prediction models with 70-30 split"""
    print("\nTraining weather prediction models with 70-30 train-test split...")
    
    # Prepare data - include all cities in dummy variables
    df = pd.get_dummies(df, columns=['City'], drop_first=False)
    
    # Get all city columns
    city_cols = [col for col in df.columns if col.startswith('City_')]
    features = ['Month', 'Season', 'Humidity (%)', 'Wind (km/h)', 'Precip (mm)'] + city_cols
    
    targets = ['TAVG', 'TMAX', 'TMIN', 'Feels Like']
    
    models = {}
    evaluation_results = []
    
    for target in targets:
        X = df[features]
        y = df[target]
        
        # 70-30 train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        models[target] = model
        evaluation_results.append({
            'Target': target,
            'MAE': mae,
            'RMSE': rmse,
            'R2 Score': r2,
            'Train Samples': len(X_train),
            'Test Samples': len(X_test)
        })
    
    # Save models and evaluation results
    joblib.dump(models, 'weather_models.joblib')
    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv('model_evaluation_results.csv', index=False)
    
    print("\nModel Evaluation Results:")
    print(evaluation_df.to_string(index=False))
    print("\nModels saved to 'weather_models.joblib'")
    print("Evaluation results saved to 'model_evaluation_results.csv'")
    
    return models, evaluation_df

def predict_weather(models):
    """Interactive weather prediction interface"""
    print("\n=== Weather Prediction System ===")
    print(f"Available cities: {', '.join(cities)}")
    
    while True:
        try:
            city = input("\nEnter city: ").title()
            if city not in cities:
                print(f"Error: {city} is not in our database. Please choose from: {', '.join(cities)}")
                continue
                
            month = int(input("Enter month (1-12): "))
            humidity = float(input("Current humidity (%): "))
            wind = float(input("Current wind speed (km/h): "))
            precip = float(input("Current precipitation (mm): "))
            
            # Determine season
            season = 1 if 3 <= month <= 5 else (2 if 6 <= month <= 9 else 3)
            
            # Prepare input - create all city columns
            input_data = {
                'Month': [month],
                'Season': [season],
                'Humidity (%)': [humidity],
                'Wind (km/h)': [wind],
                'Precip (mm)': [precip]
            }
            
            # Add all city dummy variables (set to 1 for selected city, 0 for others)
            for c in cities:
                input_data[f'City_{c}'] = [1 if city == c else 0]
            
            # Ensure columns are in the same order as during training
            model_features = list(models['TAVG'].feature_names_in_)
            input_df = pd.DataFrame(input_data)[model_features]
            
            # Make predictions
            print("\n=== Prediction Results ===")
            print(f"Location: {city} | Month: {month} ({season_names[season]})")
            print("-------------------------")
            for target, model in models.items():
                pred = model.predict(input_df)[0]
                print(f"{target}: {pred:.1f}°C")
            
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
                
        except Exception as e:
            print(f"Error: {e}\nPlease try again.")

def main():
    """Main program flow"""
    # Load or preprocess data
    if os.path.exists('cleaned_weather_data.csv'):
        df = pd.read_csv('cleaned_weather_data.csv')
        print("Loaded preprocessed data")
    else:
        df = preprocess_data()
    
    # Load or train models
    if os.path.exists('weather_models.joblib'):
        models = joblib.load('weather_models.joblib')
        print("\nLoaded trained weather models")
        
        # Load evaluation results if available
        if os.path.exists('model_evaluation_results.csv'):
            eval_results = pd.read_csv('model_evaluation_results.csv')
            print("\nPrevious Model Evaluation Results:")
            print(eval_results.to_string(index=False))
    else:
        models, eval_results = train_and_evaluate_models(df)
    
    # Run prediction interface
    predict_weather(models)
    print("\nWeather prediction system closed.")

if __name__ == "__main__":
    main()