import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import joblib
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

def train_random_forest_by_station():
    # Create directory for models if it doesn't exist
    models_dir = Path("random_forest_station_models")
    models_dir.mkdir(exist_ok=True)
    
    # Load dataset
    df = pd.read_csv("noise_data_set.csv").dropna()
    
    # Dictionary to store model information for all stations
    station_models_info = {}
    
    # Get unique stations
    stations = df['Station'].unique()
    print(f"Found {len(stations)} unique stations: {stations}")
    
    for station in stations:
        print(f"\n{'='*60}")
        print(f"Training model for station: {station}")
        print('='*60)
        
        # Filter data for this station
        station_data = df[df['Station'] == station].copy()
        
        if len(station_data) < 10:  # Minimum data points threshold
            print(f"Warning: Station {station} has only {len(station_data)} samples. Skipping...")
            continue
        
        # Prepare features and target variables
        # We'll use Year and Month as features for each station
        X = station_data[['Year', 'Month']]
        y = station_data[['Day', 'Night']]
        
        # Check if we have enough data for splitting
        if len(X) < 5:
            print(f"Warning: Not enough data for station {station}. Using all data for training.")
            X_train, X_test, y_train, y_test = X, X, y, y
            train_test_same = True
        else:
            # Split data (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            train_test_same = False
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,  # Limit depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        rf_model.fit(X_train, y_train)
        
        # Calculate R² scores
        if train_test_same:
            # If train and test are the same, use cross-validation or report training scores
            y_pred_train = rf_model.predict(X_train)
            r2_day = r2_score(y_train['Day'], y_pred_train[:, 0])
            r2_night = r2_score(y_train['Night'], y_pred_train[:, 1])
            print(f"Training R² - Day: {r2_day:.4f}, Night: {r2_night:.4f}")
        else:
            y_pred_test = rf_model.predict(X_test)
            r2_day = r2_score(y_test['Day'], y_pred_test[:, 0])
            r2_night = r2_score(y_test['Night'], y_pred_test[:, 1])
            print(f"Test R² - Day: {r2_day:.4f}, Night: {r2_night:.4f}")
        
        # Calculate feature importance
        feature_importance = rf_model.feature_importances_
        
        # Save model for this station
        model_filename = models_dir / f"rf_model_{station.replace(' ', '_').replace('/', '_')}.joblib"
        joblib.dump(rf_model, model_filename)
        
        # Store model information
        station_models_info[station] = {
            'model_path': str(model_filename),
            'r2_day': r2_day,
            'r2_night': r2_night,
            'n_samples': len(station_data),
            'years_available': sorted(station_data['Year'].unique()),
            'months_available': sorted(station_data['Month'].unique()),
            'feature_importance': {
                'Year': feature_importance[0],
                'Month': feature_importance[1]
            },
            'train_size': len(X_train),
            'test_size': len(X_test) if not train_test_same else 0
        }
        
        print(f"Model saved: {model_filename}")
        print(f"Samples: {len(station_data)}")
        print(f"Years available: {sorted(station_data['Year'].unique())}")
        print(f"Months available: {sorted(station_data['Month'].unique())}")
    
    # Save station models information to a metadata file
    metadata = {
        'stations_trained': list(station_models_info.keys()),
        'models_info': station_models_info,
        'total_stations': len(station_models_info),
        'data_summary': {
            'total_samples': len(df),
            'stations_with_models': len(station_models_info),
            'all_stations': list(stations)
        }
    }
    
    joblib.dump(metadata, models_dir / "station_models_metadata.joblib")
    
    # Create a summary CSV file
    summary_data = []
    for station, info in station_models_info.items():
        summary_data.append({
            'Station': station,
            'R2_Day': info['r2_day'],
            'R2_Night': info['r2_night'],
            'Samples': info['n_samples'],
            'Train_Size': info['train_size'],
            'Test_Size': info['test_size'],
            'Model_File': info['model_path']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(models_dir / "model_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total stations with models: {len(station_models_info)}")
    print(f"Average R² Day: {summary_df['R2_Day'].mean():.4f}")
    print(f"Average R² Night: {summary_df['R2_Night'].mean():.4f}")
    
    if len(station_models_info) > 0:
        print("\nTop 5 stations by Day R²:")
        print(summary_df.nlargest(5, 'R2_Day')[['Station', 'R2_Day', 'Samples']].to_string(index=False))
        
        print("\nTop 5 stations by Night R²:")
        print(summary_df.nlargest(5, 'R2_Night')[['Station', 'R2_Night', 'Samples']].to_string(index=False))
    
    return station_models_info, summary_df

if __name__ == "__main__":
    models_info, summary_df = train_random_forest_by_station()
    
    # Print all trained stations
    print("\nAll trained stations:")
    for station in models_info.keys():
        print(f"  - {station}")