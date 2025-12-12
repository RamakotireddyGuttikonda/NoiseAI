import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import joblib
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

def train_arima_simple():
    """
    Simple ARIMA training without optimization
    Just train ARIMA(1,0,1) for all stations
    """
    
    # Create directory for ARIMA models
    arima_dir = Path("arima_models")
    arima_dir.mkdir(exist_ok=True)
    
    # Load dataset
    csv_file_path = "noise_data_set.csv"
    data = pd.read_csv(csv_file_path).dropna()
    data.rename(columns={'Day': 'avg_day_value', 'Night': 'avg_night_value'}, inplace=True)
    
    # Sort data by Year and Month
    data['date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'].astype(str) + '-01')
    data = data.sort_values(['Station', 'date'])
    
    models_by_station = {}
    
    for station in data['Station'].unique():
        print(f"Training ARIMA for station: {station}")
        
        station_data = data[data['Station'] == station].copy()
        
        if len(station_data) < 5:  # Minimum data points
            print(f"  Skipping {station} - insufficient data ({len(station_data)} samples)")
            continue
        
        # Prepare time series data
        day_series = station_data['avg_day_value'].values
        night_series = station_data['avg_night_value'].values
        
        # Train simple ARIMA models (using fixed order ARIMA(1,0,1))
        try:
            # Day model
            day_model = ARIMA(day_series, order=(1, 0, 1)).fit()
        except:
            # If ARIMA(1,0,1) fails, try simpler ARIMA(1,0,0)
            try:
                day_model = ARIMA(day_series, order=(1, 0, 0)).fit()
            except:
                day_model = None
        
        try:
            # Night model
            night_model = ARIMA(night_series, order=(1, 0, 1)).fit()
        except:
            # If ARIMA(1,0,1) fails, try simpler ARIMA(1,0,0)
            try:
                night_model = ARIMA(night_series, order=(1, 0, 0)).fit()
            except:
                night_model = None
        
        # Save model data
        station_results = {
            'station': station,
            'day_model': day_model,
            'night_model': night_model,
            'day_data': day_series.tolist(),
            'night_data': night_series.tolist(),
            'n_samples': len(station_data),
            'last_date': station_data['date'].max().strftime('%Y-%m')
        }
        
        # Save individual station model
        model_filename = arima_dir / f"arima_{station}.joblib"
        joblib.dump(station_results, model_filename)
        
        models_by_station[station] = station_results
        print(f"  Saved model for {station}")
    
    # Save all models together for easy loading
    all_models_file = arima_dir / "all_arima_models.joblib"
    joblib.dump(models_by_station, all_models_file)
    
    print(f"\nARIMA Training Complete!")
    print(f"Trained models for {len(models_by_station)} stations")
    print(f"Models saved in: {arima_dir}")
    
    return models_by_station

if __name__ == "__main__":
    models = train_arima_simple()