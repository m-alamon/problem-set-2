"""
Name: Matthew Alamon
PART 1: ETL
"""
import pandas as pd
import os

def create_data_directory(directory='data'):
    """
    Creates a directory if it doesn't exist.
    """
    os.makedirs(directory, exist_ok=True)

def load_data():
    """
    Loads the raw datasets from the provided URLs.
    """
    pred_universe_url = 'https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1'
    arrest_events_url = 'https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1'
    
    pred_universe_raw = pd.read_csv(pred_universe_url)
    arrest_events_raw = pd.read_csv(arrest_events_url)
    
    return pred_universe_raw, arrest_events_raw

def preprocess_data(pred_universe_raw, arrest_events_raw):
    """
    Preprocessing for the raw datasets, as this solved an error I had regarding 
    the formatting of the dates present on both datasets.
    """
    # Converts 'filing_date' to datetime and update column names.
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['filing_date'])
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['filing_date'])
    
    # Drops the original 'filing_date' column to prevent redundancy.
    pred_universe_raw.drop(columns=['filing_date'], inplace=True)
    arrest_events_raw.drop(columns=['filing_date'], inplace=True)
    
    return pred_universe_raw, arrest_events_raw

def save_data(pred_universe_raw, arrest_events_raw, directory='data'):
    """
    Saves the DataFrames to CSV files in the specified directory.
    """
    pred_universe_raw.to_csv(f'{directory}/pred_universe_raw.csv', index=False)
    arrest_events_raw.to_csv(f'{directory}/arrest_events_raw.csv', index=False)
    print("Dataframes saved as arrest_events_raw.csv and pred_universe_raw.csv.")

def main():
    create_data_directory()
    pred_universe_raw, arrest_events_raw = load_data()
    pred_universe_raw, arrest_events_raw = preprocess_data(pred_universe_raw, arrest_events_raw)
    save_data(pred_universe_raw, arrest_events_raw)

if __name__ == "__main__":
    main()
