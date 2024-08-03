'''
Name: Matthew Alamon
PART 2: Pre-processing
'''

# import the necessary packages
import pandas as pd

def load_data():
    """
    Loads the datasets from the CSV files created in Part 1.
    """
    pred_universe = pd.read_csv('data/pred_universe_raw.csv')
    arrest_events = pd.read_csv('data/arrest_events_raw.csv')
    
    return pred_universe, arrest_events

def merge_datasets(pred_universe, arrest_events):
    """
    Performs a full outer join on 'person_id'.
    """
    # Strips spaces from column names for consistency purposes.
    pred_universe.columns = pred_universe.columns.str.strip()
    arrest_events.columns = arrest_events.columns.str.strip()
    
    # Ensuring that person_id is the same type in both DataFrames.
    pred_universe['person_id'] = pred_universe['person_id'].astype(str)
    arrest_events['person_id'] = arrest_events['person_id'].astype(str)
    
    df_arrests = pd.merge(pred_universe, arrest_events, on='person_id', how='outer', suffixes=('_univ', '_event'))
    
    return df_arrests

def preprocess_dates(df_arrests):
    """
    Ensures 'arrest_date_univ' and 'arrest_date_event' are datetime to 
    handle an error I was facing related to the two date columns found in the dataset.
    """
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'], errors='coerce')
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'], errors='coerce')
    return df_arrests

def create_felony_in_next_year_column(df_arrests, arrest_events):
    """
    Creates the 'y' column indicating if a person was rearrested for a felony in the next year.
    """
    def felony_in_next_year(row):
        if pd.isnull(row['arrest_date_univ']):
            return 0
        end_date = row['arrest_date_univ'] + pd.DateOffset(years=1)
        
        arrest_events_filtered = arrest_events[
            (arrest_events['person_id'] == row['person_id']) &
            (arrest_events['arrest_date_event'] > row['arrest_date_univ']) &
            (arrest_events['arrest_date_event'] <= end_date) &
            (arrest_events['charge_degree'].str.lower() == 'felony')  # Ensure consistent casing
        ]
        
        return 1 if not arrest_events_filtered.empty else 0

    df_arrests['y'] = df_arrests.apply(felony_in_next_year, axis=1)
    return df_arrests

def create_current_charge_felony_column(df_arrests):
    """
    Creates a column indicating if the current charge is a felony.
    """
    if 'charge_degree' not in df_arrests.columns:
        raise KeyError("Column 'charge_degree' does not exist in df_arrests DataFrame")
    
    # Ensures consistent casing for tidying purposes.
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].str.lower().apply(lambda x: 1 if x == 'felony' else 0)
    
    return df_arrests

def num_fel_arrests_last_year(row, arrest_events):
    """
    Counts the number of felony arrests in the last year for a given person_id.
    """
    if pd.isnull(row['arrest_date_univ']):
        return 0
    
    one_year_ago = row['arrest_date_univ'] - pd.DateOffset(years=1)
    
    num_felony_arrests = arrest_events[
        (arrest_events['person_id'] == row['person_id']) &
        (arrest_events['arrest_date_event'] > one_year_ago) &
        (arrest_events['arrest_date_event'] <= row['arrest_date_univ']) &
        (arrest_events['charge_degree'].str.lower() == 'felony')  # Ensure consistent casing for tidying purposes.
    ].shape[0]
    
    return num_felony_arrests
    
def create_num_fel_arrests_last_year_column(df_arrests, arrest_events):
    """
    Creates a column indicating the number of felony arrests in the last year.
    """
    if 'arrest_date_univ' not in df_arrests.columns or 'person_id' not in df_arrests.columns:
        raise KeyError("Required columns for creating 'num_fel_arrests_last_year' do not exist in df_arrests DataFrame")
    
    if 'arrest_date_event' not in arrest_events.columns or 'person_id' not in arrest_events.columns or 'charge_degree' not in arrest_events.columns:
        raise KeyError("Required columns for 'arrest_events' DataFrame do not exist")
    
    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(lambda row: num_fel_arrests_last_year(row, arrest_events), axis=1)
    
    return df_arrests

def print_statistics(df_arrests):
    """
    Prints required statistics.
    """
    share_rearrested_felony = df_arrests['y'].mean()
    print(f"What share of arrestees in the df_arrests table were rearrested for a felony crime in the next year? {share_rearrested_felony:.2%}")
    
    share_current_felony = df_arrests['current_charge_felony'].mean()
    print(f"What share of current charges are felonies? {share_current_felony:.2%}")
    
    average_fel_arrests_last_year = df_arrests['num_fel_arrests_last_year'].mean()
    print(f"What is the average number of felony arrests in the last year? {average_fel_arrests_last_year:.2f}")

def save_and_print_df(df_arrests):
    """Save df_arrests to CSV and print the first few rows."""
    df_arrests.to_csv('data/df_arrests.csv', index=False)
    print("Dataframe saved as df_arrests.csv.")

def main():
    pred_universe, arrest_events = load_data()
    df_arrests = merge_datasets(pred_universe, arrest_events)
    df_arrests = preprocess_dates(df_arrests)    
    df_arrests = create_felony_in_next_year_column(df_arrests, arrest_events)
    df_arrests = create_current_charge_felony_column(df_arrests)
    df_arrests = create_num_fel_arrests_last_year_column(df_arrests, arrest_events)
    print_statistics(df_arrests)
    save_and_print_df(df_arrests)

if __name__ == "__main__":
    main()
