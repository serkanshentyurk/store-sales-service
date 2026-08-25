import pandas as pd

def load_clean_data(path:str='data') -> pd.DataFrame:
    """
    Load and clean the data from the specified path.
    
    Args:
		path (str): The path to the directory containing the CSV files.

    Returns:
		pd.DataFrame: A cleaned DataFrame containing the merged and filtered data.
    """
    # Load the store and train data from CSV files
    store_df = pd.read_csv(f'{path}/store.csv')
    train_df = pd.read_csv(f'{path}/train.csv', dtype={'StateHoliday': str}) # Ensure StateHoliday is read as string to avoid issues with mixed types
    
    # Merge the train and store data on the 'Store' column
    df_full = train_df.merge(store_df, on='Store', how='left').reset_index(drop=True)   
     
    # Keep only the rows where the store is open and sales are greater than zero
    df = df_full[(df_full['Open'] == 1) & (df_full['Sales'] > 0)]
    
    # Drop the 'Customers' column as it is not needed for further analysis
    df = df.drop(columns=['Customers'])
    return df