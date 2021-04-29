import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load multiple data sources from csv files
    and combine into a single dataframe

    Parameters
    ----------
    messages_filepath : string
        Path to the csv file containing categories data
    categories_filepath : string
        Path to the csv file containing categories data 

    Returns
    -------
    df : Pandas DataFrame
        Combined data from messages and categories csv files

    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """
    Cleaned the provided dataframe by creating columns
    for each category and removing duplicate rows

    Parameters
    ----------
    df : Pandas DataFrame
        Raw, unformatted data

    Returns
    -------
    df : Pandas DataFrame
        Cleaned data to be saved in a database

    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    
    # get categories as column names 
    row = categories.iloc[0,:]
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames
    
    # create columns for each category
    # columns are of type integer and are either 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column
    # and combine with the formatted categories data
    df.drop(columns=["categories"], inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicate rows
    df.drop_duplicates(inplace=True)
    
    df['related'] = df['related'].map({0:0, 1:1, 2:1})
    
    return df


def save_data(df, database_filename):
    """
    Save the dataframe to a database

    Parameters
    ----------
    df : Pandas DataFrame
        Clean data to be saved in the database
    database_filename : string
        filename for the database

    Returns
    -------
    None.

    """
    
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("MessageCategories", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
