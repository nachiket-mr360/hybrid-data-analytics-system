"""
Data Cleaning Module
This module will contain functions for cleaning and preprocessing datasets
"""
import pandas as pd

def remove_duplicates(df):
    """Remove duplicate rows from the dataframe"""
    return df.drop_duplicates()

def handle_missing_values(df):
    """Handle missing values in the dataframe"""
    # For now, fill numeric columns with mean and categorical with mode
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].dtype in ['int64', 'float64']:
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        else:
            mode_value = df_copy[column].mode()[0] if not df_copy[column].mode().empty else ''
            df_copy[column] = df_copy[column].fillna(mode_value)
    return df_copy

def basic_cleaning(df):
    """Perform basic cleaning operations on the dataframe"""
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    return df