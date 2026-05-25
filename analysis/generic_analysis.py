"""
Generic Analysis Module
This module will contain functions for analyzing generic datasets (EDA)
"""
import pandas as pd
import numpy as np
from analysis.visualizations import select_top_numeric_columns, select_top_categorical_columns, generate_numeric_visualizations, generate_categorical_visualizations


def dataset_overview(df):
    """Return basic dataset information"""
    overview = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns)
    }
    return overview


def data_types(df):
    """Return data types of each column"""
    dtypes = df.dtypes.to_dict()
    return dtypes


def missing_values(df):
    """Return count of missing values per column"""
    missing = df.isnull().sum().to_dict()
    return missing


def basic_statistics(df):
    """Return basic statistical information"""
    # Only include numeric columns for describe
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe()
        return stats.to_dict()
    else:
        return {}


def correlation_analysis(df):
    """Generate correlation matrix for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        return corr_matrix.to_dict()
    else:
        return {}


def distribution_info(df):
    """Identify numeric columns for plotting"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def generate_visualizations(df):
    """Generate visualizations for the dataset"""
    # Select top numeric and categorical columns for visualization
    top_numeric_cols = select_top_numeric_columns(df, n=3)  # Limit to 3 for cleaner output
    top_categorical_cols = select_top_categorical_columns(df, n=2)  # Limit to 2 for cleaner output
    
    # Generate visualizations
    numeric_charts = generate_numeric_visualizations(df, top_numeric_cols)
    categorical_charts = generate_categorical_visualizations(df, top_categorical_cols)
    
    return {
        'numeric_charts': numeric_charts,
        'categorical_charts': categorical_charts,
        'selected_numeric_columns': top_numeric_cols,
        'selected_categorical_columns': top_categorical_cols
    }


def run_generic_analysis(df):
    """Main function to run all generic analyses"""
    results = {}
    
    # Run all analysis functions
    results['overview'] = dataset_overview(df)
    results['data_types'] = data_types(df)
    results['missing_values'] = missing_values(df)
    results['statistics'] = basic_statistics(df)
    results['correlation'] = correlation_analysis(df)
    results['distribution_columns'] = distribution_info(df)
    
    # Add visualizations
    results['visualizations'] = generate_visualizations(df)
    
    return results