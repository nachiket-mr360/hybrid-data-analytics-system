"""
Dynamic Insights Module
This module will generate automated insights for ANY dataset
"""
import pandas as pd
import numpy as np


def generate_numeric_kpis(df):
    """Generate KPIs for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    kpis = {}
    
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            kpis[col] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'count': int(len(col_data)),
                'missing_count': int(df[col].isna().sum())
            }
    
    return kpis


def generate_categorical_kpis(df):
    """Generate KPIs for categorical columns"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    kpis = {}
    
    for col in categorical_cols:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            value_counts = col_data.value_counts()
            most_frequent = value_counts.index[0] if len(value_counts) > 0 else None
            most_frequent_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            kpis[col] = {
                'unique_values': int(df[col].nunique()),
                'most_frequent': most_frequent,
                'most_frequent_count': int(most_frequent_count),
                'total_count': int(len(col_data)),
                'missing_count': int(df[col].isna().sum())
            }
    
    return kpis


def generate_correlation_insight(df):
    """Generate correlation insights for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find the highest correlation pair (excluding self-correlations)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = abs(corr_matrix.iloc[i, j])
            high_corr_pairs.append((col1, col2, corr_val))
    
    if high_corr_pairs:
        # Sort by correlation value (descending)
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        strongest_pair = high_corr_pairs[0]
        return {
            'feature1': strongest_pair[0],
            'feature2': strongest_pair[1],
            'correlation': float(abs(strongest_pair[2])),
            'raw_correlation': float(corr_matrix.loc[strongest_pair[0], strongest_pair[1]])
        }
    
    return None


def generate_summary(df):
    """Generate basic summary of the dataset"""
    total_missing = df.isnull().sum().sum()
    total_cells = df.size
    missing_percentage = (total_missing / total_cells * 100) if total_cells > 0 else 0
    
    summary = {
        'rows': int(len(df)),
        'columns': int(len(df.columns)),
        'total_cells': int(total_cells),
        'total_missing_values': int(total_missing),
        'missing_percentage': round(float(missing_percentage), 2),
        'numeric_columns': int(len(df.select_dtypes(include=[np.number]).columns)),
        'categorical_columns': int(len(df.select_dtypes(include=['object', 'category']).columns)),
        'column_names': list(df.columns)
    }
    
    return summary


def generate_dynamic_insights(df):
    """Main function to generate all dynamic insights"""
    results = {}
    
    # Generate all insights
    results['summary'] = generate_summary(df)
    results['numeric_kpis'] = generate_numeric_kpis(df)
    results['categorical_kpis'] = generate_categorical_kpis(df)
    results['correlation_insight'] = generate_correlation_insight(df)
    
    return results