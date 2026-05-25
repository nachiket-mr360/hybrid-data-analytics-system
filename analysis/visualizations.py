import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np

def generate_histogram(data, column_name, figsize=(8, 6)):
    """Generate histogram for a numeric column"""
    plt.figure(figsize=figsize)
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # Close the figure to free memory
    
    return img_str

def generate_boxplot(data, column_name, figsize=(8, 6)):
    """Generate boxplot for a numeric column"""
    plt.figure(figsize=figsize)
    plt.boxplot(data.dropna())  # Drop NaN values for boxplot
    plt.title(f'Box Plot of {column_name}')
    plt.ylabel(column_name)
    plt.tight_layout()
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # Close the figure to free memory
    
    return img_str

def generate_bar_chart(data, column_name, top_n=10, figsize=(10, 6)):
    """Generate bar chart for a categorical column (top N values)"""
    value_counts = data.value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(value_counts)), value_counts.values)
    plt.title(f'Top {top_n} Categories of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, value_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(value),
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()  # Close the figure to free memory
    
    return img_str

def select_top_numeric_columns(df, n=5):
    """Select top numeric columns based on variance (potential for insight)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) <= n:
        return numeric_cols
    
    # Calculate variance for each numeric column
    variances = []
    for col in numeric_cols:
        var_val = df[col].var()
        if pd.notna(var_val):  # Skip if variance is NaN
            variances.append((col, var_val))
    
    # Sort by variance descending and return top n
    variances.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in variances[:n]]

def select_top_categorical_columns(df, n=3):
    """Select top categorical columns based on distinct value count and frequency"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) <= n:
        return categorical_cols
    
    # Calculate a score based on number of unique values and frequency of most common
    scores = []
    for col in categorical_cols:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        if total_count > 0:
            # Score based on having moderate number of unique values (not too many, not too few)
            # and good distribution (not too skewed)
            most_common_ratio = df[col].value_counts().iloc[0] / total_count
            score = unique_count * (1 - most_common_ratio)  # Higher score for more balanced distributions
            scores.append((col, score))
    
    # Sort by score descending and return top n
    scores.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in scores[:n]]

def generate_numeric_visualizations(df, numeric_cols):
    """Generate visualizations for numeric columns"""
    charts = []
    
    for col in numeric_cols:
        if df[col].dtype in [np.float64, np.int64, np.int32, np.float32]:
            # Generate histogram
            hist_img = generate_histogram(df[col].dropna(), col)
            charts.append({
                'type': 'histogram',
                'column': col,
                'title': f'Distribution of {col}',
                'image_data': hist_img
            })
            
            # Generate boxplot
            box_img = generate_boxplot(df[col].dropna(), col)
            charts.append({
                'type': 'boxplot',
                'column': col,
                'title': f'Box Plot of {col}',
                'image_data': box_img
            })
    
    return charts

def generate_categorical_visualizations(df, categorical_cols):
    """Generate visualizations for categorical columns"""
    charts = []
    
    for col in categorical_cols:
        # Generate bar chart for top categories
        bar_img = generate_bar_chart(df[col].dropna(), col)
        charts.append({
            'type': 'bar',
            'column': col,
            'title': f'Top Categories of {col}',
            'image_data': bar_img
        })
    
    return charts