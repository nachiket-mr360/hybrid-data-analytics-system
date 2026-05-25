from flask import Flask, request, redirect, url_for, render_template, session
from werkzeug.utils import secure_filename
import os
import traceback
from config import Config, allowed_file
import pandas as pd
from analysis.data_cleaning import basic_cleaning
from analysis.retail_analysis import run_retail_analysis, prepare_time_series_data
from analysis.generic_analysis import run_generic_analysis
from analysis.dynamic_insights import generate_dynamic_insights
from analysis.insight_engine import generate_retail_insights, get_insight_cards
from utils.column_mapper import map_columns
from utils.data_storage import temp_manager



os.makedirs("uploads", exist_ok=True)

def read_csv_safe(filepath):
    """
    Safely read CSV file with multiple encoding fallbacks
    """
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"File loaded using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not a UnicodeDecodeError, continue to next encoding
            if "UnicodeDecodeError" in str(type(e).__name__):
                continue
            else:
                continue
    
    # If all encodings fail, raise an exception
    raise Exception("Could not read file with any of the attempted encodings: utf-8, latin1, ISO-8859-1, cp1252")


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Secret key for session
app.config.from_object(Config)

def get_meaningful_categorical_columns(df, max_columns=5):
    """
    Get categorical columns that have meaningful filter options
    Only include columns with 2-20 unique values (not IDs or timestamps)
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    meaningful_cols = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        
        # Only include columns with 2-20 unique values (not IDs or timestamps)
        # Exclude columns that are mostly unique (likely IDs)
        if 2 <= unique_count <= 20 and unique_count < total_count * 0.8:  # Less than 80% unique
            meaningful_cols.append(col)
    
    # Limit to top N columns to avoid overload
    return meaningful_cols[:max_columns]

def get_unique_values(df, column):
    """
    Get unique values for a specific column
    """
    return df[column].dropna().unique().tolist()

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Apply row-wise filters to a pandas DataFrame.
    
    Args:
        df: pandas DataFrame to filter
        filters: dict mapping column names to filter values
    
    Returns:
        Filtered pandas DataFrame
    """
    result = df.copy()
    
    for col, val in filters.items():
        if val and val != "All" and col in result.columns:
            result = result[result[col] == val]
    
    return result

def prettify_column_name(name):
    """
    Convert column name to a more readable format
    e.g. 'ship_mode' -> 'Ship Mode'
    """
    # Replace underscores with spaces and capitalize each word
    pretty = name.replace('_', ' ').replace('-', ' ').title()
    return pretty

def detect_extended_fields(columns):
    """
    Auto-detect potential extended business dimension fields based on common naming patterns
    """
    extended_fields = {}
    
    # Define patterns for different types of extended fields
    patterns = {
        'sub_category': ['subcat', 'subcategory', 'sub_category', 'sub cat'],
        'region': ['region', 'reg', 'area', 'territory', 'zone'],
        'segment': ['segment', 'seg', 'customer_segment', 'cust_seg'],
        'city': ['city', 'town', 'location'],
        'state': ['state', 'province', 'county', 'region_state'],
        'customer_type': ['customer_type', 'cust_type', 'customer_class', 'type'],
        'brand': ['brand', 'manufacturer', 'maker'],
        'supplier': ['supplier', 'vendor', 'provider'],
        'department': ['department', 'dept', 'division'],
        'channel': ['channel', 'source', 'platform', 'medium'],
        'season': ['season', 'quarter', 'period'],
        'team': ['team', 'group', 'unit', 'department'],
        'store_type': ['store_type', 'store_format', 'outlet_type'],
        'payment_method': ['payment', 'method', 'payment_method', 'pay_type']
    }
    
    for field_type, patterns_list in patterns.items():
        for col in columns:
            col_lower = col.lower()
            for pattern in patterns_list:
                if pattern in col_lower:
                    extended_fields[field_type] = col
                    break  # Move to next column after first match
    
    return extended_fields

def get_extended_field_options():
    """
    Return options for extended fields dropdown
    """
    return {
        'sub_category': 'Sub-Category',
        'region': 'Region',
        'segment': 'Segment',
        'city': 'City',
        'state': 'State',
        'customer_type': 'Customer Type',
        'brand': 'Brand',
        'supplier': 'Supplier',
        'department': 'Department',
        'channel': 'Channel',
        'season': 'Season',
        'team': 'Team',
        'store_type': 'Store Type',
        'payment_method': 'Payment Method'
    }

@app.route('/')
def home():
    # Landing page - modern homepage with project overview
    return render_template('landing_page.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # Handle GET request (initial page load - pure upload page, NO columns required)
    if request.method == 'GET':
        return render_template('upload_page.html')
    
    # Handle POST request (file upload)
    # Check if the post request has the file part
    if 'file' not in request.files:
        return "No file selected"
    
    file = request.files['file']
    
    # If user does not select file, browser submits empty part without filename
    if file.filename == '' or file.filename is None:
        return "No file selected"
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Store the file path in session
        session['uploaded_file_path'] = filepath
        session['original_filename'] = filename
        
        # Load the CSV file to get column names
        try:
            df = read_csv_safe(filepath)
            columns = list(df.columns)
            
            # Detect extended fields automatically
            extended_detected = detect_extended_fields(columns)
            
            # Get meaningful categorical columns for filtering (2-20 unique values)
            categorical_cols = get_meaningful_categorical_columns(df)
            categorical_options = {}
            categorical_labels = {}
            for col in categorical_cols:
                categorical_options[col] = get_unique_values(df, col)
                categorical_labels[col] = prettify_column_name(col)
            
            # Get extended field options
            extended_field_options = get_extended_field_options()
            
            # Render the choose_analysis template with columns and all metadata
            return render_template('choose_analysis.html', 
                                 columns=columns, 
                                 filename=filename,
                                 categorical_columns=categorical_cols,
                                 categorical_options=categorical_options,
                                 categorical_labels=categorical_labels,
                                 extended_detected=extended_detected,
                                 extended_field_options=extended_field_options)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    else:
        return "Invalid file type. Only CSV files are allowed."

@app.route('/choose_analysis', methods=['GET', 'POST'])
def choose_analysis():
    # Handle GET request - redirect to upload if no file uploaded yet
    if request.method == 'GET':
        # Check if file is already uploaded
        if 'uploaded_file_path' not in session:
            return redirect(url_for('upload_file'))
        
        # If file is uploaded, get columns and show choose_analysis page
        filepath = session.get('uploaded_file_path')
        filename = session.get('original_filename')
        
        try:
            df = read_csv_safe(filepath)
            columns = list(df.columns)
            
            # Detect extended fields automatically
            extended_detected = detect_extended_fields(columns)
            
            # Get meaningful categorical columns for filtering (2-20 unique values)
            categorical_cols = get_meaningful_categorical_columns(df)
            categorical_options = {}
            categorical_labels = {}
            for col in categorical_cols:
                categorical_options[col] = get_unique_values(df, col)
                categorical_labels[col] = prettify_column_name(col)
            
            # Get extended field options
            extended_field_options = get_extended_field_options()
            
            return render_template('choose_analysis.html', 
                                 columns=columns, 
                                 filename=filename,
                                 categorical_columns=categorical_cols,
                                 categorical_options=categorical_options,
                                 categorical_labels=categorical_labels,
                                 extended_detected=extended_detected,
                                 extended_field_options=extended_field_options)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    # Handle POST request (analysis type selection)
    """Route to handle analysis type selection"""
    analysis_type = request.form.get('analysis_type')
    session['analysis_type'] = analysis_type
    
    if analysis_type == 'retail':
        # For retail, go to mapping page
        filepath = session.get('uploaded_file_path')
        filename = session.get('original_filename')
        if not filepath:
            return "Error: No file uploaded. Please upload a file first."
        
        try:
            df = read_csv_safe(filepath)
            columns = list(df.columns)
            
            # Detect extended fields automatically
            extended_detected = detect_extended_fields(columns)
            
            # Get meaningful categorical columns for filtering (2-20 unique values)
            categorical_cols = get_meaningful_categorical_columns(df)
            categorical_options = {}
            categorical_labels = {}
            for col in categorical_cols:
                categorical_options[col] = get_unique_values(df, col)
                categorical_labels[col] = prettify_column_name(col)
            
            # Get extended field options
            extended_field_options = get_extended_field_options()
            
            return render_template('index.html', 
                                 columns=columns, 
                                 filename=filename,
                                 categorical_columns=categorical_cols,
                                 categorical_options=categorical_options,
                                 categorical_labels=categorical_labels,
                                 extended_detected=extended_detected,
                                 extended_field_options=extended_field_options)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    elif analysis_type == 'generic':
        # For generic, go directly to analysis
        return redirect(url_for('analyze_generic_file'))
    else:
        return "Invalid analysis type selected."

def process_retail_data(df, filters, user_mapping):
    """Process data using retail pipeline"""
    # Apply filters if any are provided
    if filters:
        df = apply_filters(df, filters)
        print("Filters applied")
        print("Filtered DataFrame Shape:", df.shape)
    
    # Perform basic cleaning
    cleaned_df = basic_cleaning(df.copy())
    print("Cleaning done")
    print("Cleaned DataFrame Head:")
    print(cleaned_df.head())
    print("Cleaned DataFrame Shape:", cleaned_df.shape)
    
    # Apply user-defined column mapping
    # Keep all original columns and only rename mapped ones
    mapped_df = cleaned_df.copy()
    
    print("\n" + "="*80)
    print("COLUMN MAPPING DEBUG")
    print("="*80)
    print(f"BEFORE MAPPING - Columns: {list(mapped_df.columns)}")
    print(f"User mapping received: {user_mapping}")
    
    if user_mapping:
        # Only rename columns that exist in the dataframe and are in the user_mapping
        columns_to_rename = {k: v for k, v in user_mapping.items() if k in mapped_df.columns}
        print(f"Columns that will be renamed: {columns_to_rename}")
        
        if columns_to_rename:
            mapped_df = mapped_df.rename(columns=columns_to_rename)
            print(f"✓ Applied user mapping: {columns_to_rename}")
        else:
            print(f"⚠️ No columns matched for renaming!")
            print(f"  User mapping keys: {list(user_mapping.keys())}")
            print(f"  DataFrame columns: {list(mapped_df.columns)}")
    else:
        print("⚠️ No user mapping provided!")
    
    print(f"AFTER MAPPING - Columns: {list(mapped_df.columns)}")
    print("="*80 + "\n")
    
    print("Mapping applied")
    print("Columns after mapping:", list(mapped_df.columns))
    
    # STEP 1: ENSURE NUMERIC DATA TYPES
    print("\n=== DATA VALIDATION & PREPARATION ===")
    print("Step 1: Converting numeric columns...")
    
    numeric_columns = ['Quantity', 'Selling_Price', 'Cost_Price']
    for col in numeric_columns:
        if col in mapped_df.columns:
            print(f"  Converting {col} to numeric...")
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce')
            null_count = mapped_df[col].isnull().sum()
            print(f"  {col}: {null_count} invalid values converted to NaN")
    
    # STEP 2: DROP INVALID ROWS
    print("\nStep 2: Dropping rows with invalid numeric data...")
    rows_before = len(mapped_df)
    mapped_df = mapped_df.dropna(subset=['Quantity', 'Selling_Price', 'Cost_Price'])
    rows_dropped = rows_before - len(mapped_df)
    print(f"  Dropped {rows_dropped} rows with invalid data")
    print(f"  Remaining rows: {len(mapped_df)}")
    
    # STEP 3: CALCULATE REVENUE AND PROFIT
    print("\nStep 3: Calculating Revenue and Profit...")
    
    if 'Revenue' not in mapped_df.columns:
        print("  Revenue column not found - calculating...")
        mapped_df['Revenue'] = mapped_df['Quantity'] * mapped_df['Selling_Price']
        print(f"  ✓ Revenue calculated: Range ${mapped_df['Revenue'].min():.2f} - ${mapped_df['Revenue'].max():.2f}")
    else:
        print("  ✓ Revenue column already exists")
        # Verify it's numeric
        mapped_df['Revenue'] = pd.to_numeric(mapped_df['Revenue'], errors='coerce')
    
    if 'Profit' not in mapped_df.columns:
        print("  Profit column not found - calculating...")
        if 'Cost_Price' in mapped_df.columns:
            mapped_df['Profit'] = (mapped_df['Selling_Price'] - mapped_df['Cost_Price']) * mapped_df['Quantity']
            print(f"  ✓ Profit calculated: Range ${mapped_df['Profit'].min():.2f} - ${mapped_df['Profit'].max():.2f}")
        else:
            print("  ⚠️ Cost_Price not available - setting Profit = Revenue * 0.2 (estimated)")
            mapped_df['Profit'] = mapped_df['Revenue'] * 0.2
    else:
        print("  ✓ Profit column already exists")
        # Verify it's numeric
        mapped_df['Profit'] = pd.to_numeric(mapped_df['Profit'], errors='coerce')
    
    # STEP 4: VALIDATE REQUIRED COLUMNS
    print("\nStep 4: Validating required columns for insight engine...")
    required_cols = ['Store', 'Product', 'Category', 'Quantity', 'Revenue', 'Profit']
    available_cols = [col for col in required_cols if col in mapped_df.columns]
    missing_cols = [col for col in required_cols if col not in mapped_df.columns]
    
    print(f"  Required columns: {required_cols}")
    print(f"  Available: {available_cols} ({len(available_cols)}/{len(required_cols)})")
    if missing_cols:
        print(f"  ⚠️ Missing: {missing_cols}")
    
    # STEP 5: PRINT SAMPLE DATA FOR DEBUGGING
    print("\nStep 5: Sample data (first 3 rows):")
    print(mapped_df[available_cols].head(3))
    
    print("=== END DATA VALIDATION ===\n")
    
    # Generate dynamic insights from the original cleaned data
    print("Generating dynamic insights")
    dynamic_insights = generate_dynamic_insights(cleaned_df)
    print("Dynamic Insights Generated:", dynamic_insights['summary'])
    
    # Determine if this is a retail dataset by checking for key retail columns AFTER mapping
    retail_identifiers = ['Date', 'Store', 'Product', 'Category', 'Quantity', 'Selling_Price', 'Cost_Price']
    found_retail_cols = [col for col in retail_identifiers if col in mapped_df.columns]
    
    print(f"Found retail columns: {found_retail_cols}")
    
    # Run retail analysis if enough retail columns exist
    retail_results = None
    if len(found_retail_cols) >= 2:  # Changed to 2 to be more flexible
        print("Trying retail analysis...")
        try:
            retail_results = run_retail_analysis(mapped_df)
            print("Retail analysis completed")
            
            # Debug: Print available keys
            print("Retail keys:", list(retail_results.keys()) if retail_results else "None")
            
            # Safely print key metrics with fallbacks
            if retail_results:
                total_rev = retail_results.get('total_revenue', 'N/A')
                total_prof = retail_results.get('total_profit', 'N/A')
                print(f"Total Revenue: {total_rev}")
                print(f"Total Profit: {total_prof}")
                
                # Safely access insights
                if 'insights' in retail_results:
                    insights = retail_results['insights']
                    print(f"Best Store: {insights.get('best_store', 'N/A')}")
                    print(f"Top Product: {insights.get('top_product', 'N/A')}")
                else:
                    print("Insights: Not available")
            
            # Safely print each analysis result if it exists
            if 'store_performance' in retail_results and retail_results['store_performance'] is not None and not retail_results['store_performance'].empty:
                print(f"Store Performance:\n{retail_results['store_performance']}")
            else:
                print("Store Performance: Not calculated due to missing columns or empty dataframe")
            
            if 'top_products' in retail_results and retail_results['top_products'] is not None and not retail_results['top_products'].empty:
                print(f"Top Products:\n{retail_results['top_products'].head()}""")
            else:
                print("Top Products: Not calculated due to missing columns or empty dataframe")
            
            if 'category_performance' in retail_results and retail_results['category_performance'] is not None and not retail_results['category_performance'].empty:
                print(f"Category Performance:\n{retail_results['category_performance']}")
            else:
                print("Category Performance: Not calculated due to missing columns or empty dataframe")
            
            if 'sales_trend' in retail_results and retail_results['sales_trend'] is not None and not retail_results['sales_trend'].empty:
                print(f"Sales Trend:\n{retail_results['sales_trend']}")
            else:
                print("Sales Trend: Not calculated due to missing columns or empty dataframe")
            
        except Exception as e:
            print(f"Retail analysis failed: {str(e)}")
            traceback.print_exc()
            retail_results = None
    else:
        print("Insufficient retail columns for retail analysis, skipping...")
        retail_results = None

    # Run generic analysis on the full dataset (before or after mapping, depending on needs)
    # Using the mapped dataframe to have consistent column names
    print("Starting generic analysis")
    generic_results = run_generic_analysis(mapped_df)
    print("Generic Analysis Results:")
    print(f"Dataset Overview: {generic_results['overview']}")
    print(f"Missing Values: {generic_results['missing_values']}")
    print(f"Statistics: {generic_results['statistics']}")
    print(f"Correlation: {generic_results['correlation']}")
    print("Generic analysis completed")
    
    # Prepare time series data for ML forecasting (without implementing ML yet)
    print("Preparing time series data for ML forecasting...")
    ml_ts_result = prepare_time_series_data(mapped_df, user_mapping)
    
    # RETAIL INSIGHT GENERATION ENGINE: Generate business insights
    retail_insights = None
    retail_insight_cards = None
    try:
        print("\n=== RETAIL INSIGHT GENERATION ENGINE ===")
        print("Step 1: Checking if retail analysis completed...")
        
        # VALIDATION: Check if DataFrame is empty
        if mapped_df.empty:
            print("⚠️ WARNING: DataFrame is empty after preprocessing!")
            print("  Insight generation skipped.")
        else:
            print(f"  DataFrame shape: {mapped_df.shape}")
            print(f"  DataFrame columns: {list(mapped_df.columns)}")
            
            # Check if we have the required columns for insight generation
            required_cols = ['Store', 'Product', 'Category', 'Quantity', 'Revenue', 'Profit']
            available_cols = [col for col in required_cols if col in mapped_df.columns]
            missing_cols = [col for col in required_cols if col not in mapped_df.columns]
            
            print(f"Step 2: Required columns check...")
            print(f"  Available: {len(available_cols)}/{len(required_cols)}")
            print(f"  Present: {available_cols}")
            if missing_cols:
                print(f"  Missing: {missing_cols}")
            
            # Validate data quality
            print(f"\nStep 3: Data quality validation...")
            print(f"  Total rows: {len(mapped_df)}")
            
            if 'Revenue' in mapped_df.columns:
                null_revenue = mapped_df['Revenue'].isnull().sum()
                print(f"  Revenue nulls: {null_revenue}")
                print(f"  Revenue range: ${mapped_df['Revenue'].min():.2f} - ${mapped_df['Revenue'].max():.2f}")
            
            if 'Profit' in mapped_df.columns:
                null_profit = mapped_df['Profit'].isnull().sum()
                print(f"  Profit nulls: {null_profit}")
                print(f"  Profit range: ${mapped_df['Profit'].min():.2f} - ${mapped_df['Profit'].max():.2f}")
            
            if len(available_cols) >= 4:  # Need at least 4 out of 6 columns
                print(f"\nStep 4: Generating retail insights...")
                print(f"  Calling generate_retail_insights()...")
                
                # CRITICAL DEBUG: Print exact DataFrame state before insight engine
                print("\n" + "="*80)
                print("BEFORE INSIGHT ENGINE - FINAL VALIDATION")
                print("="*80)
                print(f"DataFrame shape: {mapped_df.shape}")
                print(f"DataFrame columns: {list(mapped_df.columns)}")
                print(f"\nRequired columns check:")
                for col in required_cols:
                    exists = col in mapped_df.columns
                    print(f"  {col}: {'✓ EXISTS' if exists else '❌ MISSING'}")
                    if exists:
                        null_count = mapped_df[col].isnull().sum()
                        print(f"    Nulls: {null_count}, Sample: {mapped_df[col].head(1).values[0] if null_count < len(mapped_df) else 'All null'}")
                
                print(f"\nSample data (first 2 rows, required columns only):")
                print(mapped_df[available_cols].head(2))
                print("="*80 + "\n")
                
                retail_insights = generate_retail_insights(mapped_df)
                
                print(f"Step 5: Insights generated successfully!")
                print(f"  Keys: {list(retail_insights.keys())}")
                
                # Validate insights are not all N/A
                na_count = sum(1 for v in retail_insights.values() if 'N/A' in str(v) or 'Could not' in str(v))
                print(f"  Insights with N/A: {na_count}/{len(retail_insights)}")
                
                # Generate card format for frontend
                retail_insight_cards = get_insight_cards(retail_insights)
                
                print(f"Step 6: Insight cards created: {len(retail_insight_cards)} cards")
                
                # Print sample insights
                if 'top_store' in retail_insights:
                    print(f"\n  Sample insight (top_store): {retail_insights['top_store'][:100]}...")
                if 'loss_products' in retail_insights:
                    print(f"  Sample insight (loss_products): {retail_insights['loss_products'][:100]}...")
                if 'top_product' in retail_insights:
                    print(f"  Sample insight (top_product): {retail_insights['top_product'][:100]}...")
            else:
                print(f"\n⚠️ Step 4: SKIPPED - Only {len(available_cols)}/6 required columns available")
                print(f"  Need at least 4 columns for insight generation")
        
        print("=== END INSIGHT GENERATION ===\n")
    except Exception as e:
        print(f"\n❌ Insight generation failed: {str(e)}")
        traceback.print_exc()
    
    # For retail analysis, return None for retail results
    return retail_results, generic_results, dynamic_insights, ml_ts_result, retail_insights, retail_insight_cards

def process_generic_data(df, filters=None):
    """Process data using generic pipeline - no mapping, no retail logic"""
    # Apply filters if provided
    if filters:
        df = apply_filters(df, filters)
        print("Filters applied to generic data")
        print("Filtered DataFrame Shape:", df.shape)
    
    # Perform basic cleaning
    cleaned_df = basic_cleaning(df.copy())
    print("Generic data cleaning done")
    print("Cleaned DataFrame Shape:", cleaned_df.shape)
    
    # Generate dynamic insights from the original cleaned data
    print("Generating dynamic insights for generic data")
    dynamic_insights = generate_dynamic_insights(cleaned_df)
    print("Dynamic Insights Generated:", dynamic_insights['summary'])
    
    # Run generic analysis - no mapping needed, use original column names
    print("Starting generic analysis")
    generic_results = run_generic_analysis(cleaned_df)
    print("Generic Analysis Results:")
    print(f"Dataset Overview: {generic_results['overview']}")
    print(f"Missing Values: {generic_results['missing_values']}")
    print(f"Statistics: {generic_results['statistics']}")
    print(f"Correlation: {generic_results['correlation']}")
    print("Generic analysis completed")
    
    # For generic analysis, return None for retail results
    return None, generic_results, dynamic_insights


def get_generic_filter_columns(df, max_unique=20, min_unique=2):
    """
    Get categorical columns suitable for filtering in generic analysis
    Only includes columns with meaningful number of unique values (not IDs or timestamps)
    """
    # Detect all categorical-like columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    meaningful_cols = []
    
    for col in categorical_cols:
        unique_count = df[col].nunique()
        total_count = len(df[col])
        
        # Only include columns with 2-20 unique values (not IDs or timestamps)
        # Exclude columns that are mostly unique (likely IDs)
        if min_unique <= unique_count <= max_unique and unique_count < total_count * 0.8:  # Less than 80% unique
            meaningful_cols.append(col)
    
    # Limit to top N columns to avoid overload
    return meaningful_cols[:5]  # Max 5 filter columns


@app.route('/analyze', methods=['GET', 'POST'])
def analyze_file():
    """Handle retail analysis pipeline - supports both GET and POST"""
    try:
        print("Starting retail analysis process")
        
        # Check if file is in session
        if 'uploaded_file_path' not in session:
            return "Error: No file uploaded. Please upload a file first."
        
        filepath = session['uploaded_file_path']
        filename = session.get('original_filename', 'unknown')
        
        print(f"File path from session: {filepath}")
        
        # Initialize variables for both request methods
        filters = {}
        user_mapping = {}
        df = None
        retail_results = None
        generic_results = None
        dynamic_insights = None
        ml_ts_result = None
        
        # Handle different request methods
        if request.method == 'POST':
            # Process uploaded file and mapping data
            # Get filter selections
            filters = {}
            categorical_cols_str = request.form.get('categorical_columns', '')
            if categorical_cols_str:
                categorical_cols = categorical_cols_str.split(',')
                for col in categorical_cols:
                    if col.strip():
                        filter_value = request.form.get(f'filter_{col.strip()}', 'All')
                        if filter_value != 'All':
                            filters[col.strip()] = filter_value
            
            print(f"Filters applied: {filters}")
            
            # Store current filters in session (specifically for retail)
            session['retail_filters'] = filters
            
            # Get user mapping from form - core fields
            user_mapping = {}
            
            # Define standard fields and their form names
            core_field_mappings = {
                'Date': request.form.get('date_col'),
                'Store': request.form.get('store_col'),
                'Product': request.form.get('product_col'),
                'Category': request.form.get('category_col'),
                'Quantity': request.form.get('quantity_col'),
                'Selling_Price': request.form.get('selling_price_col'),
                'Cost_Price': request.form.get('cost_price_col')
            }
            
            # Extended field mappings
            extended_field_mappings = {}
            extended_field_options = get_extended_field_options()
            
            for field_type, field_label in extended_field_options.items():
                field_value = request.form.get(f'{field_type}_col')
                if field_value and field_value != "":
                    extended_field_mappings[field_type] = field_value
            
            # Combine core and extended mappings
            for standard_field, user_column in core_field_mappings.items():
                if user_column and user_column != "":
                    user_mapping[user_column] = standard_field
            
            for field_type, user_column in extended_field_mappings.items():
                if user_column and user_column != "":
                    user_mapping[user_column] = field_type  # Use the field type as the standard name
            
            print(f"User mapping created: {user_mapping}")
            
            # Load and analyze the CSV file using safe reader
            df = read_csv_safe(filepath)
            print("File loaded successfully")
            print("DataFrame Head:")
            print(df.head())
            print("DataFrame Shape:", df.shape)
            
            # Process using retail pipeline
            result = process_retail_data(df, filters, user_mapping)
            retail_results, generic_results, dynamic_insights, ml_ts_result, retail_insights, retail_insight_cards = result
            
            # INTELLIGENCE INTEGRATION: Store decision_log and quality metrics in session
            # Run advanced cleaning to get intelligence data
            print("\n=== INTELLIGENCE INTEGRATION DEBUG ===")
            print("Step 1: Importing AdvancedDataCleaner...")
            from analysis.advanced_cleaning import AdvancedDataCleaner
            print("✓ Import successful")
            
            print("Step 2: Creating cleaned DataFrame...")
            cleaned_df = basic_cleaning(df.copy())
            print(f"✓ Cleaned DataFrame shape: {cleaned_df.shape}")
            
            if user_mapping:
                print(f"Step 3: Applying user mapping: {user_mapping}")
                columns_to_rename = {k: v for k, v in user_mapping.items() if k in cleaned_df.columns}
                if columns_to_rename:
                    print(f"  Columns to rename: {columns_to_rename}")
                    cleaned_df = cleaned_df.rename(columns=columns_to_rename)
                    print(f"  ✓ Renamed columns. New shape: {cleaned_df.shape}")
            else:
                print("Step 3: No user mapping provided")
            
            print("Step 4: Creating AdvancedDataCleaner instance...")
            cleaner = AdvancedDataCleaner(cleaned_df)
            print("✓ AdvancedDataCleaner created")
            
            # EXECUTION CHECKPOINT: Verify auto_clean is called
            print(">>> CALLING AUTO CLEAN <<<")
            report = cleaner.auto_clean(convert_types=True, handle_outliers=True, outlier_method='cap', safe_mode=False)
            print(">>> AUTO CLEAN COMPLETED <<<")
            
            print("Step 6: Checking report structure...")
            print(f"  Report type: {type(report)}")
            print(f"  Report keys: {report.keys() if isinstance(report, dict) else 'NOT A DICT'}")
            
            print("Step 7: Extracting decision_log...")
            decision_log_raw = report.get('decision_log', [])
            print(f"  Decision Log Type: {type(decision_log_raw)}")
            print(f"  Decision Log Length: {len(decision_log_raw)}")
            if len(decision_log_raw) > 0:
                print(f"  First 3 decisions: {decision_log_raw[:3]}")
            else:
                print("  ⚠️ WARNING: Decision log is EMPTY!")
            
            print("Step 8: Storing in session...")
            session['decision_log'] = decision_log_raw
            session['ai_insights'] = report.get('insights', [])
            session['ai_warnings'] = report.get('critical_warnings', [])
            session['quality_score'] = report.get('data_quality_score', 0)
            
            # ENHANCED BUSINESS INTELLIGENCE: Generate data-driven insights
            print("\nStep 9: Generating enhanced business intelligence...")
            from analysis.enhanced_business_intelligence_v2 import generate_enhanced_insights, generate_enhanced_recommendations, generate_business_warnings
            
            enhanced_insights = generate_enhanced_insights(df, retail_results if retail_results else {})
            enhanced_recommendations = generate_enhanced_recommendations(df, retail_results if retail_results else {})
            business_warnings = generate_business_warnings(df, retail_results if retail_results else {})
            
            print(f"✓ Generated {len(enhanced_insights)} high-quality insights")
            print(f"✓ Generated {len(enhanced_recommendations)} specific recommendations")
            print(f"✓ Generated {len(business_warnings)} business warnings")
            
            # Store enhanced insights in session
            session['enhanced_insights'] = enhanced_insights
            session['enhanced_recommendations'] = enhanced_recommendations
            session['business_warnings'] = business_warnings
            
            # Store generic_results and dynamic_insights for analyst view
            session['generic_results'] = safe_generic_results if 'safe_generic_results' in locals() else None
            session['dynamic_insights'] = dynamic_insights
            session['retail_results'] = safe_retail_results if 'safe_retail_results' in locals() else None
            
            print(f"✓ Session stored:")
            print(f"  - decision_log length: {len(session['decision_log'])}")
            print(f"  - ai_insights length: {len(session['ai_insights'])}")
            print(f"  - enhanced_insights length: {len(session['enhanced_insights'])}")
            print(f"  - enhanced_recommendations length: {len(session['enhanced_recommendations'])}")
            print(f"  - ai_warnings length: {len(session['ai_warnings'])}")
            print(f"  - quality_score: {session['quality_score']}")
            print("=== END INTELLIGENCE DEBUG ===\n")
            
            # Store data file path in session (not the dataframe itself)
            data_path = temp_manager.save_dataframe(df, "retail_data")
            session['retail_data_path'] = data_path
            session['user_mapping'] = user_mapping  # This is small enough to keep in session
            
            # CRITICAL: Store FINAL column names after all preprocessing (single source of truth)
            final_columns = df.columns.tolist()
            session['final_columns'] = final_columns
            print(f"✓ Stored final columns in session: {len(final_columns)} columns")
        
        elif request.method == 'GET':
            # Reuse stored data and apply filters
            filters = session.get('retail_filters', {})  # Get retail-specific filters
            user_mapping = session.get('user_mapping', {})
            
            # Load original dataframe from file path
            data_path = session.get('retail_data_path')
            if data_path:
                df = temp_manager.load_dataframe(data_path)
            else:
                # Fallback to reading from file if temp data is lost
                df = read_csv_safe(filepath)
            
            print("Reusing stored dataframe for filtering")
            
            # Process the data again with current filters
            retail_results, generic_results, dynamic_insights, ml_ts_result, retail_insights, retail_insight_cards = process_retail_data(df, filters, user_mapping)
        
        # STEP 1: Build chart_data BEFORE converting DataFrames
        print("\n=== BUILDING CHART DATA FROM ORIGINAL DATAFRAMES ===")
        chart_data = {}
        if retail_results:
            # Build from ORIGINAL retail_results (still has DataFrames)
            top_products_df = retail_results.get("top_products")
            if isinstance(top_products_df, pd.DataFrame) and not top_products_df.empty:
                cols = top_products_df.columns.tolist()
                if len(cols) >= 2:
                    chart_data["top_products"] = {
                        "labels": top_products_df[cols[0]].astype(str).tolist(),
                        "values": top_products_df[cols[1]].fillna(0).tolist()
                    }
                print(f"✓ Built top_products chart: {len(chart_data['top_products']['labels'])} items")
            
            category_perf_df = retail_results.get("category_performance")
            if isinstance(category_perf_df, pd.DataFrame) and not category_perf_df.empty:
                cols = category_perf_df.columns.tolist()
                if len(cols) >= 2:
                    chart_data["category_performance"] = {
                        "labels": category_perf_df[cols[0]].astype(str).tolist(),
                        "values": category_perf_df[cols[1]].fillna(0).tolist()
                    }
                print(f"✓ Built category_performance chart: {len(chart_data['category_performance']['labels'])} items")
            
            sales_trend_df = retail_results.get("sales_trend")
            if isinstance(sales_trend_df, pd.DataFrame) and not sales_trend_df.empty:
                cols = sales_trend_df.columns.tolist()
                if len(cols) >= 2:
                    chart_data["sales_trend"] = {
                        "labels": sales_trend_df[cols[0]].astype(str).tolist(),
                        "values": sales_trend_df[cols[1]].fillna(0).tolist()
                    }
                print(f"✓ Built sales_trend chart: {len(chart_data['sales_trend']['labels'])} items")
            
            store_perf_df = retail_results.get("store_performance")
            if isinstance(store_perf_df, pd.DataFrame) and not store_perf_df.empty:
                cols = store_perf_df.columns.tolist()
                if len(cols) >= 1:
                    chart_data["store_performance"] = {
                        "labels": store_perf_df.index.astype(str).tolist(),
                        "values": store_perf_df[cols[0]].fillna(0).tolist()
                    }
                print(f"✓ Built store_performance chart: {len(chart_data['store_performance']['labels'])} items")
        
        # STEP 4: Final chart data debug with JSON serialization check
        print("\n=== FINAL CHART DATA ===")
        print(chart_data)
        
        # Verify JSON serializability
        import json
        try:
            json_str = json.dumps(chart_data)
            print(f"✓ Chart data is JSON serializable ({len(json_str)} bytes)")
            # Verify each chart's data types
            for chart_name, data in chart_data.items():
                print(f"  - {chart_name}: labels={type(data['labels']).__name__}({len(data['labels'])}), values={type(data['values']).__name__}({len(data['values'])})")
        except TypeError as e:
            print(f"✗ JSON serialization error: {e}")
            print("This means chart_data contains non-serializable objects (pandas Series, numpy arrays, etc.)")
        
        print("=== END FINAL CHART DATA ===\n")
        
        # STEP 2: Convert DataFrames to dict for template safety
        print("\n=== CONVERTING DATAFRAMES TO DICT FOR TEMPLATE SAFETY ===")
        safe_retail_results = None
        if retail_results:
            safe_retail_results = {}
            for key, value in retail_results.items():
                print(f"DEBUG: Processing retail_results[{key}], type: {type(value)}")
                if isinstance(value, pd.DataFrame):
                    print(f"DEBUG: Converting DataFrame to dict for key: {key}")
                    if value is not None and not value.empty:
                        # Convert DataFrame to records for safe template usage
                        safe_retail_results[key] = value.to_dict('records')
                    else:
                        safe_retail_results[key] = None
                else:
                    # Value is already a dict, list, or other type - keep as is
                    print(f"DEBUG: Keeping value as-is for key: {key}, type: {type(value)}")
                    safe_retail_results[key] = value
        
        # STEP 3: Convert generic DataFrames to dict for template safety
        print("\n=== CONVERTING GENERIC DATAFRAMES TO DICT ===")
        safe_generic_results = {}
        if generic_results:
            for key, value in generic_results.items():
                print(f"DEBUG: Processing generic_results[{key}], type: {type(value)}")
                if isinstance(value, pd.DataFrame):
                    print(f"DEBUG: Converting DataFrame to dict for key: {key}")
                    if value is not None and not value.empty:
                        # Convert DataFrame to records for safe template usage
                        safe_generic_results[key] = value.to_dict('records')
                    else:
                        safe_generic_results[key] = None
                else:
                    # Value is already a dict, list, or other type - keep as is
                    print(f"DEBUG: Keeping value as-is for key: {key}, type: {type(value)}")
                    safe_generic_results[key] = value
        
        print("DEBUG: Final generic_results type:", type(safe_generic_results))
        if safe_generic_results:
            for key, value in safe_generic_results.items():
                print(f"DEBUG: Final generic_results[{key}] type: {type(value)}")
        
        print("DEBUG: Dynamic insights type:", type(dynamic_insights))
        
        # Also pass the original dataframe info for filter controls
        original_df = read_csv_safe(filepath)
        if filters:
            original_df = apply_filters(original_df, filters)
        categorical_cols = get_meaningful_categorical_columns(original_df)
        categorical_options = {}
        categorical_labels = {}
        for col in categorical_cols:
            categorical_options[col] = ['All'] + get_unique_values(original_df, col)
            categorical_labels[col] = prettify_column_name(col)
        
        print("Template rendering started with safe data")
        print(f"DEBUG: Passing chart_data with keys: {list(chart_data.keys()) if chart_data else 'EMPTY'}")
        
        # Ensure chart_data is ALWAYS a dict (never None or undefined)
        # For retail analysis: use actual chart_data
        # For generic analysis: use empty dict {}
        if not chart_data:
            chart_data = {}
            print("DEBUG: chart_data was None/empty, set to empty dict for safety")
        
        # Extract forecast data safely from retail_results
        forecast_data = None
        if retail_results and isinstance(retail_results, dict):
            forecast_data = retail_results.get('forecast')
            print(f"DEBUG: Forecast data available: {forecast_data is not None}")
        
        # Extract recommendations safely from retail_results
        recommendations = None
        if retail_results and isinstance(retail_results, dict):
            # Use enhanced recommendations if available, otherwise fallback to old ones
            recommendations = session.get('enhanced_recommendations') or retail_results.get('recommendations')
            print(f"DEBUG: Recommendations available: {recommendations is not None}")
            if recommendations:
                print(f"DEBUG: Using {len(recommendations)} enhanced recommendations")
        
        # INTELLIGENCE INTEGRATION: Extract decision_log, insights, warnings
        print("\n=== TEMPLATE DATA PREPARATION ===")
        print("Retrieving intelligence from session...")
        decision_log = session.get('decision_log', [])
        ai_insights = session.get('ai_insights', [])
        ai_warnings = session.get('ai_warnings', [])
        quality_score = session.get('quality_score', None)
        
        print(f"✓ Retrieved from session:")
        print(f"  - decision_log type: {type(decision_log)}")
        print(f"  - decision_log length: {len(decision_log)}")
        if len(decision_log) > 0:
            print(f"  - First decision: {decision_log[0]}")
        else:
            print(f"  ⚠️ WARNING: decision_log is EMPTY in session!")
            print(f"  Session keys: {list(session.keys())}")
        
        print(f"  - ai_insights length: {len(ai_insights)}")
        print(f"  - ai_warnings length: {len(ai_warnings)}")
        print(f"  - quality_score: {quality_score}")
        print("=== END TEMPLATE DATA PREPARATION ===\n")
        
        return render_template('result.html', 
                             retail_results=safe_retail_results, 
                             generic_results=safe_generic_results,
                             dynamic_insights=dynamic_insights,
                             categorical_columns=categorical_cols,
                             categorical_options=categorical_options,
                             categorical_labels=categorical_labels,
                             current_filters=filters,
                             chart_data=chart_data,
                             forecast_data=forecast_data,
                             recommendations=recommendations,
                             decision_log=decision_log,
                             ai_insights=ai_insights,
                             ai_warnings=ai_warnings,
                             quality_score=quality_score,
                             retail_insights=retail_insights,
                             retail_insight_cards=retail_insight_cards,
                             enhanced_insights=session.get('enhanced_insights', []),
                             business_warnings=session.get('business_warnings', []))
                
    except Exception as e:
        print("FULL ERROR OCCURRED:")
        traceback.print_exc()
        return f"FULL ERROR: {str(e)}"


@app.route('/analyst-view')
def analyst_view():
    """Analyst View - Technical data analysis page for data analysts"""
    try:
        # Get all technical data from session
        decision_log = session.get('decision_log', [])
        ai_insights = session.get('ai_insights', [])
        ai_warnings = session.get('ai_warnings', [])
        quality_score = session.get('quality_score', None)
        
        # Load results from session
        retail_results = session.get('retail_results', None)
        generic_results = session.get('generic_results', None)
        dynamic_insights = session.get('dynamic_insights', None)
        
        # If not in session, try to load from data path
        if not dynamic_insights:
            data_path = session.get('retail_data_path') or session.get('generic_data_path')
            if data_path:
                df = temp_manager.load_dataframe(data_path)
                if df is not None:
                    from analysis.dynamic_insights import generate_dynamic_insights
                    dynamic_insights = generate_dynamic_insights(df)
                    session['dynamic_insights'] = dynamic_insights
        
        return render_template('analyst_view.html',
                             decision_log=decision_log,
                             ai_insights=ai_insights,
                             ai_warnings=ai_warnings,
                             quality_score=quality_score,
                             retail_results=retail_results,
                             generic_results=generic_results,
                             dynamic_insights=dynamic_insights)
    except Exception as e:
        print("ERROR in analyst_view:")
        traceback.print_exc()
        return f"Error loading analyst view: {str(e)}"


@app.route('/analyze_generic', methods=['GET', 'POST'])
def analyze_generic_file():
    """Handle generic analysis pipeline - supports both GET and POST"""
    try:
        print("Starting generic analysis process")
        
        # Check if file is in session
        if 'uploaded_file_path' not in session:
            return "Error: No file uploaded. Please upload a file first."
        
        filepath = session['uploaded_file_path']
        filename = session.get('original_filename', 'unknown')
        
        print(f"File path from session: {filepath}")
        
        # Initialize variables for both request methods
        filters = {}
        retail_results = None
        generic_results = None
        dynamic_insights = None
        df = None
        
        # Handle different request methods
        if request.method == 'POST':
            # Process uploaded file and mapping data
            # Get filter selections
            filters = session.get('generic_filters', {})  # Get generic-specific filters
            print(f"Filters applied: {filters}")
            
            # Load and analyze the CSV file using safe reader
            df = read_csv_safe(filepath)
            print("File loaded successfully")
            print("DataFrame Head:")
            print(df.head())
            print("DataFrame Shape:", df.shape)
            
            # Process the data and store results temporarily
            retail_results, generic_results, dynamic_insights = process_generic_data(df, filters)
            
            # Store data file path in session (not the dataframe itself)
            data_path = temp_manager.save_dataframe(df, "generic_data")
            session['generic_data_path'] = data_path
            
            # CRITICAL: Store FINAL column names after all preprocessing (single source of truth)
            final_columns = df.columns.tolist()
            session['final_columns'] = final_columns
            print(f"✓ Stored final columns in session: {len(final_columns)} columns")
            
        elif request.method == 'GET':
            # Reuse stored data and apply filters
            filters = session.get('generic_filters', {})  # Get generic-specific filters
            
            # Load original dataframe from file path
            data_path = session.get('generic_data_path')
            if data_path:
                df = temp_manager.load_dataframe(data_path)
            else:
                # Fallback to reading from file if temp data is lost
                df = read_csv_safe(filepath)
            
            print("Reusing stored dataframe for filtering (generic)")
            
            # Process the data again with current filters
            retail_results, generic_results, dynamic_insights = process_generic_data(df, filters)
        
        print("Preparing to render template with safe data")
        
        # Create safe versions of results for template
        safe_generic_results = {}
        if generic_results:
            for key, value in generic_results.items():
                print(f"DEBUG: Processing generic_results[{key}], type: {type(value)}")
                if isinstance(value, pd.DataFrame):
                    print(f"DEBUG: Converting DataFrame to dict for key: {key}")
                    if value is not None and not value.empty:
                        # Convert DataFrame to records for safe template usage
                        safe_generic_results[key] = value.to_dict('records')
                    else:
                        safe_generic_results[key] = None
                else:
                    # Value is already a dict, list, or other type - keep as is
                    print(f"DEBUG: Keeping value as-is for key: {key}, type: {type(value)}")
                    safe_generic_results[key] = value
        
        print("DEBUG: Final generic_results type:", type(safe_generic_results))
        if safe_generic_results:
            for key, value in safe_generic_results.items():
                print(f"DEBUG: Final generic_results[{key}] type: {type(value)}")
        
        print("DEBUG: Dynamic insights type:", type(dynamic_insights))
        print("DEBUG: Retail results type:", type(retail_results))
        
        # Initialize chart_data as empty dict for generic analysis
        # This prevents "Object of type Undefined is not JSON serializable" error
        chart_data = {}
        print("DEBUG: chart_data initialized as empty dict for generic analysis")
        
        # Generate filter options for generic analysis
        original_df = read_csv_safe(filepath)
        if filters:
            original_df = apply_filters(original_df, filters)
        
        # Get meaningful categorical columns for filtering (2-20 unique values)
        categorical_cols = get_generic_filter_columns(original_df)
        categorical_options = {}
        categorical_labels = {}
        for col in categorical_cols:
            col_data = original_df[col]
            unique_values = col_data.dropna().unique().tolist()
            categorical_options[col] = ['All'] + sorted(unique_values)
            categorical_labels[col] = prettify_column_name(col)
        
        print("Template rendering started with safe data")
        return render_template('result.html', 
                             retail_results=retail_results,  # Will be None for generic
                             generic_results=safe_generic_results,
                             dynamic_insights=dynamic_insights,
                             categorical_columns=categorical_cols,
                             categorical_options=categorical_options,
                             categorical_labels=categorical_labels,
                             current_filters=filters,
                             chart_data=chart_data,  # Empty dict {} for generic analysis
                             forecast_data=None,  # No forecast for generic
                             recommendations=None)  # No recommendations for generic
                
    except Exception as e:
        print("FULL ERROR OCCURRED:")
        traceback.print_exc()
        return f"FULL ERROR: {str(e)}"


# Route to reset filters (clear filters but keep dataset)
@app.route('/reset_filters', methods=['GET', 'POST'])
def reset_filters():
    try:
        # Check if file is in session
        if 'uploaded_file_path' not in session:
            return redirect(url_for('home'))
        
        # Get analysis type from session to determine which filters to clear
        analysis_type = session.get('analysis_type', 'retail')
        
        # Clear filters based on analysis type
        if analysis_type == 'retail':
            session.pop('retail_filters', None)
        elif analysis_type == 'generic':
            session.pop('generic_filters', None)
        else:
            # Clear both if unknown type
            session.pop('retail_filters', None)
            session.pop('generic_filters', None)
        
        # Get analysis type from session to redirect appropriately
        if analysis_type == 'retail':
            return redirect(url_for('analyze_file'))
        elif analysis_type == 'generic':
            return redirect(url_for('analyze_generic_file'))
        else:
            return redirect(url_for('analyze_file'))  # Default to retail
        
    except Exception as e:
        print("ERROR in reset_filters:")
        traceback.print_exc()
        return f"Reset Error: {str(e)}"


# API endpoint to get final column names (single source of truth)
@app.route('/get_columns', methods=['GET'])
def get_columns():
    """Return final column names stored in session after preprocessing"""
    try:
        final_columns = session.get('final_columns', [])
        print(f"✓ Returning {len(final_columns)} columns from session")
        return {'columns': final_columns}
    except Exception as e:
        print(f"ERROR in get_columns: {str(e)}")
        return {'error': str(e), 'columns': []}, 500

# Route to re-analyze with filters (for AJAX/filter updates)
@app.route('/apply_filters', methods=['POST'])
def apply_filters_route():
    try:
        # Get filter selections
        filters = {}
        categorical_cols_str = request.form.get('categorical_columns', '')
        if categorical_cols_str:
            categorical_cols = categorical_cols_str.split(',')
            for col in categorical_cols:
                if col.strip():
                    filter_value = request.form.get(f'filter_{col.strip()}', 'All')
                    if filter_value != 'All':
                        filters[col.strip()] = filter_value
        
        print(f"Filters applied in apply_filters: {filters}")
        
        # Store filters in session based on analysis type
        analysis_type = session.get('analysis_type', 'retail')
        if analysis_type == 'retail':
            session['retail_filters'] = filters
        elif analysis_type == 'generic':
            session['generic_filters'] = filters
        
        # Get analysis type from session
        if analysis_type == 'retail':
            return redirect(url_for('analyze_file'))  # This will now use GET method
        elif analysis_type == 'generic':
            return redirect(url_for('analyze_generic_file'))  # This will now use GET method
        else:
            return redirect(url_for('analyze_file'))  # Default to retail
        
    except Exception as e:
        print("ERROR in apply_filters_route:")
        traceback.print_exc()
        return f"Filter Error: {str(e)}"


@app.route('/custom_chart_data', methods=['POST'])
def custom_chart_data():
    """Backend route for custom visualization builder data"""
    try:
        import json
        from pandas.api.types import is_numeric_dtype
        
        # Get chart parameters from request
        data = request.get_json()
        x_axis = data.get('x_axis')
        y_axis = data.get('y_axis')
        chart_type = data.get('chart_type', 'bar')
        
        print(f"Custom chart request: x={x_axis}, y={y_axis}, type={chart_type}")
        
        # Validate inputs
        if not x_axis or not y_axis:
            return {'error': 'Both X and Y axis columns are required'}, 400
        
        # Load the dataframe from session using temp_manager (same as dashboard)
        analysis_type = session.get('analysis_type', 'retail')
        
        if analysis_type == 'retail':
            data_path = session.get('retail_data_path')
        else:
            data_path = session.get('generic_data_path')
        
        if not data_path:
            return {'error': 'Data not found in session. Please re-upload your file.'}, 400
        
        # Check if temp file exists, fallback to CSV if missing
        import os
        
        # Convert to absolute path for reliable checking
        data_path = os.path.abspath(data_path)
        print(f"Checking path: {data_path}")
        
        # Try loading with exception handling
        df = None
        try:
            if not os.path.exists(data_path):
                print("⚠️  Temp file not found")
                raise FileNotFoundError(f"Temp file missing: {data_path}")
            
            df = temp_manager.load_dataframe(data_path)
            print("✓ Loaded from temp file")
            
        except Exception as e:
            print(f"Entering fallback due to: {str(e)}")
            
            # Check if we have original file path
            uploaded_file_path = session.get('uploaded_file_path')
            if not uploaded_file_path:
                return {'error': 'Session expired. Please re-upload file.'}, 400
            
            print("🔄 Reloading CSV...")
            
            # Reload from CSV with same safe reader
            df = read_csv_safe(uploaded_file_path)
            print(f"✓ Reloaded from CSV: {df.shape}")
            
            # Apply SAME preprocessing pipeline as main dashboard
            # Step 1: Basic cleaning (remove duplicates, handle missing values)
            df = basic_cleaning(df.copy())
            print(f"✓ Applied basic cleaning: {df.shape}")
            
            # Step 2: Apply column mapping for retail analysis (if retail type)
            # CRITICAL: Only apply mapping if it was used in the original session
            # This ensures column names match what frontend expects
            if analysis_type == 'retail':
                user_mapping = session.get('user_mapping', {})
                if user_mapping:
                    # Only rename columns that exist in the dataframe
                    columns_to_rename = {k: v for k, v in user_mapping.items() if k in df.columns}
                    if columns_to_rename:
                        df = df.rename(columns=columns_to_rename)
                        print(f"✓ Applied column mapping: {columns_to_rename}")
                        print(f"  Column names now: {list(df.columns)}")
            
            # Save again to temp_manager
            new_path = temp_manager.save_dataframe(df, f"{analysis_type}_data")
            new_path = os.path.abspath(new_path)
            print(f"✓ Saved new temp file: {new_path}")
            
            # Update session with new path
            if analysis_type == 'retail':
                session['retail_data_path'] = new_path
            else:
                session['generic_data_path'] = new_path
        else:
            # Normal path - load from existing temp file
            df = temp_manager.load_dataframe(data_path)
        
        if df is None:
            return {'error': 'Failed to load data from session. Please re-upload your file.'}, 500
        
        # Apply current filters if any
        analysis_type = session.get('analysis_type', 'retail')
        if analysis_type == 'retail':
            filters = session.get('retail_filters', {})
        else:
            filters = session.get('generic_filters', {})
        
        if filters:
            df = apply_filters(df, filters)
            print(f"Applied filters to custom chart data: {filters}")
        
        # CRITICAL FIX: Normalize column names for consistent matching
        # Store original column names for response
        original_columns = df.columns.tolist()
        
        # Create normalized column mapping (original -> normalized)
        column_mapping = {col: col.strip().lower() for col in df.columns}
        
        # Print available columns for debugging
        print(f"Available columns (original): {original_columns}")
        print(f"Available columns (normalized): {list(column_mapping.values())}")
        print(f"Requested X-axis: '{x_axis}'")
        print(f"Requested Y-axis: '{y_axis}'")
        
        # Normalize requested column names
        x_axis_normalized = x_axis.strip().lower()
        y_axis_normalized = y_axis.strip().lower()
        
        # Find matching columns (case-insensitive)
        x_axis_match = None
        y_axis_match = None
        
        for orig_col, norm_col in column_mapping.items():
            if norm_col == x_axis_normalized:
                x_axis_match = orig_col
            if norm_col == y_axis_normalized:
                y_axis_match = orig_col
        
        # Use matched column names if found
        if x_axis_match:
            x_axis = x_axis_match
            print(f"✓ Matched X-axis: '{x_axis_match}'")
        else:
            print(f"✗ X-axis not found: '{x_axis}'")
        
        if y_axis_match:
            y_axis = y_axis_match
            print(f"✓ Matched Y-axis: '{y_axis_match}'")
        else:
            print(f"✗ Y-axis not found: '{y_axis}'")
        
        # Validate columns exist
        if x_axis not in df.columns or y_axis not in df.columns:
            return {
                'error': f'Selected columns not found in dataset',
                'available_columns': original_columns,
                'requested_x': x_axis,
                'requested_y': y_axis
            }, 400
        
        # Validate Y-axis is numeric
        if not is_numeric_dtype(df[y_axis]):
            return {'error': f'Y-axis column "{y_axis}" must be numeric'}, 400
        
        # Check cardinality limit based on chart type (intelligent protection)
        unique_count = df[x_axis].nunique()
        
        # Set limit based on chart type for optimal UX
        if chart_type in ['pie', 'doughnut']:
            limit = 20  # Too many slices = unreadable
        elif chart_type == 'scatter':
            limit = 200  # Scatter can handle more points
        else:
            limit = 50  # bar, line charts
        
        if unique_count > limit:
            return {
                'error': f'Too many unique values ({unique_count}). Max allowed for {chart_type} is {limit}. Apply filters or choose another column.'
            }, 400
        
        # Group data by X-axis and sum Y-axis values
        grouped_df = df.groupby(x_axis)[y_axis].sum().reset_index()  # type: ignore
        
        # Sort by Y-axis values descending
        grouped_df = grouped_df.sort_values(by=y_axis, ascending=False)  # type: ignore
        
        # Prepare response data
        result = {
            'labels': grouped_df[x_axis].astype(str).tolist(),
            'values': grouped_df[y_axis].fillna(0).tolist(),
            'x_axis': x_axis,
            'y_axis': y_axis,
            'chart_type': chart_type
        }
        
        print(f"✓ Custom chart data prepared: {len(result['labels'])} data points")
        return result  # type: ignore
        
    except Exception as e:
        print("ERROR in custom_chart_data:")
        traceback.print_exc()
        return {'error': str(e)}, 500

# ========================================
# MODERN UI ROUTES (NEW)
# ========================================

@app.route('/modern')
def modern_home():
    """Modern upload page"""
    return render_template('upload_modern.html')

@app.route('/modern/upload', methods=['POST'])
def modern_upload():
    """Handle modern file upload and redirect to choose analysis"""
    print("MODERN UPLOAD ROUTE HIT")
    
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return redirect(url_for('modern_home'))
    
    file = request.files['file']
    
    if file.filename == '' or file.filename is None:
        print("ERROR: Empty filename")
        return redirect(url_for('modern_home'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        session['uploaded_file_path'] = filepath
        session['original_filename'] = filename
        
        print(f"File uploaded successfully: {filename}")
        print(f"Redirecting to choose analysis...")
        
        # Redirect to choose analysis page (step 2)
        return redirect(url_for('modern_choose_analysis'))
    
    print("ERROR: Invalid file type")
    return redirect(url_for('modern_home'))

@app.route('/modern/choose-analysis')
def modern_choose_analysis():
    """Modern choose analysis type page"""
    filepath = session.get('uploaded_file_path')
    if not filepath:
        return redirect(url_for('modern_home'))
    
    try:
        df = read_csv_safe(filepath)
        columns = list(df.columns)
        extended_detected = detect_extended_fields(columns)
        categorical_cols = get_meaningful_categorical_columns(df)
        categorical_options = {}
        categorical_labels = {}
        for col in categorical_cols:
            categorical_options[col] = get_unique_values(df, col)
            categorical_labels[col] = prettify_column_name(col)
        extended_field_options = get_extended_field_options()
        
        return render_template('choose_analysis.html',
                             columns=columns,
                             filename=session.get('original_filename'),
                             categorical_columns=categorical_cols,
                             categorical_options=categorical_options,
                             categorical_labels=categorical_labels,
                             extended_detected=extended_detected,
                             extended_field_options=extended_field_options)
    except Exception as e:
        return f"Error loading file: {str(e)}"

@app.route('/modern/analyze', methods=['POST'])
def modern_analyze():
    """Modern analysis with AI-powered dashboard"""
    print("\n=== MODERN ANALYZE ROUTE STARTED ===")
    
    # Check if file is being upl
    # oaded directly
    if 'file' in request.files:
        print("✓ File detected in request, processing upload...")
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
            print("✗ ERROR: Empty filename")
            return redirect(url_for('modern_home'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            session['uploaded_file_path'] = filepath
            session['original_filename'] = filename
            
            print(f"✓ File saved successfully: {filepath}")
        else:
            print("✗ ERROR: Invalid file type")
            return redirect(url_for('modern_home'))
    else:
        # Fallback to session (for backward compatibility)
        filepath = session.get('uploaded_file_path')
        if not filepath:
            print("✗ ERROR: No file path in session and no file in request")
            return redirect(url_for('modern_home'))
    
    try:
        print("Step 1: Reading CSV file...")
        df = read_csv_safe(filepath)
        print(f"✓ CSV loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Get filters from form
        filters = {}
        for key, value in request.form.items():
            if key.startswith('filter_'):
                col = key.replace('filter_', '')
                if value and value != 'All':
                    filters[col] = value
        
        if filters:
            print(f"Step 2: Applying filters: {filters}")
            filtered_df = apply_filters(df, filters)
            print(f"✓ Filters applied: {filtered_df.shape[0]} rows remaining")
        else:
            print("Step 2: No filters applied")
            filtered_df = df
        
        # Run advanced cleaning with analytical intelligence
        print("Step 3: Running AdvancedDataCleaner...")
        from analysis.advanced_cleaning import AdvancedDataCleaner
        cleaner = AdvancedDataCleaner(filtered_df)
        report = cleaner.auto_clean(
            convert_types=True,
            handle_outliers=True,
            outlier_method='cap',
            safe_mode=False
        )
        print(f"✓ Cleaning complete. Quality Score: {report.get('data_quality_score', 0)}/100")
        
        cleaned_df = cleaner.get_cleaned_data()
        print(f"✓ Cleaned data shape: {cleaned_df.shape}")
        
        # Extract report data with validation
        quality_score = report.get('data_quality_score', 0)
        complexity = report.get('dataset_complexity', 'MEDIUM')
        insights = report.get('insights', [])
        warnings = report.get('critical_warnings', [])
        decision_log = report.get('decision_log', [])
        ai_summary = report.get('summary', '')
        
        print(f"Step 4: Extracted metrics:")
        print(f"  - Quality Score: {quality_score}")
        print(f"  - Complexity: {complexity}")
        print(f"  - Insights count: {len(insights)}")
        print(f"  - Warnings count: {len(warnings)}")
        print(f"  - Decision log count: {len(decision_log)}")
        
        # Calculate metrics
        rows, cols = cleaned_df.shape
        missing_before = report['missing_values_summary']['total_missing_before']
        missing_after = report['missing_values_summary']['total_missing_after']
        
        print(f"Step 5: Missing values - Before: {missing_before}, After: {missing_after}")
        
        # Average confidence
        if decision_log:
            avg_confidence = sum(d.get('confidence', 0.5) for d in decision_log) / len(decision_log)
            avg_confidence_pct = round(avg_confidence * 100)
        else:
            avg_confidence_pct = 85
        
        print(f"  - Average Confidence: {avg_confidence_pct}%")
        
        # Quality distribution for charts - ENSURE POSITIVE VALUES ONLY
        total_issues = missing_before + report.get('total_outliers', 0)
        quality_clean = max(0, min(100, 100 - total_issues))  # Clamp between 0-100
        quality_fixed = min(missing_before, 50)  # Cap at 50
        quality_outliers = min(report.get('total_outliers', 0), 30)  # Cap at 30
        
        # Ensure percentages add up to 100
        quality_total = quality_clean + quality_fixed + quality_outliers
        if quality_total > 100:
            quality_clean = max(0, 100 - quality_fixed - quality_outliers)
        
        print(f"Step 6: Quality distribution - Clean: {quality_clean}%, Fixed: {quality_fixed}%, Outliers: {quality_outliers}%")
        
        # Generate summary text
        summary_text = f"Analyzed {rows:,} rows × {cols} columns | Quality Score: {quality_score}/100"
        
        # Prepare charts data - LIMITED TO SMALL SAMPLES
        charts_data = {}
        
        # Add distribution charts for top numeric columns - LIMIT DATA
        numeric_cols = cleaned_df.select_dtypes(include=['number']).columns[:4]
        for i, col in enumerate(numeric_cols):
            chart_type = 'bar' if cleaned_df[col].nunique() < 20 else 'line'
            # CRITICAL FIX: Limit to top 10 values only to prevent infinite expansion
            value_counts = cleaned_df[col].value_counts().head(10)
            charts_data[f'{col} Distribution'] = {
                'type': chart_type,
                'data': {
                    'labels': [str(x) for x in value_counts.index.tolist()],
                    'datasets': [{
                        'label': col,
                        'data': value_counts.tolist(),
                        'backgroundColor': f'hsla({i * 60}, 70%, 60%, 0.6)',
                        'borderColor': f'hsla({i * 60}, 70%, 60%, 1)',
                        'borderWidth': 2
                    }]
                },
                'show_legend': False
            }
        
        print(f"Step 7: Generated {len(charts_data)} visualization charts")
        
        # Validate all required data before rendering
        print("Step 8: Validating template data...")
        template_data = {
            'filename': session.get('original_filename'),
            'quality_score': quality_score,
            'complexity': complexity,
            'rows': rows,
            'cols': cols,
            'avg_confidence': avg_confidence_pct,
            'insights': insights,
            'warnings': warnings,
            'decision_log': decision_log,
            'ai_summary': ai_summary,
            'summary_text': summary_text,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'quality_clean': quality_clean,
            'quality_fixed': quality_fixed,
            'quality_outliers': quality_outliers,
            'charts_data': charts_data
        }
        
        # Verify no None values in critical fields
        for key, value in template_data.items():
            if value is None:
                print(f"  ⚠️ WARNING: {key} is None, setting default")
                if isinstance(value, (int, float)):
                    template_data[key] = 0
                elif isinstance(value, (list, dict)):
                    template_data[key] = [] if isinstance(value, list) else {}
                else:
                    template_data[key] = 'N/A'
        
        print("✓ All template data validated successfully")
        print("=== RENDERING RESULT PAGE ===\n")
        
        return render_template('result_modern.html', **template_data)
    
    except Exception as e:
        print(f"\n✗✗✗ CRITICAL ERROR IN MODERN_ANALYZE ✗✗✗")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        print("=== END ERROR ===\n")
        return f"Error during analysis: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)