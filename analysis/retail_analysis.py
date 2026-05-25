import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def prepare_time_series_data(df, user_mapping=None):
    """
    Prepare time series data for ML forecasting with flexible column detection
    
    Args:
        df: Input dataframe with retail data
        user_mapping: Dictionary mapping standard fields to user columns
                     e.g., {'order_date': 'Date', 'total_sales': 'Revenue'}
    
    Returns:
        dict with status and prepared data
    """
    print("\n" + "="*60)
    print("Preparing time series data for ML forecasting...")
    print("="*60)
    
    # Work on a copy
    ts_df = df.copy()
    
    # CRITICAL FIX: Store original columns and normalize immediately
    ORIGINAL_COLUMNS = ts_df.columns.tolist()
    print("\n[STEP 0] Normalizing column names...")
    print("Original columns:", ORIGINAL_COLUMNS)
    ts_df.columns = ts_df.columns.str.strip().str.lower()
    print("Normalized columns:", ts_df.columns.tolist())
    
    # Initialize column variables
    date_col = None
    target_col = None
    quantity_col = None
    price_col = None
    cost_col = None
    
    # =========================================================
    # STEP 1: Use MAPPING FIRST (highest priority)
    # =========================================================
    if user_mapping:
        print("\n[STEP 1] Checking user mapping...")
        
        # CRITICAL: Normalize BOTH keys AND values in mapping
        normalized_mapping = {
            k.strip().lower(): v.strip().lower()
            for k, v in user_mapping.items()
        }
        print(f"Normalized mapping: {normalized_mapping}")
        
        # Reverse mapping: find which user column maps to each standard field
        for user_col_norm, standard_field in normalized_mapping.items():
            if standard_field == 'date':
                date_col = user_col_norm
            elif standard_field == 'revenue':
                # Direct revenue/sales column
                target_col = user_col_norm
            elif standard_field == 'selling_price':
                # This is unit price, not total revenue
                price_col = user_col_norm
            elif standard_field == 'quantity':
                quantity_col = user_col_norm
            elif standard_field == 'cost_price':
                cost_col = user_col_norm
        
        # VERY IMPORTANT: Debug print mapped columns
        print("\n" + "="*60)
        print("DEBUG: Mapped Column Detection")
        print("="*60)
        print("Mapped date:", date_col)
        print("Mapped revenue:", target_col if target_col else "None (direct)")
        print("Mapped quantity:", quantity_col)
        print("Mapped price:", price_col)
        print("="*60)
    
    # =========================================================
    # STEP 2: Validate/Find Date Column (MAPPING HAS PRIORITY)
    # =========================================================
    print("\n[STEP 2] Validating date column...")
    
    # CRITICAL: Check column existence BEFORE validation
    print("Checking column existence...")
    print("date_col:", date_col)
    print("df.columns:", ts_df.columns.tolist())
    print("Exists?", date_col in ts_df.columns if date_col else False)
    
    # If no mapped date column, try common names as fallback
    if not date_col:
        print("No mapped date found, searching by common names as fallback...")
        common_date_names = ['date', 'order_date', 'transaction_date', 'time', 'timestamp', 'sale_date']
        for col_name in common_date_names:
            if col_name in ts_df.columns:
                date_col = col_name
                print(f"Found date column by common name: {date_col}")
                break
    else:
        print(f"✓ Using MAPPED date column: {date_col}")
    
    # Validate date column exists
    if not date_col or date_col not in ts_df.columns:
        # CRITICAL FIX: Try loose matching (ignore spaces/underscores)
        if date_col:
            print(f"\nAttempting loose matching for '{date_col}'...")
            date_col_normalized = date_col.replace(" ", "").replace("_", "")
            for c in ts_df.columns:
                c_normalized = c.replace(" ", "").replace("_", "")
                if date_col_normalized == c_normalized:
                    date_col = c
                    print(f"✓ Fixed column match using loose matching: {date_col}")
                    break
        
        # SMART FALLBACK: If still not found, search for date-like columns
        if not date_col or date_col not in ts_df.columns:
            print(f"\n⚠️ Mapped date column not found, searching for alternatives...")
            
            # Priority 1: Look for columns containing 'date'
            date_keywords = ['date', 'time', 'timestamp', 'dt']
            for keyword in date_keywords:
                for c in ts_df.columns:
                    if keyword in c.lower():
                        date_col = c
                        print(f"⚠️ Fallback date column used: {date_col}")
                        break
                if date_col:
                    break
            
            # Priority 2: Check if any column looks like a date (datetime type)
            if not date_col:
                datetime_cols = ts_df.select_dtypes(include=['datetime', 'datetimetz']).columns
                if len(datetime_cols) > 0:
                    date_col = datetime_cols[0]
                    print(f"⚠️ Fallback: Using datetime-type column: {date_col}")
            
            # Final validation - ONLY return not_ready if truly no date column
            if not date_col or date_col not in ts_df.columns:
                print(f"❌ ML Forecasting: No date-like column found!")
                print("Available columns:", ts_df.columns.tolist())
                print("Original columns:", ORIGINAL_COLUMNS)
                return {
                    "status": "not_ready",
                    "message": "Date column required for time series forecasting (none found in data)",
                    "data": None
                }
    
    print(f"✓ Using date column: {date_col}")
    
    # Debug: Show sample date values
    print("Sample date values:", ts_df[date_col].head())
    
    # =========================================================
    # STEP 3: Identify Target Column (MAPPING-BASED PRIORITY)
    # =========================================================
    print("\n[STEP 3] Searching for target column...")
    
    # Priority A: Direct revenue/sales from mapping
    if target_col:
        # Check if target exists (with loose matching fallback)
        target_exists = target_col in ts_df.columns
        
        if not target_exists:
            print(f"\nAttempting loose matching for target '{target_col}'...")
            target_normalized = target_col.replace(" ", "").replace("_", "")
            for c in ts_df.columns:
                if c.replace(" ", "").replace("_", "") == target_normalized:
                    target_col = c
                    target_exists = True
                    print(f"✓ Fixed target match using loose matching: {target_col}")
                    break
        
        # SMART FALLBACK: If mapped target not found, search for alternatives
        if not target_exists:
            print(f"\n⚠️ Mapped target column '{target_col}' not found, searching for alternatives...")
            
            # Strategy 1: Look for revenue/sales keywords
            revenue_keywords = ['revenue', 'sales', 'amount', 'total', 'turnover', 'income']
            for keyword in revenue_keywords:
                for c in ts_df.columns:
                    if keyword in c.lower():
                        # Verify it's numeric
                        if pd.api.types.is_numeric_dtype(ts_df[c]):
                            target_col = c
                            print(f"⚠️ Fallback target column used: {target_col}")
                            target_exists = True
                            break
                if target_exists:
                    break
            
            # Strategy 2: Look for any numeric column that could be a target
            if not target_exists:
                print("Searching for numeric columns as potential targets...")
                numeric_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Exclude obvious non-target columns
                exclude_patterns = ['id', 'count', 'qty', 'quantity', 'price', 'cost', 'discount', 'profit']
                potential_targets = []
                
                for col in numeric_cols:
                    is_excluded = any(pattern in col.lower() for pattern in exclude_patterns)
                    if not is_excluded and col != date_col:
                        potential_targets.append(col)
                
                if len(potential_targets) > 0:
                    # Prefer columns with 'total' or 'sum' in name
                    for col in potential_targets:
                        if 'total' in col.lower() or 'sum' in col.lower():
                            target_col = col
                            print(f"⚠️ Fallback: Using numeric column: {target_col}")
                            target_exists = True
                            break
                    
                    # If no 'total' column, use first numeric column
                    if not target_exists and len(potential_targets) > 0:
                        target_col = potential_targets[0]
                        print(f"⚠️ Fallback: Using first numeric column: {target_col}")
                        target_exists = True
        
        if target_exists:
            print(f"✓ Using target column: {target_col}")
        else:
            target_col = None  # Reset so we can try derivation or other fallbacks
    
    # Priority B: Try to derive revenue from quantity * price (both must be mapped)
    elif quantity_col and price_col:
        # Check if both columns exist (with loose matching fallback)
        qty_exists = quantity_col in ts_df.columns
        price_exists = price_col in ts_df.columns
        
        # Try loose matching if not found directly
        if not qty_exists:
            qty_normalized = quantity_col.replace(" ", "").replace("_", "")
            for c in ts_df.columns:
                if c.replace(" ", "").replace("_", "") == qty_normalized:
                    quantity_col = c
                    qty_exists = True
                    print(f"✓ Fixed quantity match using loose matching: {quantity_col}")
                    break
        
        if not price_exists:
            price_normalized = price_col.replace(" ", "").replace("_", "")
            for c in ts_df.columns:
                if c.replace(" ", "").replace("_", "") == price_normalized:
                    price_col = c
                    price_exists = True
                    print(f"✓ Fixed price match using loose matching: {price_col}")
                    break
        
        if qty_exists and price_exists:
            print(f"✓ Deriving revenue from MAPPED columns: {quantity_col} * {price_col}")
            ts_df['Computed_Revenue'] = pd.to_numeric(ts_df[quantity_col], errors='coerce') * pd.to_numeric(ts_df[price_col], errors='coerce')
            target_col = 'Computed_Revenue'
            print(f"✓ Created derived target column: {target_col}")
    
    # Priority C: Fallback to direct column name detection (ONLY if mapping failed)
    else:
        print("No mapped target found, using FALLBACK detection...")
        
        # Check for direct sales/revenue columns
        direct_target_names = ['Revenue', 'revenue', 'Sales', 'sales', 'Total_Sales', 'total_sales', 'turnover', 'Turnover']
        for col_name in direct_target_names:
            if col_name in ts_df.columns:
                target_col = col_name
                print(f"Found direct target column: {target_col}")
                break
        
        # If still no target, try deriving from raw column names
        if not target_col:
            # Find quantity by common names
            common_qty_names = ['quantity', 'qty', 'units', 'count']
            for col_name in common_qty_names:
                if col_name in ts_df.columns:
                    quantity_col = col_name
                    break
            
            # Find price by common names
            common_price_names = ['selling_price', 'price', 'unit_price']
            for col_name in common_price_names:
                if col_name in ts_df.columns:
                    price_col = col_name
                    break
            
            # Derive if both found
            if quantity_col and price_col:
                print(f"✓ Deriving revenue from detected columns: {quantity_col} * {price_col}")
                ts_df['Computed_Revenue'] = pd.to_numeric(ts_df[quantity_col], errors='coerce') *pd.to_numeric(ts_df[price_col], errors='coerce')
                target_col = 'Computed_Revenue'
                print(f"✓ Created derived target column: {target_col}")
    
    # Final validation
    if not target_col:
        # Try one last loose matching attempt for any potential matches
        print("\nFinal attempt: Checking all columns for potential target matches...")
        common_target_patterns = ['revenue', 'sales', 'amount', 'total', 'turnover']
        for pattern in common_target_patterns:
            for c in ts_df.columns:
                if pattern in c:
                    target_col = c
                    print(f"✓ Found potential target by pattern '{pattern}': {target_col}")
                    break
            if target_col:
                break
        
        # Last resort: use any numeric column
        if not target_col:
            print("\nLast resort: Searching for ANY usable numeric column...")
            numeric_cols = ts_df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col != date_col:  # Don't use date column as target
                    target_col = col
                    print(f"⚠️ Using numeric column as last resort: {target_col}")
                    break
        
        if not target_col:
            print("❌ ML Forecasting: No suitable target column found!")
            print("Available columns:", ts_df.columns.tolist())
            return {
                "status": "not_ready",
                "message": "Revenue or sales column required (direct or derivable from quantity*price)",
                "data": None
            }
    
    # VERY IMPORTANT: Print final target used
    print("\n" + "="*60)
    print("DEBUG: Final Column Selection")
    print("="*60)
    print("Final target used:", target_col)
    print("Final date_col:", date_col)
    print("="*60)
    
    # =========================================================
    # STEP 4: Data Cleaning & Preparation
    # =========================================================
    print("\n[STEP 4] Cleaning and preparing data...")
    
    # Convert date column to datetime
    print(f"Converting date column: {date_col}")
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors='coerce')
    
    # Debug: Check null dates after conversion
    print("Null dates after conversion:", ts_df[date_col].isnull().sum())
    
    # Drop rows with invalid dates
    valid_dates_before = len(ts_df)
    ts_df = ts_df.dropna(subset=[date_col])
    valid_dates_after = len(ts_df)
    print(f"Dropped {valid_dates_before - valid_dates_after} rows with invalid dates")
    
    # Ensure target is numeric and clean
    if target_col in ts_df.columns:  # Only if it's an actual column (not computed)
        print(f"Cleaning target column: {target_col}")
        ts_df[target_col] = pd.to_numeric(ts_df[target_col], errors='coerce')
        
        # Debug: Check null targets after conversion
        print("Null targets after conversion:", ts_df[target_col].isnull().sum())
        
        # Drop rows with invalid target values
        valid_target_before = len(ts_df)
        ts_df = ts_df.dropna(subset=[target_col])
        valid_target_after = len(ts_df)
        print(f"Dropped {valid_target_before - valid_target_after} rows with invalid target values")
    
    # Check if we have enough data after cleaning
    if len(ts_df) < 10:
        print(f"❌ Insufficient data ({len(ts_df)} rows after cleaning)")
        return {
            "status": "not_ready",
            "message": f"Insufficient data points ({len(ts_df)}). Need at least 10 for forecasting.",
            "data": None
        }
    
    # Sort by date
    ts_df = ts_df.sort_values(date_col)
    
    # Aggregate to daily level
    daily_data = ts_df.groupby(date_col)[target_col].sum().reset_index()
    daily_data.columns = ['date', 'value']
    
    # Ensure continuous date range with smart filling
    if len(daily_data) > 0:
        full_date_range = pd.date_range(
            start=daily_data['date'].min(),
            end=daily_data['date'].max(),
            freq='D'
        )
        daily_data = daily_data.set_index('date').reindex(full_date_range)
        
        # Smart filling strategy: forward fill then backward fill
        daily_data['value'] = daily_data['value'].ffill().bfill()
        
        # If still missing (all zeros), use rolling average
        if daily_data['value'].isnull().any():
            daily_data['value'] = daily_data['value'].fillna(daily_data['value'].rolling(7, min_periods=1).mean())
        
        # Final fallback to 0 only if absolutely necessary
        daily_data['value'] = daily_data['value'].fillna(0)
        
        daily_data = daily_data.reset_index()
        daily_data.columns = ['date', 'value']
        
        print(f"✓ Smart filling applied: {daily_data['value'].isnull().sum()} nulls remaining")
    
    print(f"\n✓ ML Forecasting: Prepared {len(daily_data)} days of data")
    print(f"✓ Date range: {daily_data['date'].min()} to {daily_data['date'].max()}")
    print(f"✓ Target variable range: {daily_data['value'].min():.2f} to {daily_data['value'].max():.2f}")
    print(f"✓ Average daily value: {daily_data['value'].mean():.2f}")
    print("="*60)
    
    return {
        "status": "ready",
        "message": "Time series data prepared successfully",
        "data": daily_data,
        "metadata": {
            "date_column": date_col,
            "target_column": target_col,
            "quantity_column": quantity_col,
            "price_column": price_col,
            "start_date": str(daily_data['date'].min()),
            "end_date": str(daily_data['date'].max()),
            "total_days": len(daily_data),
            "avg_daily_value": float(daily_data['value'].mean()),
            "total_records_processed": len(ts_df)
        }
    }


def run_retail_analysis(df):
    """
    Main function to run retail-specific analysis
    
    Args:
        df: Input dataframe with retail data
    
    Returns:
        dict with retail analysis results
    """
    print("\n" + "="*60)
    print("Running Retail Analysis...")
    print("="*60)
    
    results = {}
    
    # Basic dataset overview
    results['overview'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns)
    }
    
    # Data types
    results['data_types'] = df.dtypes.to_dict()
    
    # Missing values
    results['missing_values'] = df.isnull().sum().to_dict()
    
    # =========================================================
    # Calculate Key Metrics (with safe column detection)
    # =========================================================
    print("\nCalculating key metrics...")
    
    # Initialize with default values
    results['total_revenue'] = 0.0
    results['total_orders'] = len(df)
    results['average_order_value'] = 0.0
    results['total_profit'] = 0.0
    results['profit_margin'] = 0.0
    
    # Try to find revenue/sales column
    revenue_col = None
    revenue_keywords = ['revenue', 'sales', 'amount', 'total', 'turnover']
    for keyword in revenue_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    revenue_col = col
                    break
        if revenue_col:
            break
    
    if revenue_col:
        results['total_revenue'] = float(df[revenue_col].sum())
        results['average_order_value'] = float(df[revenue_col].mean())
        print(f"✓ Found revenue column '{revenue_col}': Total = {results['total_revenue']:.2f}")
    else:
        print("⚠️ No revenue column found, using default values")
    
    # Try to find profit column
    profit_col = None
    profit_keywords = ['profit', 'earnings', 'gain']
    for keyword in profit_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    profit_col = col
                    break
        if profit_col:
            break
    
    if profit_col:
        results['total_profit'] = float(df[profit_col].sum())
        if results['total_revenue'] > 0:
            results['profit_margin'] = float((results['total_profit'] / results['total_revenue']) * 100)
        print(f"✓ Found profit column '{profit_col}': Total = {results['total_profit']:.2f}")
    else:
        print("⚠️ No profit column found, calculating from revenue - cost if available")
        # Try to calculate profit from revenue and cost
        cost_col = None
        cost_keywords = ['cost', 'expense', 'cogs']
        for keyword in cost_keywords:
            for col in df.columns:
                if keyword.lower() in col.lower():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        cost_col = col
                        break
            if cost_col:
                break
        
        if revenue_col and cost_col:
            results['total_profit'] = float(df[revenue_col].sum() - df[cost_col].sum())
            if results['total_revenue'] > 0:
                results['profit_margin'] = float((results['total_profit'] / results['total_revenue']) * 100)
            print(f"✓ Calculated profit from revenue - cost: {results['total_profit']:.2f}")
    
    # =========================================================
    # Generate Insights (with safe column detection)
    # =========================================================
    print("\nGenerating insights...")
    
    results['insights'] = {
        'best_store': 'N/A',
        'top_product': 'N/A',
        'best_category': 'N/A',
        'peak_sales_day': 'N/A'
    }
    
    # Find store column
    store_col = None
    store_keywords = ['store', 'location', 'branch', 'outlet', 'city']
    for keyword in store_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                store_col = col
                break
        if store_col:
            break
    
    print(f"DEBUG: Detected store column = {store_col}")
    print(f"DEBUG: Revenue column = {revenue_col}")
    
    if store_col and revenue_col:
        # Clean data - drop rows where store is null
        df_clean = df.dropna(subset=[store_col])
        print(f"DEBUG: Original rows = {len(df)}, Clean rows = {len(df_clean)}")
        
        # Calculate store performance
        store_perf = df_clean.groupby(store_col)[revenue_col].sum()
        print(f"DEBUG: Store sales breakdown:\n{store_perf}")
        
        if len(store_perf) > 0:
            results['insights']['best_store'] = str(store_perf.idxmax())
            print(f"✓ Best store: {results['insights']['best_store']} (Sales: ${store_perf.max():,.2f})")
        else:
            print("⚠️ No valid store data found after grouping")
    else:
        if not store_col:
            print("⚠️ Store column not found. Checked: " + ', '.join(store_keywords))
        if not revenue_col:
            print("⚠️ Revenue column not found")
    
    # Find product column
    product_col = None
    product_keywords = ['product', 'item', 'sku', 'product_name', 'item_name']
    for keyword in product_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                product_col = col
                break
        if product_col:
            break
    
    if product_col and revenue_col:
        product_perf = df.groupby(product_col)[revenue_col].sum()
        if len(product_perf) > 0:
            results['insights']['top_product'] = str(product_perf.idxmax())
            print(f"✓ Top product: {results['insights']['top_product']}")
    
    # Find category column
    category_col = None
    category_keywords = ['category', 'cat', 'department', 'dept', 'class']
    for keyword in category_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                category_col = col
                break
        if category_col:
            break
    
    if category_col and revenue_col:
        cat_perf = df.groupby(category_col)[revenue_col].sum()
        if len(cat_perf) > 0:
            results['insights']['best_category'] = str(cat_perf.idxmax())
            print(f"✓ Best category: {results['insights']['best_category']}")
    
    # Find date column for peak sales day
    date_col = None
    date_keywords = ['date', 'time', 'timestamp']
    for keyword in date_keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                date_col = col
                break
        if date_col:
            break
    
    if date_col and revenue_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            daily_sales = df.groupby(date_col)[revenue_col].sum()
            if len(daily_sales) > 0:
                peak_day = daily_sales.idxmax()
                results['insights']['peak_sales_day'] = str(peak_day)
                print(f"✓ Peak sales day: {results['insights']['peak_sales_day']}")
        except Exception as e:
            print(f"⚠️ Could not calculate peak sales day: {e}")
    
    # =========================================================
    # Detailed Performance Metrics (DataFrames)
    # =========================================================
    print("\nCalculating performance metrics...")
    
    # Store performance
    if store_col and revenue_col:
        results['store_performance'] = df.groupby(store_col).agg({
            revenue_col: ['sum', 'mean', 'count']
        }).round(2)
        results['store_performance'].columns = ['Total_Sales', 'Avg_Sales', 'Transaction_Count']
        results['store_performance'] = results['store_performance'].sort_values('Total_Sales', ascending=False)
    else:
        results['store_performance'] = None
    
    # Top products
    if product_col and revenue_col:
        results['top_products'] = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(10)
        results['top_products'] = results['top_products'].reset_index()
        results['top_products'].columns = [product_col, 'Revenue']
    else:
        results['top_products'] = None
    
    # Category performance
    if category_col and revenue_col:
        results['category_performance'] = df.groupby(category_col).agg({
            revenue_col: ['sum', 'mean']
        }).round(2)
        results['category_performance'].columns = ['Total_Sales', 'Avg_Sales']
        results['category_performance'] = results['category_performance'].sort_values('Total_Sales', ascending=False)
    else:
        results['category_performance'] = None
    
    # Sales trend
    if date_col and revenue_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            results['sales_trend'] = df.groupby(date_col)[revenue_col].sum().sort_index()
            results['sales_trend'] = results['sales_trend'].reset_index()
            results['sales_trend'].columns = [date_col, 'Sales']
        except Exception as e:
            print(f"⚠️ Could not calculate sales trend: {e}")
            results['sales_trend'] = None
    else:
        results['sales_trend'] = None
    
    # Try to prepare time series data for ML forecasting
    ts_result = prepare_time_series_data(df, user_mapping=None)
    results['ml_forecasting_readiness'] = {
        'status': ts_result['status'],
        'message': ts_result['message']
    }
    
    if ts_result['status'] == 'ready':
        results['time_series_metadata'] = ts_result['metadata']
    
    # Run sales forecasting using Linear Regression
    print("\n[STEP 7] Running ML forecasting...")
    forecast = run_sales_forecasting(df)
    results['forecast'] = forecast
    print(f"FORECAST: {forecast}")
    
    # Generate business recommendations
    print("\n[STEP 8] Generating business recommendations...")
    recommendations = generate_recommendations(results, df)
    results['recommendations'] = recommendations
    print(f"Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. {rec}")
    if len(recommendations) > 5:
        print(f"  ... and {len(recommendations) - 5} more")
    
    print("\n✓ Retail Analysis completed")
    print(f"Total Revenue: {results['total_revenue']:.2f}")
    print(f"Total Orders: {results['total_orders']}")
    print(f"Average Order Value: {results['average_order_value']:.2f}")
    print("="*60)
    
    return results


def run_sales_forecasting(df):
    """
    ML forecasting using Facebook Prophet with Linear Regression fallback
    
    Args:
        df: Input dataframe with date and sales columns
    
    Returns:
        dict with future_dates and predictions
    """
    print("\n" + "="*60)
    print("Running Sales Forecasting")
    print("="*60)
    print("🔍 Checking Prophet availability...")
    
    # Check if Prophet is available
    prophet_available = False
    try:
        from prophet import Prophet
        print("✓ Prophet library is available")
        prophet_available = True
    except ImportError as e:
        print(f"✗ Prophet library NOT available: {e}")
        print("🔄 Will use Enhanced Linear Regression fallback")
        prophet_available = False
    
    # Create working copy
    forecast_df = df.copy()
    
    # Find date and sales columns
    date_col = None
    sales_col = None
    
    # Search for date column
    date_keywords = ['date', 'order_date', 'transaction_date', 'time']
    for keyword in date_keywords:
        for col in forecast_df.columns:
            if keyword.lower() in col.lower():
                date_col = col
                break
        if date_col:
            break
    
    # Search for sales/revenue column
    sales_keywords = ['sales', 'revenue', 'amount', 'total']
    for keyword in sales_keywords:
        for col in forecast_df.columns:
            if keyword.lower() in col.lower():
                if pd.api.types.is_numeric_dtype(forecast_df[col]):
                    sales_col = col
                    break
        if sales_col:
            break
    
    # Validate columns found
    if not date_col or not sales_col:
        print(f"⚠️ Cannot run forecasting: Missing required columns")
        print(f"  Date column: {date_col}")
        print(f"  Sales column: {sales_col}")
        return {
            "future_dates": [],
            "predictions": []
        }
    
    print(f"✓ Using date column: {date_col}")
    print(f"✓ Using sales column: {sales_col}")
    
    # Convert date to datetime
    try:
        forecast_df[date_col] = pd.to_datetime(forecast_df[date_col], errors='coerce')
        forecast_df = forecast_df.dropna(subset=[date_col, sales_col])
        
        if len(forecast_df) < 2:
            print(f"⚠️ Insufficient data for forecasting (need at least 2 rows)")
            return {
                "future_dates": [],
                "predictions": []
            }
        
        # Sort by date
        forecast_df = forecast_df.sort_values(date_col)
        
        # Aggregate to daily level
        daily_sales = forecast_df.groupby(date_col)[sales_col].sum().reset_index()
        
        # Add feature engineering for fallback models (Prophet handles internally)
        daily_sales['day_of_week'] = daily_sales[date_col].dt.dayofweek
        daily_sales['month'] = daily_sales[date_col].dt.month
        daily_sales['quarter'] = daily_sales[date_col].dt.quarter
        daily_sales['is_weekend'] = daily_sales['day_of_week'].isin([5, 6]).astype(int)
        
        # Add lag features (for fallback models)
        daily_sales['lag_1'] = daily_sales[sales_col].shift(1)
        daily_sales['lag_7'] = daily_sales[sales_col].shift(7)
        daily_sales['lag_30'] = daily_sales[sales_col].shift(30)
        
        # Add rolling statistics (for fallback models)
        daily_sales['rolling_mean_7'] = daily_sales[sales_col].rolling(7, min_periods=1).mean()
        daily_sales['rolling_mean_30'] = daily_sales[sales_col].rolling(30, min_periods=1).mean()
        daily_sales['rolling_std_7'] = daily_sales[sales_col].rolling(7, min_periods=1).std()
        
        print(f"✓ Added {len(daily_sales.columns)} features including lag, rolling, and time-based features")
        
        # Check if we have enough data for Prophet (minimum 2 data points)
        if len(daily_sales) < 2:
            print(f"⚠️ Insufficient daily data points ({len(daily_sales)}), falling back to Linear Regression")
            return _fallback_linear_regression(daily_sales, date_col, sales_col)
        
        print(f"✓ Prepared {len(daily_sales)} daily data points for Prophet")
        
        # Prepare dataframe for Prophet (must have columns 'ds' and 'y')
        prophet_df = pd.DataFrame({
            'ds': daily_sales[date_col],
            'y': daily_sales[sales_col]
        })
        
        # Initialize and train Prophet model with improved configuration
        try:
            from prophet import Prophet
            
            print("🔮 Training Facebook Prophet model with improved configuration...")
            model = Prophet(
                daily_seasonality=True,           # Enable daily seasonality
                weekly_seasonality=True,          # Enable weekly seasonality  
                yearly_seasonality=True,          # Enable yearly seasonality
                changepoint_prior_scale=0.1,     # More flexible trend changes (was 0.05)
                seasonality_prior_scale=10.0,    # Stronger seasonality detection (new)
                seasonality_mode='multiplicative', # Better for sales data (new)
                interval_width=0.95,
                mcmc_samples=0,                  # Faster fitting (new)
                uncertainty_samples=1000         # Better uncertainty estimates (new)
            )
            
            # Fit the model
            model.fit(prophet_df)
            print("✓ Prophet model trained successfully")
            
            # Create future dataframe for next 30 days
            future = model.make_future_dataframe(periods=30, freq='D')
            
            # Make predictions
            forecast = model.predict(future)
            
            # Extract only future dates (last 30 rows)
            future_forecast = forecast.tail(30)
            
            # Extract dates and predictions with uncertainty intervals
            future_dates = future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
            predictions = future_forecast['yhat'].fillna(0).tolist()
            lower_bound = future_forecast['yhat_lower'].fillna(0).tolist()
            upper_bound = future_forecast['yhat_upper'].fillna(0).tolist()
            
            # Ensure non-negative predictions
            predictions = [max(0, float(p)) for p in predictions]
            lower_bound = [max(0, float(p)) for p in lower_bound]
            upper_bound = [max(0, float(p)) for p in upper_bound]
            
            # Calculate forecast quality metrics
            historical_mean = daily_sales[sales_col].mean()
            historical_std = daily_sales[sales_col].std()
            forecast_mean = np.mean(predictions)
            forecast_std = np.std(predictions)
            
            # Detect seasonality
            has_weekly_pattern = len(daily_sales) >= 14
            has_monthly_pattern = len(daily_sales) >= 30
            
            result = {
                "future_dates": future_dates,
                "predictions": predictions,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "model_used": "Prophet",
                "confidence_interval": 0.95,
                "diagnostics": {
                    "training_data_size": len(daily_sales),
                    "historical_mean": float(historical_mean),
                    "historical_std": float(historical_std),
                    "forecast_mean": float(forecast_mean),
                    "forecast_std": float(forecast_std),
                    "has_weekly_pattern": has_weekly_pattern,
                    "has_monthly_pattern": has_monthly_pattern,
                    "date_range": f"{daily_sales[date_col].min()} to {daily_sales[date_col].max()}"
                }
            }
            
            print(f"✓ Generated Prophet forecasts for {len(future_dates)} days")
            print(f"  First prediction: {predictions[0]:.2f} ({future_dates[0]})")
            print(f"  Last prediction: {predictions[-1]:.2f} ({future_dates[-1]})")
            print(f"  Historical mean: {historical_mean:.2f}, Forecast mean: {forecast_mean:.2f}")
            print(f"  Weekly pattern detected: {has_weekly_pattern}, Monthly pattern detected: {has_monthly_pattern}")
            print("="*60)
            
            return result
            
        except ImportError as e:
            print(f"⚠️ Prophet import failed: {e}")
            print("🔄 Falling back to Linear Regression")
            return _fallback_linear_regression(daily_sales, date_col, sales_col)
            
        except Exception as e:
            print(f"⚠️ Prophet training/prediction failed: {e}")
            print("🔄 Falling back to Linear Regression")
            return _fallback_linear_regression(daily_sales, date_col, sales_col)
        
    except Exception as e:
        print(f"✗ Forecasting error: {e}")
        return {
            "future_dates": [],
            "predictions": []
        }


def _fallback_linear_regression(daily_sales, date_col, sales_col):
    """
    Enhanced Linear Regression with feature engineering for fallback
    
    Args:
        daily_sales: DataFrame with aggregated daily sales and features
        date_col: Name of date column
        sales_col: Name of sales column
    
    Returns:
        dict with future_dates and predictions
    """
    print("\n" + "="*60)
    print("Running Enhanced Linear Regression Fallback")
    print("="*60)
    
    try:
        if len(daily_sales) < 2:
            print(f"⚠️ Insufficient data for fallback")
            return {
                "future_dates": [],
                "predictions": [],
                "model_used": "LinearRegression",
                "error": "Insufficient data"
            }
        
        daily_sales = daily_sales.copy()
        
        # Convert date to ordinal
        daily_sales['date_ordinal'] = daily_sales[date_col].map(pd.Timestamp.toordinal)
        
        # Add polynomial features for non-linear trends
        daily_sales['date_ordinal_sq'] = daily_sales['date_ordinal'] ** 2
        
        # Fill missing lag/rolling features with mean
        if 'lag_7' in daily_sales.columns:
            daily_sales['lag_7'] = daily_sales['lag_7'].fillna(daily_sales['lag_7'].mean())
        if 'rolling_mean_7' in daily_sales.columns:
            daily_sales['rolling_mean_7'] = daily_sales['rolling_mean_7'].fillna(daily_sales['rolling_mean_7'].mean())
        
        # Select features
        feature_cols = ['date_ordinal', 'date_ordinal_sq']
        if 'lag_7' in daily_sales.columns and not daily_sales['lag_7'].isnull().all():
            feature_cols.append('lag_7')
        if 'rolling_mean_7' in daily_sales.columns and not daily_sales['rolling_mean_7'].isnull().all():
            feature_cols.append('rolling_mean_7')
        
        # Prepare training data
        X = daily_sales[feature_cols].values
        y = daily_sales[sales_col].values
        
        # Train Linear Regression
        model = LinearRegression()
        model.fit(X, y)
        
        r2_score = model.score(X, y)
        
        print(f"✓ Enhanced Linear Regression trained on {len(daily_sales)} data points")
        print(f"  Features used: {feature_cols}")
        print(f"  R² score: {r2_score:.4f}")
        
        # Generate future dates
        last_date = daily_sales[date_col].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
        future_ordinals = [d.toordinal() for d in future_dates]
        future_ordinals_sq = [o ** 2 for o in future_ordinals]
        
        # Prepare future features
        future_X = []
        for i in range(len(future_ordinals)):
            row = [future_ordinals[i], future_ordinals_sq[i]]
            if 'lag_7' in feature_cols:
                # Use last 7-day average as lag proxy
                row.append(daily_sales[sales_col].tail(7).mean())
            if 'rolling_mean_7' in feature_cols:
                row.append(daily_sales[sales_col].tail(7).mean())
            future_X.append(row)
        
        # Predict
        predictions = model.predict(np.array(future_X))
        
        # Ensure non-negative
        predictions = [max(0, p) for p in predictions]
        
        # Calculate forecast quality metrics
        historical_mean = daily_sales[sales_col].mean()
        historical_std = daily_sales[sales_col].std()
        forecast_mean = np.mean(predictions)
        forecast_std = np.std(predictions)
        
        result = {
            "future_dates": [d.strftime('%Y-%m-%d') for d in future_dates],
            "predictions": [float(p) for p in predictions],
            "model_used": "Enhanced Linear Regression",
            "features": feature_cols,
            "r2_score": r2_score,
            "diagnostics": {
                "training_data_size": len(daily_sales),
                "historical_mean": float(historical_mean),
                "historical_std": float(historical_std),
                "forecast_mean": float(forecast_mean),
                "forecast_std": float(forecast_std),
                "r2_score": r2_score,
                "date_range": f"{daily_sales[date_col].min()} to {daily_sales[date_col].max()}"
            }
        }
        
        print(f"✓ Generated enhanced fallback forecasts for {len(future_dates)} days")
        print(f"  First prediction: {predictions[0]:.2f} ({future_dates[0].strftime('%Y-%m-%d')})")
        print(f"  Last prediction: {predictions[-1]:.2f} ({future_dates[-1].strftime('%Y-%m-%d')})")
        print(f"  Historical mean: {historical_mean:.2f}, Forecast mean: {forecast_mean:.2f}")
        print(f"  R² score: {r2_score:.4f}")
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"✗ Enhanced fallback error: {e}")
        return {
            "future_dates": [],
            "predictions": [],
            "model_used": "LinearRegression",
            "error": str(e)
        }


def generate_recommendations(results, df):
    """
    Generate comprehensive business recommendations based on analysis results
    
    Args:
        results: Dictionary containing retail analysis results
        df: Original dataframe
    
    Returns:
        List of recommendation strings
    """
    print("\n" + "="*60)
    print("Generating Business Recommendations...")
    print("="*60)
    
    recommendations = []
    
    # =========================================================
    # 1. PRODUCT-BASED RECOMMENDATIONS
    # =========================================================
    print("\n[1] Analyzing product performance...")
    
    if results.get('insights', {}).get('top_product') != 'N/A':
        top_product = results['insights']['top_product']
        recommendations.append(
            f"🏆 Promote high-performing product '{top_product}' through featured placement and marketing campaigns"
        )
        
        # Check if we have top_products DataFrame for more details
        top_products_df = results.get('top_products')
        if top_products_df is not None:
            try:
                import pandas as pd
                if isinstance(top_products_df, pd.DataFrame) and len(top_products_df) > 0:
                    # Get top 3 products
                    top_3 = top_products_df.head(3)
                    if len(top_3) >= 3:
                        product_names = top_3.iloc[:, 0].tolist()
                        recommendations.append(
                            f"📦 Create bundle offers with top products: {', '.join(product_names[:3])}"
                        )
            except Exception:
                pass
    
    # =========================================================
    # 2. CATEGORY-BASED RECOMMENDATIONS
    # =========================================================
    print("\n[2] Analyzing category performance...")
    
    if results.get('insights', {}).get('best_category') != 'N/A':
        best_category = results['insights']['best_category']
        recommendations.append(
            f"⭐ Expand inventory in best-performing category '{best_category}'"
        )
    
    # Check for underperforming categories
    category_perf = results.get('category_performance')
    if category_perf is not None:
        try:
            import pandas as pd
            if isinstance(category_perf, pd.DataFrame) and len(category_perf) > 1:
                # Get lowest performing category
                if 'Total_Sales' in category_perf.columns:
                    lowest_cat = category_perf.iloc[-1].name
                    if lowest_cat != best_category:
                        recommendations.append(
                            f"📉 Review and improve underperforming category '{lowest_cat}' - consider promotions or discontinuation"
                        )
        except Exception:
            pass
    
    # =========================================================
    # 3. STORE-BASED RECOMMENDATIONS
    # =========================================================
    print("\n[3] Analyzing store performance...")
    
    if results.get('insights', {}).get('best_store') != 'N/A':
        best_store = results['insights']['best_store']
        recommendations.append(
            f"🏪 Replicate success strategies from top-performing store '{best_store}' to other locations"
        )
        
        # Check for multiple stores
        store_perf = results.get('store_performance')
        if store_perf is not None:
            try:
                import pandas as pd
                if isinstance(store_perf, pd.DataFrame) and len(store_perf) > 1:
                    recommendations.append(
                        "📊 Implement cross-store knowledge sharing program to spread best practices"
                    )
            except Exception:
                pass
    
    # =========================================================
    # 4. SALES TREND RECOMMENDATIONS
    # =========================================================
    print("\n[4] Analyzing sales trends...")
    
    # Check forecast trend
    forecast = results.get('forecast', {})
    if forecast and len(forecast.get('predictions', [])) > 0:
        predictions = forecast['predictions']
        if len(predictions) >= 10:
            # Compare first week vs last week predictions
            first_week_avg = sum(predictions[:7]) / 7
            last_week_avg = sum(predictions[-7:]) / 7
            
            if last_week_avg > first_week_avg * 1.1:  # 10% increase
                recommendations.append(
                    "📈 Sales trend is increasing - consider scaling up inventory and staff capacity"
                )
            elif last_week_avg < first_week_avg * 0.9:  # 10% decrease
                recommendations.append(
                    "⚠️ Sales trend shows decline - implement promotional campaigns to boost demand"
                )
            else:
                recommendations.append(
                    "➡️ Sales trend is stable - maintain current inventory levels and monitor closely"
                )
    
    # =========================================================
    # 5. PROFIT-BASED RECOMMENDATIONS
    # =========================================================
    print("\n[5] Analyzing profitability...")
    
    profit_margin = results.get('profit_margin', 0)
    if profit_margin > 0:
        if profit_margin > 30:
            recommendations.append(
                f"💰 Excellent profit margin ({profit_margin:.1f}%) - consider reinvesting profits in growth initiatives"
            )
        elif profit_margin > 15:
            recommendations.append(
                f"✅ Good profit margin ({profit_margin:.1f}%) - explore opportunities to optimize costs further"
            )
        else:
            recommendations.append(
                f"⚠️ Low profit margin ({profit_margin:.1f}%) - review pricing strategy and reduce operational costs"
            )
    
    # Check if profit can be improved
    total_profit = results.get('total_profit', 0)
    total_revenue = results.get('total_revenue', 0)
    if total_profit > 0 and total_revenue > 0:
        if total_profit < total_revenue * 0.1:  # Less than 10% of revenue
            recommendations.append(
                "💡 Focus on high-margin products to improve overall profitability"
            )
    
    # =========================================================
    # 6. REVENUE INSIGHTS
    # =========================================================
    print("\n[6] Analyzing revenue metrics...")
    
    total_revenue = results.get('total_revenue', 0)
    avg_order_value = results.get('average_order_value', 0)
    total_orders = results.get('total_orders', 0)
    
    if avg_order_value > 0:
        if avg_order_value < 50:
            recommendations.append(
                f"🛒 Average order value (${avg_order_value:.2f}) is low - implement upselling strategies to increase basket size"
            )
        elif avg_order_value > 200:
            recommendations.append(
                f"💎 High average order value (${avg_order_value:.2f}) - introduce loyalty program to retain valuable customers"
            )
        else:
            recommendations.append(
                f"📊 Moderate average order value (${avg_order_value:.2f}) - offer bundle deals to increase transaction size"
            )
    
    if total_orders > 0:
        recommendations.append(
            f"📦 Process volume: {total_orders} orders - ensure operational efficiency meets customer expectations"
        )
    
    # =========================================================
    # 7. FORECAST-BASED RECOMMENDATIONS
    # =========================================================
    print("\n[7] Generating forecast-based insights...")
    
    if forecast and len(forecast.get('predictions', [])) > 0:
        predictions = forecast['predictions']
        max_prediction = max(predictions)
        min_prediction = min(predictions)
        
        if max_prediction > 0:
            recommendations.append(
                f"🔮 Peak predicted demand: ${max_prediction:.2f} - prepare inventory and logistics accordingly"
            )
        
        if len(predictions) > 15:
            # Identify volatility
            avg_prediction = sum(predictions) / len(predictions)
            volatility = (max_prediction - min_prediction) / avg_prediction
            
            if volatility > 0.3:  # High volatility
                recommendations.append(
                    "⚡ High demand variability predicted - maintain flexible inventory management"
                )
    
    # =========================================================
    # 8. DATA QUALITY RECOMMENDATIONS
    # =========================================================
    print("\n[8] Checking data quality...")
    
    missing_values = results.get('missing_values', {})
    if missing_values:
        total_missing = sum(missing_values.values())
        total_rows = results.get('overview', {}).get('rows', 1)
        missing_pct = (total_missing / (total_rows * len(missing_values))) * 100
        
        if missing_pct > 5:
            recommendations.append(
                f"🔍 Data quality: {missing_pct:.1f}% missing values - improve data collection to enhance analysis accuracy"
            )
    
    # =========================================================
    # 9. SEASONAL/TEMPORAL INSIGHTS
    # =========================================================
    print("\n[9] Analyzing temporal patterns...")
    
    if results.get('insights', {}).get('peak_sales_day') != 'N/A':
        peak_day = results['insights']['peak_sales_day']
        recommendations.append(
            f"📅 Peak sales occurred on {peak_day} - analyze factors contributing to this success"
        )
        recommendations.append(
            "🗓️ Plan marketing campaigns and inventory around identified peak periods"
        )
    
    # =========================================================
    # 10. STRATEGIC RECOMMENDATIONS
    # =========================================================
    print("\n[10] Generating strategic insights...")
    
    # Add general strategic advice based on overall metrics
    if total_revenue > 0 and total_orders > 0:
        revenue_per_order = total_revenue / total_orders
        if revenue_per_order > 100:
            recommendations.append(
                "🎯 High-value customer base - focus on premium service and retention strategies"
            )
        else:
            recommendations.append(
                "👥 Volume-driven business - optimize for efficiency and customer acquisition"
            )
    
    # Always add continuous improvement recommendation
    recommendations.append(
        "🔄 Continuously monitor KPIs and adjust strategies based on real-time performance data"
    )
    
    print("\n" + "="*60)
    print(f"✓ Generated {len(recommendations)} recommendations")
    print("="*60)
    
    return recommendations