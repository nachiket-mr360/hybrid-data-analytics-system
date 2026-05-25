"""
Advanced Data Cleaning Module
Provides comprehensive data cleaning with detection, transformation, and reporting capabilities.
"""
import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Optional, Tuple, Any


def safe_print(msg: str):
    """
    Safely print messages to console, handling encoding issues on Windows.
    
    Args:
        msg: Message to print
    """
    try:
        # Try printing directly
        print(msg)
    except UnicodeEncodeError:
        # If that fails, encode/decode to ASCII (ignore errors)
        safe_msg = msg.encode('ascii', 'ignore').decode('ascii')
        print(safe_msg)


class AdvancedDataCleaner:
    """Advanced data cleaning with automatic detection and intelligent handling."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the advanced data cleaner.
        
        Args:
            df: Input DataFrame to clean
        """
        self.original_df = df.copy()
        self.cleaned_df = df.copy()
        self.column_types = {}
        self.missing_report = {}
        self.conversion_report = {}
        self.outlier_report = {}
        self.cleaning_actions = []
        
        # ENHANCED INTELLIGENCE TRACKING
        self.decision_log = []  # All cleaning decisions with reasoning
        self.ai_insights = []   # High-level insights about data quality
        self.warnings = []      # Critical warnings requiring attention
        self.duplicates_found = 0
        self.negative_values_found = {}
        self.skewness_report = {}
        self.datatype_issues = []
    
    def log(self, message: str):
        """
        Centralized logging method for controlled visibility.
        All important execution steps use this instead of raw print().
        
        Args:
            message: Log message to display with [CLEANER] prefix
        """
        print(f"[CLEANER] {message}")
    
    def detect_column_types(self) -> Dict[str, str]:
        """
        Automatically detect column types (numeric, datetime, categorical, text).
        
        Returns:
            Dictionary mapping column names to detected types
        """
        print("\n=== DETECTING COLUMN TYPES ===")
        type_detection = {}
        
        for col in self.cleaned_df.columns:
            col_data = self.cleaned_df[col].dropna()
            
            if len(col_data) == 0:
                type_detection[col] = 'empty'
                print(f"  {col}: EMPTY (no data)")
                continue
            
            # Check if numeric
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                type_detection[col] = 'numeric'
                print(f"  {col}: NUMERIC")
                continue
            
            # Try to convert to numeric (might be stored as strings)
            try:
                converted = pd.to_numeric(col_data, errors='coerce')
                non_null_ratio = converted.notna().sum() / len(converted)
                
                if non_null_ratio > 0.8:  # 80% successfully converted
                    type_detection[col] = 'numeric_convertible'
                    print(f"  {col}: NUMERIC_CONVERTIBLE ({non_null_ratio*100:.1f}% convertible)")
                    continue
            except:
                pass
            
            # Try to convert to datetime with format inference
            try:
                # Infer format from first few non-null values
                sample_dates = col_data.head(10)
                inferred_format = None
                
                for date_str in sample_dates:
                    if pd.isna(date_str):
                        continue
                    
                    # Try common formats
                    formats_to_try = [
                        '%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d/%m/%Y',
                        '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ'
                    ]
                    
                    for fmt in formats_to_try:
                        try:
                            pd.to_datetime(str(date_str), format=fmt)
                            inferred_format = fmt
                            break
                        except:
                            continue
                    
                    if inferred_format:
                        break
                
                # Convert using inferred format
                if inferred_format:
                    converted = pd.to_datetime(col_data, format=inferred_format, errors='coerce')
                else:
                    converted = pd.to_datetime(col_data, errors='coerce')
                
                non_null_ratio = converted.notna().sum() / len(converted)
                
                if non_null_ratio > 0.8:  # 80% successfully converted
                    type_detection[col] = 'datetime_convertible'
                    format_info = f", format='{inferred_format}'" if inferred_format else ""
                    print(f"  {col}: DATETIME_CONVERTIBLE ({non_null_ratio*100:.1f}% convertible{format_info})")
                    continue
            except:
                pass
            
            # Check if categorical (few unique values)
            unique_ratio = col_data.nunique() / len(col_data)
            if unique_ratio < 0.05 or col_data.nunique() < 20:
                type_detection[col] = 'categorical'
                print(f"  {col}: CATEGORICAL ({col_data.nunique()} unique values)")
            else:
                type_detection[col] = 'text'
                print(f"  {col}: TEXT ({col_data.nunique()} unique values)")
        
        self.column_types = type_detection
        return type_detection
    
    def analyze_missing_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze missing values in each column.
        
        Returns:
            Dictionary with missing value statistics per column
        """
        print("\n=== ANALYZING MISSING VALUES ===")
        missing_report = {}
        
        total_rows = len(self.cleaned_df)
        
        for col in self.cleaned_df.columns:
            missing_count = self.cleaned_df[col].isnull().sum()
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            
            if missing_count > 0:
                missing_report[col] = {
                    'count': int(missing_count),
                    'percentage': round(float(missing_pct), 2),
                    'strategy': None  # To be filled by user choice
                }
                print(f"  {col}: {missing_count} missing ({missing_pct:.2f}%)")
        
        self.missing_report = missing_report
        return missing_report
    
    def detect_and_handle_duplicates(self) -> Dict[str, Any]:
        """
        Detect and handle duplicate rows.
        
        Returns:
            Dictionary with duplicate statistics
        """
        print("\n=== DETECTING DUPLICATES ===")
        
        total_rows = len(self.cleaned_df)
        duplicates = self.cleaned_df.duplicated().sum()
        duplicate_pct = (duplicates / total_rows * 100) if total_rows > 0 else 0
        
        result = {
            'total_rows': int(total_rows),
            'duplicates_found': int(duplicates),
            'duplicate_percentage': round(float(duplicate_pct), 2)
        }
        
        self.duplicates_found = int(duplicates)
        
        if duplicates > 0:
            # Log decision for duplicate removal
            confidence = min(0.95, 0.7 + (duplicate_pct / 100))  # Higher % → higher confidence
            
            self.decision_log.append({
                'column': 'entire_dataset',
                'strategy': 'duplicate_removal',
                'confidence': round(confidence, 2),
                'reason': f'{duplicates} duplicate rows detected ({duplicate_pct:.1f}%), removed to ensure data integrity'
            })
            
            # Remove duplicates
            self.cleaned_df = self.cleaned_df.drop_duplicates()
            rows_after = len(self.cleaned_df)
            
            print(f"  Found {duplicates} duplicates ({duplicate_pct:.1f}%)")
            print(f"  Removed {total_rows - rows_after} duplicate rows")
            
            # Add insight
            if duplicate_pct > 5:
                self.ai_insights.append(
                    f"High duplication rate ({duplicate_pct:.1f}%) suggests possible data collection issues or repeated entries"
                )
        else:
            print("  No duplicates found")
        
        return result
    
    def detect_negative_values(self, numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect negative values in numeric columns with business context analysis.
        
        Args:
            numeric_columns: Specific columns to check (all numeric if None)
        
        Returns:
            Dictionary with negative value statistics and business impact
        """
        print("\n=== DETECTING NEGATIVE VALUES ===")
        
        if numeric_columns is None:
            numeric_columns = [col for col in self.cleaned_df.columns 
                             if pd.api.types.is_numeric_dtype(self.cleaned_df[col])]
        
        negative_info = {}
        
        for col in numeric_columns:
            negative_count = (self.cleaned_df[col] < 0).sum()
            
            if negative_count > 0:
                negative_pct = (negative_count / len(self.cleaned_df[col]) * 100)
                negative_info[col] = {
                    'count': int(negative_count),
                    'percentage': round(float(negative_pct), 2),
                    'min_value': float(self.cleaned_df[col].min()),
                    'mean_of_negatives': float(self.cleaned_df[self.cleaned_df[col] < 0][col].mean()),
                    'total_impact': float(self.cleaned_df[self.cleaned_df[col] < 0][col].sum())
                }
                
                self.negative_values_found[col] = negative_info[col]
                
                # DYNAMIC CONFIDENCE: Based on percentage AND business context
                base_confidence = 0.5
                pct_factor = min(0.3, negative_pct / 100)  # Higher % → higher confidence
                
                # Business context boost
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['quantity', 'qty', 'units']):
                    business_context_boost = 0.2  # Negative quantity almost always = returns
                elif any(keyword in col_lower for keyword in ['revenue', 'sales', 'profit', 'price']):
                    business_context_boost = 0.15  # Could be refunds or discounts
                else:
                    business_context_boost = 0.05  # Generic numeric column
                
                confidence = min(0.95, base_confidence + pct_factor + business_context_boost)
                
                # ANALYST-LEVEL REASONING: Explain WHY it matters
                if negative_pct > 5:
                    severity = "critical"
                    business_impact = f"This affects {negative_pct:.1f}% of transactions - significant operational issue"
                elif negative_pct > 1:
                    severity = "moderate"
                    business_impact = f"Moderate frequency suggests regular business operations (returns/refunds)"
                else:
                    severity = "low"
                    business_impact = f"Rare occurrence - likely exceptional cases or data entry errors"
                
                # Log decision with enhanced reasoning
                self.decision_log.append({
                    'column': col,
                    'strategy': 'negative_value_detection',
                    'confidence': round(confidence, 2),
                    'reason': f'{negative_count} negative values ({negative_pct:.1f}%) detected - requires business validation',
                    'severity': severity,
                    'business_impact': business_impact
                })
                
                print(f"  {col}: {negative_count} negative values ({negative_pct:.1f}%), min={negative_info[col]['min_value']:.2f}")
        
        # ENHANCED INSIGHTS: Explain WHAT + WHY with business context
        if negative_info:
            for col, info in negative_info.items():
                col_lower = col.lower()
                
                # Context-specific insights with business impact
                if any(keyword in col_lower for keyword in ['quantity', 'qty', 'units']):
                    financial_impact = abs(info['total_impact'])
                    self.ai_insights.append(
                        f"Negative quantities detected ({info['count']} occurrences, {info['percentage']:.1f}% of data) "
                        f"→ Likely product returns or inventory adjustments. "
                        f"Financial impact: ${financial_impact:,.2f} in reversed transactions. "
                        f"Recommend: Analyze return patterns by product/category for quality issues."
                    )
                    
                    if info['percentage'] > 2:
                        self.warnings.append(
                            f"High return rate in {col} ({info['percentage']:.1f}%) - investigate product quality or fulfillment issues"
                        )
                
                elif any(keyword in col_lower for keyword in ['revenue', 'sales', 'profit']):
                    self.ai_insights.append(
                        f"Negative revenue detected ({info['count']} transactions, {info['percentage']:.1f}%) "
                        f"→ Indicates customer refunds, returns, or chargebacks. "
                        f"Total revenue impact: ${info['total_impact']:,.2f}. "
                        f"Business implication: Consider separate tracking for refunds vs discounts for better margin analysis."
                    )
                    
                    if abs(info['total_impact']) > 10000:
                        self.warnings.append(
                            f"Significant refund activity in {col} (${abs(info['total_impact']):,.2f}) - verify cash flow impact"
                        )
                
                elif any(keyword in col_lower for keyword in ['price', 'cost']):
                    self.ai_insights.append(
                        f"Negative pricing detected ({info['count']} records) "
                        f"→ Possible data entry errors, promotional credits, or supplier rebates. "
                        f"Validate against business rules: prices should typically be positive."
                    )
                
                else:
                    self.ai_insights.append(
                        f"Negative values in {col} ({info['count']} occurrences) require business context validation. "
                        f"Range: [{info['min_value']:.2f}, 0]. Determine if this represents valid business scenarios or data quality issues."
                    )
        else:
            print("  ✓ No negative values found in numeric columns")
        
        return negative_info
    
    def analyze_skewness(self, skew_threshold: float = 1.0) -> Dict[str, Any]:
        """
        Analyze distribution skewness in numeric columns.
        
        Args:
            skew_threshold: Absolute skewness value to flag (default 1.0)
        
        Returns:
            Dictionary with skewness statistics
        """
        print("\n=== ANALYZING SKEWNESS ===")
        
        numeric_columns = [col for col in self.cleaned_df.columns 
                         if pd.api.types.is_numeric_dtype(self.cleaned_df[col])]
        
        skewness_report = {}
        
        for col in numeric_columns:
            col_data = self.cleaned_df[col].dropna()
            
            if len(col_data) > 10:  # Need minimum data points
                skewness = col_data.skew()
                
                if abs(skewness) > skew_threshold:
                    direction = 'right' if skewness > 0 else 'left'
                    severity = 'extreme' if abs(skewness) > 3 else 'moderate'
                    
                    skewness_report[col] = {
                        'skewness': round(float(skewness), 2),
                        'direction': direction,
                        'severity': severity,
                        'affected_rows': int(len(col_data))
                    }
                    
                    self.skewness_report[col] = skewness_report[col]
                    
                    # Dynamic confidence based on skewness magnitude
                    confidence = min(0.95, 0.75 + (abs(skewness) / 20))
                    
                    # Log decision
                    self.decision_log.append({
                        'column': col,
                        'strategy': 'skewness_detection',
                        'confidence': round(confidence, 2),
                        'reason': f'Data is highly {direction}-skewed (skewness={skewness:.2f}) - may need transformation for analysis'
                    })
                    
                    print(f"  {col}: {direction}-skewed (skewness={skewness:.2f}, {severity})")
        
        # Generate insights
        if skewness_report:
            for col, info in skewness_report.items():
                if info['severity'] == 'extreme':
                    self.ai_insights.append(
                        f"Dataset has extreme {info['direction']} skewness in {col} (skewness={info['skewness']}) - consider log transformation"
                    )
                else:
                    self.warnings.append(
                        f"Moderate skewness detected in {col} - statistical assumptions may be affected"
                    )
        else:
            print("  No significant skewness detected")
        
        return skewness_report
    
    def standardize_datatypes(self) -> List[Dict[str, str]]:
        """
        Standardize data types across columns.
        
        Returns:
            List of datatype standardization actions
        """
        print("\n=== STANDARDIZING DATATYPES ===")
        
        datatype_actions = []
        
        for col in self.cleaned_df.columns:
            original_dtype = str(self.cleaned_df[col].dtype)
            action_taken = None
            
            # Standardize string types
            if self.cleaned_df[col].dtype == 'object':
                # Check if should be datetime
                try:
                    converted = pd.to_datetime(
                                            self.cleaned_df[col],
                                            errors='coerce',
                                            format='mixed'
                                            )
                    conversion_rate = converted.notna().sum() / len(converted)
                    
                    if conversion_rate > 0.8:
                        self.cleaned_df[col] = converted
                        action_taken = f"Converted from object to datetime ({conversion_rate*100:.0f}% successful)"
                        
                        # Add warning for inconsistent formats
                        if conversion_rate < 1.0:
                            self.warnings.append(
                                f"Inconsistent date formats in {col} - {((1-conversion_rate)*100):.0f}% could not be parsed"
                            )
                except:
                    pass
            
            # Standardize integer types for count columns
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['count', 'quantity', 'id', 'number']):
                    if self.cleaned_df[col].notna().all() and (self.cleaned_df[col] % 1 == 0).all():
                        original = self.cleaned_df[col].dtype
                        self.cleaned_df[col] = self.cleaned_df[col].astype('int64')
                        action_taken = f"Converted from {original} to int64 (whole numbers detected)"
            
            if action_taken:
                datatype_actions.append({
                    'column': col,
                    'action': action_taken
                })
                
                # Log decision with dynamic confidence
                confidence = 0.9 if 'datetime' in action_taken.lower() else 0.85
                
                self.decision_log.append({
                    'column': col,
                    'strategy': 'datatype_standardization',
                    'confidence': confidence,
                    'reason': action_taken
                })
                
                print(f"  {col}: {action_taken}")
        
        self.datatype_issues = datatype_actions
        
        if datatype_actions:
            print(f"\nStandardized {len(datatype_actions)} columns")
        else:
            print("  No datatype standardization needed")
        
        return datatype_actions
    
    def convert_columns(self, conversions: Dict[str, str]) -> pd.DataFrame:
        """
        Convert columns to specified types.
        
        Args:
            conversions: Dictionary mapping column names to target types
                        ('numeric', 'datetime', 'category', 'string')
        
        Returns:
            Updated DataFrame
        """
        print("\n=== CONVERTING COLUMNS ===")
        conversion_details = {}
        
        for col, target_type in conversions.items():
            if col not in self.cleaned_df.columns:
                print(f"   Column '{col}' not found, skipping")
                continue
            
            try:
                original_dtype = str(self.cleaned_df[col].dtype)
                
                if target_type == 'numeric':
                    self.cleaned_df[col] = pd.to_numeric(self.cleaned_df[col], errors='coerce')
                    converted_dtype = str(self.cleaned_df[col].dtype)
                    conversion_details[col] = {
                        'from': original_dtype,
                        'to': converted_dtype,
                        'method': 'numeric_conversion'
                    }
                    print(f"  [OK] {col}: {original_dtype} -> {converted_dtype} (numeric)")
                
                elif target_type == 'datetime':
                    # ANALYST-GRADE DATE PROCESSING with strict validation
                    print(f"  [DATE] {col}: Applying strict datetime validation...")
                    
                    # Try multiple parsing strategies
                    converted = None
                    parsing_strategy = None
                    
                    # Strategy 1: dayfirst=True (international format)
                    try:
                        test_conversion = pd.to_datetime(self.cleaned_df[col], errors='coerce', dayfirst=True)
                        valid_pct = test_conversion.notna().sum() / len(test_conversion) * 100
                        if valid_pct > 50:  # At least 50% valid
                            converted = test_conversion
                            parsing_strategy = f'dayfirst=True ({valid_pct:.1f}% valid)'
                    except:
                        pass
                    
                    # Strategy 2: Default pandas parser
                    if converted is None:
                        try:
                            test_conversion = pd.to_datetime(
                                                    self.cleaned_df[col],
                                                    errors='coerce',
                                                    format='mixed'
                                                )
                            valid_pct = test_conversion.notna().sum() / len(test_conversion) * 100
                            if valid_pct > 50:
                                converted = test_conversion
                                parsing_strategy = f'default parser ({valid_pct:.1f}% valid)'
                        except:
                            pass
                    
                    # Strategy 3: Infer format from samples
                    if converted is None:
                        inferred_format = self._infer_datetime_format(col)
                        if inferred_format:
                            try:
                                test_conversion = pd.to_datetime(self.cleaned_df[col], format=inferred_format, errors='coerce')
                                valid_pct = test_conversion.notna().sum() / len(test_conversion) * 100
                                if valid_pct > 50:
                                    converted = test_conversion
                                    parsing_strategy = f'inferred format {inferred_format} ({valid_pct:.1f}% valid)'
                            except:
                                pass
                    
                    # Final fallback
                    if converted is None:
                        converted = pd.to_datetime(
                                            self.cleaned_df[col],
                                            errors='coerce',
                                            format='mixed'
                                                        )
                        valid_pct = converted.notna().sum() / len(converted) * 100
                        parsing_strategy = f'fallback ({valid_pct:.1f}% valid)'
                    
                    # STRICT VALIDATION: Check % of invalid dates
                    total_rows = len(self.cleaned_df)
                    valid_dates = converted.notna().sum()
                    invalid_dates = total_rows - valid_dates
                    invalid_pct = (invalid_dates / total_rows * 100) if total_rows > 0 else 0
                    
                    self.cleaned_df[col] = converted
                    
                    converted_dtype = str(self.cleaned_df[col].dtype)
                    conversion_details[col] = {
                        'from': original_dtype,
                        'to': converted_dtype,
                        'method': 'datetime_conversion',
                        'parsing_strategy': parsing_strategy,
                        'valid_dates': int(valid_dates),
                        'invalid_dates': int(invalid_dates),
                        'invalid_percentage': round(float(invalid_pct), 2)
                    }
                    
                    # CRITICAL WARNING if invalid > 20%
                    if invalid_pct > 20:
                        critical_msg = f"CRITICAL: {col} has {invalid_pct:.1f}% invalid dates ({invalid_dates} rows)"
                        print(f"  [ALERT] {critical_msg}")
                        
                        # Add critical warning
                        self.warnings.append(
                            f"{critical_msg} - Insufficient valid temporal data for time-series analysis"
                        )
                        
                        # Generate insight
                        self.ai_insights.append(
                            f"TEMPORAL DATA QUALITY ISSUE: {col} has {invalid_pct:.1f}% invalid/missing dates "
                            f"({invalid_dates} rows). Time-series forecasting and trend analysis will be unreliable. "
                            f"RECOMMENDATION: Verify date source system or exclude temporal analysis."
                        )
                        
                        # Log decision with low confidence
                        self.decision_log.append({
                            'column': col,
                            'strategy': 'datetime_conversion_failed',
                            'confidence': 0.3,
                            'reason': f'{invalid_pct:.1f}% invalid dates - temporal analysis compromised',
                            'severity': 'critical'
                        })
                    else:
                        print(f"  [OK] {col}: {original_dtype} -> {converted_dtype} (datetime, {parsing_strategy})")
                    
                    # Log successful conversion
                    if invalid_pct <= 20:
                        self.decision_log.append({
                            'column': col,
                            'strategy': 'datetime_conversion',
                            'confidence': 0.9 if invalid_pct < 5 else 0.7,
                            'reason': f'Date column converted using {parsing_strategy}',
                            'valid_percentage': round(100 - invalid_pct, 2)
                        })
                
                elif target_type == 'category':
                    self.cleaned_df[col] = self.cleaned_df[col].astype('category')
                    converted_dtype = str(self.cleaned_df[col].dtype)
                    conversion_details[col] = {
                        'from': original_dtype,
                        'to': converted_dtype,
                        'method': 'category_conversion'
                    }
                    print(f"  [OK] {col}: {original_dtype} -> {converted_dtype} (category)")
                
                elif target_type == 'string':
                    self.cleaned_df[col] = self.cleaned_df[col].astype(str)
                    converted_dtype = str(self.cleaned_df[col].dtype)
                    conversion_details[col] = {
                        'from': original_dtype,
                        'to': converted_dtype,
                        'method': 'string_conversion'
                    }
                    print(f"  [OK] {col}: {original_dtype} -> {converted_dtype} (string)")
                
                self.cleaning_actions.append(f"Converted {col} to {target_type}")
                
                # LOG DECISION FOR TYPE CONVERSION
                if not hasattr(self, 'decision_log'):
                    self.decision_log = []
                
                self.decision_log.append({
                    'column': col,
                    'strategy': f'{target_type}_conversion',
                    'confidence': 0.9,
                    'reason': f'Column converted from {original_dtype} to {converted_dtype}'
                })
                
            except Exception as e:
                print(f"  [ERROR] Failed to convert {col}: {str(e)}")
                conversion_details[col] = {
                    'from': original_dtype,
                    'to': 'failed',
                    'error': str(e)
                }
        
        self.conversion_report = conversion_details
        return self.cleaned_df
    
    def handle_missing_values(self, strategies: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values using specified strategies.
        ENHANCED: Added 'keep_original' strategy for low-missing columns.
        
        Args:
            strategies: Dictionary mapping column names to strategies
                       ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill', 'keep_original')
        
        Returns:
            Updated DataFrame
        """
        print("\n=== HANDLING MISSING VALUES ===")
        handling_details = {}
        
        for col, strategy in strategies.items():
            if col not in self.cleaned_df.columns:
                print(f"   Column '{col}' not found, skipping")
                continue
            
            missing_before = self.cleaned_df[col].isnull().sum()
            
            if missing_before == 0:
                print(f"  - {col}: No missing values")
                continue
            
            try:
                if strategy == 'keep_original':
                    # NEW: Skip cleaning for columns with very low missing
                    print(f"   {col}: Keeping original (low missing rate, no imputation)")
                    handling_details[col] = {
                        'strategy': 'keep_original',
                        'missing_before': int(missing_before),
                        'missing_after': int(missing_before),
                        'note': 'No action taken due to low missing rate'
                    }
                    continue
                
                elif strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                        fill_value = self.cleaned_df[col].mean()
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                        handling_details[col] = {
                            'strategy': 'mean',
                            'filled_with': float(fill_value),
                            'missing_before': int(missing_before),
                            'missing_after': 0
                        }
                        print(f"   {col}: Filled {missing_before} missing with mean={fill_value:.2f}")
                    else:
                        print(f"   {col}: Cannot use mean on non-numeric column")
                        continue
                
                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                        fill_value = self.cleaned_df[col].median()
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                        handling_details[col] = {
                            'strategy': 'median',
                            'filled_with': float(fill_value),
                            'missing_before': int(missing_before),
                            'missing_after': 0
                        }
                        print(f"   {col}: Filled {missing_before} missing with median={fill_value:.2f}")
                    else:
                        print(f"   {col}: Cannot use median on non-numeric column")
                        continue
                
                elif strategy == 'mode':
                    mode_value = self.cleaned_df[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        self.cleaned_df[col] = self.cleaned_df[col].fillna(fill_value)
                        handling_details[col] = {
                            'strategy': 'mode',
                            'filled_with': str(fill_value),
                            'missing_before': int(missing_before),
                            'missing_after': 0
                        }
                        print(f"   {col}: Filled {missing_before} missing with mode='{fill_value}'")
                    else:
                        print(f"   {col}: No mode found")
                        continue
                
                elif strategy == 'drop':
                    rows_before = len(self.cleaned_df)
                    self.cleaned_df = self.cleaned_df.dropna(subset=[col])
                    rows_after = len(self.cleaned_df)
                    dropped = rows_before - rows_after
                    handling_details[col] = {
                        'strategy': 'drop',
                        'rows_dropped': int(dropped),
                        'missing_before': int(missing_before),
                        'missing_after': 0
                    }
                    print(f"   {col}: Dropped {dropped} rows with missing values")
                
                elif strategy == 'forward_fill':
                    self.cleaned_df[col] = self.cleaned_df[col].ffill()
                    missing_after = self.cleaned_df[col].isnull().sum()
                    handling_details[col] = {
                        'strategy': 'forward_fill',
                        'missing_before': int(missing_before),
                        'missing_after': int(missing_after)
                    }
                    print(f"   {col}: Forward filled ({missing_after} still missing)")
                
                elif strategy == 'backward_fill':
                    self.cleaned_df[col] = self.cleaned_df[col].bfill()
                    missing_after = self.cleaned_df[col].isnull().sum()
                    handling_details[col] = {
                        'strategy': 'backward_fill',
                        'missing_before': int(missing_before),
                        'missing_after': int(missing_after)
                    }
                    print(f"   {col}: Backward filled ({missing_after} still missing)")
                
                self.cleaning_actions.append(f"Handled missing in {col} using {strategy}")
                
                # DECISION LOGGING ALREADY HAPPENS IN auto_clean() FOR MISSING VALUES
                # This is just a safety check to ensure decision_log exists
                if not hasattr(self, 'decision_log'):
                    self.decision_log = []
                
            except Exception as e:
                print(f"   Failed to handle missing in {col}: {str(e)}")
        
        return self.cleaned_df
    
    def detect_outliers_iqr(self, numeric_columns: Optional[List[str]] = None, 
                           threshold: float = 1.5) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using IQR method with BUSINESS-CRITICAL column protection.
        
        ANALYST-GRADE LOGIC:
        - Identifies business-critical columns (revenue, sales, profit, quantity, price)
        - Does NOT automatically cap these columns
        - Flags as 'business_outliers' with financial impact assessment
        - Recommends segmentation for business outliers
        
        Args:
            numeric_columns: List of numeric columns to check (auto-detect if None)
            threshold: IQR multiplier (default 1.5, use 3.0 for extreme outliers)
        
        Returns:
            Dictionary with outlier statistics per column including business classification
        """
        # BUSINESS-CRITICAL COLUMN IDENTIFICATION
        business_critical_keywords = ['revenue', 'sales', 'profit', 'quantity', 'price', 'cost', 'amount']
        
        print(f"\n=== DETECTING OUTLIERS (IQR Method, threshold={threshold}) ===")
        
        # Auto-detect numeric columns if not specified
        if numeric_columns is None:
            numeric_columns = self.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_report = {}
        
        for col in numeric_columns:
            if col not in self.cleaned_df.columns:
                continue
            
            Q1 = self.cleaned_df[col].quantile(0.25)
            Q3 = self.cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            
            outliers = self.cleaned_df[
                (self.cleaned_df[col] < lower_bound) | 
                (self.cleaned_df[col] > upper_bound)
            ]
            
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(self.cleaned_df) * 100) if len(self.cleaned_df) > 0 else 0
            
            if outlier_count > 0:
                # Determine if business-critical column
                col_lower = col.lower()
                is_business_critical = any(keyword in col_lower for keyword in business_critical_keywords)
                
                # Calculate financial impact for business columns
                financial_impact = 0.0
                if is_business_critical:
                    # Sum of outlier values (potential high-value transactions)
                    financial_impact = float(outliers[col].sum())
                
                outlier_report[col] = {
                    'Q1': float(Q1),
                    'Q3': float(Q3),
                    'IQR': float(IQR),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': round(float(outlier_pct), 2),
                    'min_value': float(self.cleaned_df[col].min()),
                    'max_value': float(self.cleaned_df[col].max()),
                    'threshold': threshold,
                    'is_business_critical': is_business_critical,
                    'financial_impact': round(financial_impact, 2),
                    'classification': 'business_outliers' if is_business_critical else 'statistical_outliers'
                }
                
                classification_label = "BUSINESS-CRITICAL" if is_business_critical else "STATISTICAL"
                print(f"   [{classification_label}] {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
                print(f"      Range: [{lower_bound:.2f} - {upper_bound:.2f}], Impact: ${financial_impact:,.2f}")
        
        self.outlier_report = outlier_report
        return outlier_report
    
    def remove_outliers_iqr(self, columns: Optional[List[str]] = None, 
                           threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR method.
        
        Args:
            columns: Columns to remove outliers from (all numeric if None)
            threshold: IQR multiplier (default 1.5)
        
        Returns:
            DataFrame with outliers removed
        """
        print(f"\n=== REMOVING OUTLIERS (IQR Method, threshold={threshold}) ===")
        
        if columns is None:
            columns = self.outlier_report.keys()
        
        rows_before = len(self.cleaned_df)
        
        for col in columns:
            if col not in self.outlier_report:
                continue
            
            bounds = self.outlier_report[col]
            lower_bound = bounds['lower_bound']
            upper_bound = bounds['upper_bound']
            
            self.cleaned_df = self.cleaned_df[
                (self.cleaned_df[col] >= lower_bound) & 
                (self.cleaned_df[col] <= upper_bound)
            ]
            
            print(f"   {col}: Removed {bounds['outlier_count']} outliers")
        
        rows_after = len(self.cleaned_df)
        removed = rows_before - rows_after
        
        print(f"\nTotal rows removed: {removed} ({(removed/rows_before*100) if rows_before > 0 else 0:.2f}%)")
        self.cleaning_actions.append(f"Removed {removed} outlier rows using IQR method")
        
        return self.cleaned_df
    
    def generate_cleaning_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive cleaning report.
        ENHANCED: Includes cross-column analysis, dataset complexity, and summary.
        
        Returns:
            Dictionary with complete cleaning summary
        """
        print("\n=== GENERATING CLEANING REPORT ===")
        
        # Generate insights and warnings
        insights = self._generate_insights()
        critical_warnings = self._check_critical_warnings()
        
        # NEW: Cross-column analysis
        cross_column = self._analyze_cross_column_relationships()
        
        # NEW: Dataset complexity
        complexity = self._calculate_dataset_complexity()
        
        # NEW: Final summary
        summary = self._generate_summary()
        
        report = {
            'original_shape': list(self.original_df.shape),
            'cleaned_shape': list(self.cleaned_df.shape),
            'rows_removed': int(self.original_df.shape[0] - self.cleaned_df.shape[0]),
            'columns_analyzed': len(self.original_df.columns),
            'column_types': self.column_types,
            'missing_values_summary': {
                'total_missing_before': int(self.original_df.isnull().sum().sum()),
                'total_missing_after': int(self.cleaned_df.isnull().sum().sum()),
                'columns_affected': len(self.missing_report),
                'details': self.missing_report
            },
            'type_conversions': self.conversion_report,
            'outliers_detected': self.outlier_report,
            'total_outliers': sum(info['outlier_count'] for info in self.outlier_report.values()),
            'cleaning_actions_performed': self.cleaning_actions,
            'data_quality_score': self._calculate_quality_score(),
            'insights': insights,  # Summary insights
            'critical_warnings': critical_warnings,  # Critical warnings
            'cross_column_analysis': cross_column,  # NEW: Correlation analysis
            'dataset_complexity': complexity,  # NEW: Complexity score
            'summary': summary  # NEW: Final summary
        }
        
        print(f"  Original shape: {report['original_shape']}")
        print(f"  Cleaned shape: {report['cleaned_shape']}")
        print(f"  Rows removed: {report['rows_removed']}")
        print(f"  Missing values before: {report['missing_values_summary']['total_missing_before']}")
        print(f"  Missing values after: {report['missing_values_summary']['total_missing_after']}")
        print(f"  Outliers detected: {report['total_outliers']}")
        print(f"  Data quality score: {report['data_quality_score']}/100")
        print(f"  Dataset complexity: {complexity}")
        print(f"  Insights generated: {len(insights)}")
        print(f"  Critical warnings: {len(critical_warnings)}")
        print(f"  Cross-column correlations: {len(cross_column['correlations'])}")
        
        return report
    
    def _calculate_quality_score(self) -> float:
        """
        Calculate overall data quality score (0-100).
        IMPROVED: Uses STRICTER difficulty-based penalties.
        Makes it HARDER to reach 90+
        
        Penalty rules (STRICTER):
        - missing >10%  -10
        - missing >30%  -20 (additional tier)
        - outliers >10%  -15
        - too many fills (>30%)  -20
        - too many transformations (>5)  -15
        
        Returns:
            Quality score (stricter than before)
        """
        score = 100.0
        
        # 1. Check missing percentage penalty (STRICTER)
        total_cells = self.cleaned_df.shape[0] * self.cleaned_df.shape[1]
        if total_cells > 0:
            missing_pct = (self.cleaned_df.isnull().sum().sum() / total_cells) * 100
            if missing_pct > 30:
                score -= 20  # Increased from 10
                print(f"  [WARN] Very high missing rate ({missing_pct:.1f}%): -20 penalty")
            elif missing_pct > 10:
                score -= 10  # New tier (was >20%)
                print(f"  [WARN] High missing rate ({missing_pct:.1f}%): -10 penalty")
        
        # 2. Check outlier percentage penalty (STRICTER)
        if self.outlier_report:
            total_outliers = sum(info['outlier_count'] for info in self.outlier_report.values())
            if total_outliers > 0:
                outlier_pct = (total_outliers / (self.cleaned_df.shape[0] * len(self.outlier_report))) * 100
                if outlier_pct > 10:
                    score -= 15  # Increased from 10
                    print(f"  [WARN] High outlier rate ({outlier_pct:.1f}%): -15 penalty")
        
        # 3. Check fill ratio penalty (STRICTER)
        if hasattr(self, 'fill_operations') and self.fill_operations:
            filled_count = sum(
                self.missing_report.get(col, {}).get('count', 0) 
                for col in self.fill_operations.keys()
            )
            original_total = self.original_df.shape[0] * len(self.fill_operations)
            if original_total > 0:
                fill_ratio = (filled_count / original_total) * 100
                if fill_ratio > 30:
                    score -= 20  # Increased from 15, threshold lowered from 50%
                    print(f"  [WARN] Very high fill ratio ({fill_ratio:.1f}%): -20 penalty")
        
        # 4. Check transformation count penalty (STRICTER)
        if len(self.cleaning_actions) > 5:
            score -= 15  # Increased from 10
            print(f"  [WARN] Many transformations ({len(self.cleaning_actions)}): -15 penalty")
        
        # 5. Penalize for dropped rows (PROPORTIONAL)
        rows_dropped = self.original_df.shape[0] - self.cleaned_df.shape[0]
        if rows_dropped > 0:
            drop_pct = (rows_dropped / self.original_df.shape[0]) * 100
            if drop_pct > 20:
                score -= 20
                print(f"  [WARN] Dropped {rows_dropped} rows ({drop_pct:.1f}%): -20 penalty")
            elif drop_pct > 10:
                score -= 15
                print(f"  [WARN] Dropped {rows_dropped} rows ({drop_pct:.1f}%): -15 penalty")
            elif drop_pct > 5:
                score -= 10
                print(f"  [WARN] Dropped {rows_dropped} rows ({drop_pct:.1f}%): -10 penalty")
            else:
                score -= 5
                print(f"  [WARN] Dropped {rows_dropped} rows ({drop_pct:.1f}%): -5 penalty")
        
        # Ensure score is between 0-100
        final_score = max(0, min(100, round(score, 2)))
        
        # Add interpretation guide
        if final_score >= 90:
            print(f"  [OK] Quality score {final_score}/100: EXCELLENT (very clean data)")
        elif final_score >= 75:
            print(f"  [OK] Quality score {final_score}/100: GOOD (minor issues)")
        elif final_score >= 60:
            print(f"  [WARN] Quality score {final_score}/100: FAIR (moderate cleaning needed)")
        else:
            print(f"  [WARN] Quality score {final_score}/100: POOR (extensive cleaning required)")
        
        return final_score
    
    def winsorize_outliers(self, columns: Optional[List[str]] = None, 
                          threshold: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using Winsorization with BUSINESS-CRITICAL COLUMN PROTECTION.
        
        ANALYST-GRADE LOGIC:
        - NEVER caps business-critical columns (revenue, sales, profit, quantity, price)
        - Only applies capping to non-business columns
        - Logs justification for each capping decision
        - Generates insights about business outliers for segmentation analysis
        
        Args:
            columns: Columns to cap (all numeric if None)
            threshold: IQR multiplier (default 1.5)
        
        Returns:
            DataFrame with outliers capped (non-business columns only)
        """
        # Business-critical column protection
        business_critical_keywords = ['revenue', 'sales', 'profit', 'quantity', 'price', 'cost', 'amount']
        
        print(f"\n=== WINSORIZING OUTLIERS (Business-Critical Protection Enabled) ===")
        
        if columns is None:
            columns = self.outlier_report.keys()
        
        capped_count = 0
        business_columns_skipped = []
        
        for col in columns:
            if col not in self.outlier_report:
                continue
            
            bounds = self.outlier_report[col]
            
            # ANALYST-GRADE CHECK: Skip business-critical columns
            col_lower = col.lower()
            is_business_critical = any(keyword in col_lower for keyword in business_critical_keywords)
            
            if is_business_critical:
                business_columns_skipped.append(col)
                outlier_count = bounds['outlier_count']
                financial_impact = bounds.get('financial_impact', 0)
                
                # Log decision to NOT cap business columns
                self.decision_log.append({
                    'column': col,
                    'strategy': 'business_outlier_preservation',
                    'confidence': 0.95,
                    'reason': f'{outlier_count} business outliers (${financial_impact:,.2f}) preserved for segmentation analysis - likely bulk orders or high-value customers',
                    'justification': 'Business-critical revenue/quantity data should not be artificially capped'
                })
                
                # Generate insight about business outliers
                self.ai_insights.append(
                    f"BUSINESS OUTLIERS DETECTED: {col} has {outlier_count} extreme values "
                    f"(${financial_impact:,.2f} total impact). These likely represent bulk orders, "
                    f"high-value customers, or wholesale transactions. "
                    f"RECOMMENDATION: Segment analysis by customer tier or product category instead of capping."
                )
                
                print(f"   [SKIP] {col}: {outlier_count} business outliers preserved (${financial_impact:,.2f})")
                print(f"          → Recommended: Customer/product segmentation analysis")
                continue
            
            # Only cap non-business columns
            lower_bound = bounds['lower_bound']
            upper_bound = bounds['upper_bound']
            
            # Count values being capped
            below_lower = (self.cleaned_df[col] < lower_bound).sum()
            above_upper = (self.cleaned_df[col] > upper_bound).sum()
            
            # Cap values
            original_values = self.cleaned_df[col].copy()
            self.cleaned_df[col] = self.cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            capped = below_lower + above_upper
            capped_count += int(capped)
            
            print(f"   [CAP] {col}: Capped {capped} statistical outliers [{lower_bound:.2f} - {upper_bound:.2f}]")
            
            # ANALYST-GRADE DECISION LOGGING with justification
            self.decision_log.append({
                'column': col,
                'strategy': 'iqr_capping',
                'confidence': 0.85,
                'reason': f'{capped} statistical outliers capped using IQR method',
                'justification': f'Non-business column - safe to cap for statistical normalization'
            })
        
        print(f"\nTotal values capped: {capped_count}")
        if business_columns_skipped:
            print(f"Business columns skipped: {', '.join(business_columns_skipped)}")
            print(f"  → Business outliers preserved for segmentation analysis")
        
        self.cleaning_actions.append(f"Winsorized {capped_count} outlier values (business-critical columns protected)")
        
        return self.cleaned_df
    
    def _is_timeseries_column(self, col: str) -> bool:
        """
        Check if a column is likely a time series.
        
        Args:
            col: Column name
        
        Returns:
            True if column appears to be time-based
        """
        # Check if column name contains time-related keywords
        time_keywords = ['date', 'time', 'timestamp', 'dt', 'day', 'month', 'year', 'hour']
        col_lower = col.lower()
        
        # Check column name or if it's datetime type
        if any(keyword in col_lower for keyword in time_keywords):
            return True
        
        if pd.api.types.is_datetime64_any_dtype(self.cleaned_df[col]):
            return True
        
        return False
    
    def _is_skewed(self, col: str, skew_threshold: float = 1.0) -> bool:
        """
        Check if a numeric column is skewed.
        
        Args:
            col: Column name
            skew_threshold: Absolute skewness value to consider skewed (default 1.0)
        
        Returns:
            True if column is skewed
        """
        try:
            skewness = self.cleaned_df[col].skew()
            is_skewed = abs(skewness) > skew_threshold
            return is_skewed
        except:
            return False
    
    def auto_select_strategy(self, col: str) -> str:
        """
        Automatically select the best missing value strategy for a column.
        
        Rules:
        - Time series  forward_fill
        - Numeric skewed  median
        - Numeric normal  mean
        - Categorical  mode
        
        Args:
            col: Column name
        
        Returns:
            Strategy string
        """
        col_type = self.column_types.get(col, 'unknown')
        
        # Rule 1: Time series
        if self._is_timeseries_column(col):
            return 'forward_fill'
        
        # Rule 2: Numeric columns
        if col_type in ['numeric', 'numeric_convertible']:
            # Check skewness
            if self._is_skewed(col):
                return 'median'  # Skewed  median
            else:
                return 'mean'    # Normal  mean
        
        # Rule 3: Categorical
        if col_type in ['categorical', 'text']:
            return 'mode'
        
        # Default fallback
        return 'median'
    
    def _analyze_column_characteristics(self, col: str) -> Dict[str, float]:
        """
        Calculate comprehensive statistics for decision engine.
        
        Args:
            col: Column name
        
        Returns:
            Dictionary with skewness, kurtosis, outlier %, missing %
        """
        result = {
            'skewness': 0.0,
            'kurtosis': 0.0,
            'outlier_percentage': 0.0,
            'missing_percentage': 0.0
        }
        
        try:
            # Missing percentage
            total_rows = len(self.cleaned_df)
            missing_count = self.cleaned_df[col].isnull().sum()
            result['missing_percentage'] = (missing_count / total_rows * 100) if total_rows > 0 else 0
            
            # For numeric columns only
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                col_data = self.cleaned_df[col].dropna()
                
                if len(col_data) > 0:
                    # Skewness
                    result['skewness'] = float(col_data.skew())
                    
                    # Kurtosis
                    result['kurtosis'] = float(col_data.kurtosis())
                    
                    # Outlier percentage using IQR
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                        result['outlier_percentage'] = (outliers / len(col_data) * 100)
        except:
            pass
        
        return result
    
    def _is_id_column(self, col: str) -> bool:
        """
        Check if column is ID-like and should be skipped.
        
        Args:
            col: Column name
        
        Returns:
            True if column appears to be an identifier
        """
        id_keywords = ['id', 'transaction', 'user_id', 'customer_id', 'order_id', 
                      'product_id', 'store_id', 'index', 'key', 'code']
        col_lower = col.lower()
        
        return any(keyword in col_lower for keyword in id_keywords)
    
    def _classify_column_type(self, col: str) -> str:
        """
        Classify column into priority categories for context-aware cleaning.
        
        Categories:
        - identifier: ID-like columns (id, transaction_id, user_id)
        - target/important: Business-critical columns (revenue, sales, price, income)
        - categorical: Category columns
        - auxiliary: Supporting columns (notes, feedback, descriptions)
        
        Args:
            col: Column name
        
        Returns:
            Classification string
        """
        col_lower = col.lower()
        
        # Identifier columns - skip completely
        id_keywords = ['id', 'transaction', 'user_id', 'customer_id', 'order_id', 
                      'product_id', 'store_id', 'index', 'key', 'code']
        if any(keyword in col_lower for keyword in id_keywords):
            return 'identifier'
        
        # Target/important columns - use safest strategy
        target_keywords = ['revenue', 'sales', 'price', 'income', 'profit', 'cost', 
                          'amount', 'salary', 'budget', 'earnings', 'payment']
        if any(keyword in col_lower for keyword in target_keywords):
            return 'target'
        
        # Auxiliary columns - check BEFORE categorical (allow aggressive cleaning)
        auxiliary_keywords = ['note', 'feedback', 'comment', 'description', 'remark', 
                             'observation', 'text', 'review', 'opinion']
        if any(keyword in col_lower for keyword in auxiliary_keywords):
            return 'auxiliary'
        
        # Categorical columns (after auxiliary check)
        if self.column_types.get(col) in ['categorical', 'text']:
            return 'categorical'
        
        # Default: treat as regular numeric/text based on type
        if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
            return 'numeric'
        
        return 'categorical'
    
    def _calculate_confidence_score(self, col: str, characteristics: Dict[str, float]) -> float:
        """
        Calculate confidence score for decision based on statistical strength.
        IMPROVED: Reduces overconfidence with realistic range 0.5-0.9
        
        Confidence is based on:
        - Strength of skewness (higher absolute = more confident)
        - Outlier percentage (lower = more confident)
        - Missing percentage (lower = more confident)
        - Sample size (larger = more confident)
        
        Args:
            col: Column name
            characteristics: Dictionary with skewness, kurtosis, outlier %, missing %
        
        Returns:
            Confidence score between 0.5 and 0.9 (realistic range)
        """
        # Start with moderate confidence (not overconfident)
        confidence = 0.75
        
        skewness = abs(characteristics.get('skewness', 0))
        outlier_pct = characteristics.get('outlier_percentage', 0)
        missing_pct = characteristics.get('missing_percentage', 0)
        sample_size = len(self.cleaned_df[col].dropna())
        total_size = len(self.cleaned_df[col])
        
        # Penalize for high missing percentage (SIGNIFICANT IMPACT)
        if missing_pct > 40:
            confidence -= 0.25  # Severe penalty
        elif missing_pct > 30:
            confidence -= 0.20
        elif missing_pct > 20:
            confidence -= 0.15  # Significant penalty
        elif missing_pct > 10:
            confidence -= 0.10
        elif missing_pct > 5:
            confidence -= 0.05
        
        # Penalize for high outlier percentage (MODERATE IMPACT)
        if outlier_pct > 20:
            confidence -= 0.15
        elif outlier_pct > 15:
            confidence -= 0.12
        elif outlier_pct > 10:
            confidence -= 0.10
        elif outlier_pct > 5:
            confidence -= 0.05
        
        # Penalize for high skewness uncertainty
        if skewness >= 2.0:
            confidence -= 0.10  # Very skewed = uncertain
        elif skewness >= 1.5:
            confidence -= 0.08
        elif skewness >= 1.0:
            confidence -= 0.05
        elif 0.5 <= skewness < 1.0:
            confidence -= 0.03  # Moderate skewness
        
        # Penalize for low sample size
        valid_ratio = sample_size / total_size if total_size > 0 else 0
        if valid_ratio < 0.6:
            confidence -= 0.15  # Less than 60% data available
        elif valid_ratio < 0.7:
            confidence -= 0.10
        elif valid_ratio < 0.8:
            confidence -= 0.05
        elif valid_ratio < 0.9:
            confidence -= 0.02
        
        # Ensure confidence is in realistic range 0.5-0.9
        return round(max(0.5, min(0.9, confidence)), 2)
    
    def _generate_insights(self) -> List[str]:
        """
        Generate summary insights about dataset quality issues.
        ENHANCED: Includes actionable recommendations.
        
        Returns:
            List of insight strings with recommendations
        """
        insights = []
        
        # Check for high skewness across columns
        highly_skewed_cols = []
        for col in self.cleaned_df.select_dtypes(include=[np.number]).columns:
            try:
                skewness = self.cleaned_df[col].dropna().skew()
                if abs(skewness) > 1:
                    highly_skewed_cols.append((col, skewness))
            except:
                pass
        
        if highly_skewed_cols:
            cols_str = ', '.join([f"{col} (skew={skew:.2f})" for col, skew in highly_skewed_cols[:3]])
            if len(highly_skewed_cols) > 3:
                cols_str += f" and {len(highly_skewed_cols) - 3} more"
            insights.append(f"Dataset has high skewness in {len(highly_skewed_cols)} columns: {cols_str}")
        
        # Check for significant missing data WITH RECOMMENDATIONS
        if self.missing_report:
            high_missing = [(col, info['percentage']) for col, info in self.missing_report.items() 
                          if info['percentage'] > 10]
            if high_missing:
                for col, pct in high_missing:
                    # Add actionable recommendation based on severity
                    if pct > 40:
                        insights.append(f"{col} has {pct:.0f}% missing -> consider dropping column (too much data missing)")
                    elif pct > 30:
                        insights.append(f"{col} has {pct:.0f}% missing -> recommend median imputation (robust to outliers)")
                    elif pct > 20:
                        insights.append(f"{col} has {pct:.0f}% missing -> recommend median imputation")
                    else:
                        insights.append(f"{col} has {pct:.0f}% missing -> recommend mean/median imputation")
        
        # Check for outliers
        if self.outlier_report:
            high_outlier_cols = [(col, info['outlier_percentage']) for col, info in self.outlier_report.items()
                               if info['outlier_percentage'] > 5]
            if high_outlier_cols:
                for col, pct in high_outlier_cols:
                    if pct > 15:
                        insights.append(f"Outliers heavily present in {col} ({pct:.1f}%) -> recommend winsorization (capping)")
                    else:
                        insights.append(f"Outliers detected in {col} ({pct:.1f}%) -> consider capping or transformation")
        
        # Check for high fill ratio
        if hasattr(self, 'fill_operations') and self.fill_operations:
            filled_count = sum(
                self.missing_report.get(col, {}).get('count', 0) 
                for col in self.fill_operations.keys()
            )
            original_total = self.original_df.shape[0] * len(self.fill_operations)
            if original_total > 0:
                fill_ratio = (filled_count / original_total) * 100
                if fill_ratio > 30:
                    insights.append(f"High imputation rate ({fill_ratio:.1f}%) -> consider improving data collection processes")
        
        # General quality assessment
        rows_dropped = self.original_df.shape[0] - self.cleaned_df.shape[0]
        if rows_dropped > 0:
            drop_pct = (rows_dropped / self.original_df.shape[0]) * 100
            if drop_pct > 10:
                insights.append(f"Significant data loss from cleaning ({drop_pct:.1f}% rows dropped) -> review data pipeline for systematic issues")
        
        if not insights:
            insights.append("Dataset quality is generally good - no major issues detected")
        
        return insights
    
    def _check_critical_warnings(self) -> List[str]:
        """
        Check for critical data quality issues that require attention.
        
        Returns:
            List of warning strings with [CRITICAL] prefix
        """
        warnings = []
        
        # Check for extreme missing values (>40%)
        if self.missing_report:
            for col, info in self.missing_report.items():
                if info['percentage'] > 40:
                    warnings.append(f"[CRITICAL] {col}: {info['percentage']:.1f}% missing data - column may be unreliable")
        
        # Check for extreme outlier percentages (>20%)
        if self.outlier_report:
            for col, info in self.outlier_report.items():
                if info['outlier_percentage'] > 20:
                    warnings.append(f"[CRITICAL] {col}: {info['outlier_percentage']:.1f}% outliers - investigate data source")
        
        # Check for too many transformations
        if len(self.cleaning_actions) > 10:
            warnings.append(f"[CRITICAL] Excessive transformations ({len(self.cleaning_actions)}) - review data pipeline")
        
        # Check for massive data loss
        rows_dropped = self.original_df.shape[0] - self.cleaned_df.shape[0]
        if rows_dropped > 0:
            drop_pct = (rows_dropped / self.original_df.shape[0]) * 100
            if drop_pct > 30:
                warnings.append(f"[CRITICAL] Massive data loss ({drop_pct:.1f}% rows dropped) - results may be biased")
        
        return warnings
    
    def _analyze_cross_column_relationships(self) -> Dict[str, Any]:
        """
        Analyze relationships between columns.
        NEW: Detects correlations, redundancies, and strong relationships.
        
        Returns:
            Dictionary with correlation analysis
        """
        result = {
            'correlations': [],
            'redundant_columns': [],
            'strong_relationships': []
        }
        
        try:
            # Get numeric columns
            numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return result
            
            # Calculate correlation matrix
            corr_matrix = self.cleaned_df[numeric_cols].corr()
            
            # Find high correlations (excluding self-correlation)
            high_correlations = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr_value = corr_matrix.loc[col1, col2]
                    
                    if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': round(float(corr_value), 2),
                            'strength': 'very strong' if abs(corr_value) > 0.9 else 'strong'
                        })
            
            result['correlations'] = high_correlations
            
            # Identify potentially redundant columns (correlation > 0.9)
            redundant = []
            for corr in high_correlations:
                if abs(corr['correlation']) > 0.9:
                    redundant.append({
                        'column1': corr['column1'],
                        'column2': corr['column2'],
                        'correlation': corr['correlation'],
                        'recommendation': f"Consider removing one of these highly correlated columns"
                    })
            
            result['redundant_columns'] = redundant
            
            # Identify strong relationships for insights
            strong_rels = [c for c in high_correlations if abs(c['correlation']) > 0.8]
            result['strong_relationships'] = strong_rels
            
        except Exception as e:
            pass  # Return empty result if correlation analysis fails
        
        return result
    
    def _calculate_dataset_complexity(self) -> str:
        """
        Calculate overall dataset difficulty/complexity score.
        NEW: Based on missing %, skewness, and outliers.
        
        Returns:
            Complexity level: 'LOW', 'MEDIUM', or 'HIGH'
        """
        try:
            # Factor 1: Missing percentage
            total_cells = self.cleaned_df.shape[0] * self.cleaned_df.shape[1]
            missing_pct = 0
            if total_cells > 0:
                missing_pct = (self.cleaned_df.isnull().sum().sum() / total_cells) * 100
            
            # Factor 2: Average skewness across numeric columns
            numeric_cols = self.cleaned_df.select_dtypes(include=[np.number]).columns
            avg_skewness = 0
            if len(numeric_cols) > 0:
                skewness_values = [
                    abs(self.cleaned_df[col].dropna().skew()) 
                    for col in numeric_cols 
                    if len(self.cleaned_df[col].dropna()) > 0
                ]
                if skewness_values:
                    avg_skewness = np.mean(skewness_values)
            
            # Factor 3: Outlier percentage
            total_outliers = sum(info['outlier_count'] for info in self.outlier_report.values())
            total_cells_checked = self.cleaned_df.shape[0] * len(self.outlier_report) if self.outlier_report else 1
            outlier_pct = (total_outliers / total_cells_checked * 100) if total_cells_checked > 0 else 0
            
            # Calculate complexity score (0-100)
            complexity_score = 0
            
            # Missing contribution (0-40 points)
            if missing_pct > 30:
                complexity_score += 40
            elif missing_pct > 20:
                complexity_score += 30
            elif missing_pct > 10:
                complexity_score += 20
            elif missing_pct > 5:
                complexity_score += 10
            
            # Skewness contribution (0-30 points)
            if avg_skewness > 2:
                complexity_score += 30
            elif avg_skewness > 1:
                complexity_score += 20
            elif avg_skewness > 0.5:
                complexity_score += 10
            
            # Outlier contribution (0-30 points)
            if outlier_pct > 15:
                complexity_score += 30
            elif outlier_pct > 10:
                complexity_score += 20
            elif outlier_pct > 5:
                complexity_score += 10
            
            # Determine complexity level
            if complexity_score >= 60:
                return 'HIGH'
            elif complexity_score >= 30:
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except:
            return 'MEDIUM'  # Default to medium if calculation fails
    
    def _generate_summary(self) -> str:
        """
        Generate final comprehensive summary of data quality and cleaning decisions.
        NEW: Provides overall assessment and recommendations.
        
        Returns:
            Summary string
        """
        try:
            # Gather key metrics
            quality_score = self._calculate_quality_score()
            complexity = self._calculate_dataset_complexity()
            total_missing = self.cleaned_df.isnull().sum().sum()
            original_missing = self.original_df.isnull().sum().sum()
            rows_dropped = self.original_df.shape[0] - self.cleaned_df.shape[0]
            
            # Assess confidence levels
            if hasattr(self, 'decision_log') and self.decision_log:
                avg_confidence = np.mean([d.get('confidence', 0.5) for d in self.decision_log])
            else:
                avg_confidence = 0.7
            
            # Build summary based on findings
            summary_parts = []
            
            # Overall quality assessment
            if quality_score >= 80:
                summary_parts.append("This dataset has good overall quality")
            elif quality_score >= 60:
                summary_parts.append("This dataset has moderate quality issues")
            else:
                summary_parts.append("This dataset has significant quality challenges")
            
            # Missing data assessment
            if original_missing > 0:
                if total_missing == 0:
                    summary_parts.append("with all missing values successfully handled")
                else:
                    summary_parts.append(f"with {total_missing} remaining missing values after cleaning")
            
            # Complexity assessment
            if complexity == 'HIGH':
                summary_parts.append("The dataset presents high complexity due to distribution characteristics")
            elif complexity == 'MEDIUM':
                summary_parts.append("with moderate data complexity")
            
            # Confidence assessment
            if avg_confidence >= 0.8:
                summary_parts.append("Cleaning decisions were made with high confidence")
            elif avg_confidence >= 0.6:
                summary_parts.append("Cleaning decisions have moderate confidence levels")
            else:
                summary_parts.append("Some cleaning decisions have lower confidence due to data characteristics")
            
            # Row drop assessment
            if rows_dropped > 0:
                drop_pct = (rows_dropped / self.original_df.shape[0]) * 100
                if drop_pct > 10:
                    summary_parts.append(f"Significant data loss occurred ({drop_pct:.1f}% rows dropped)")
            
            # Recommendation
            if quality_score < 70 or complexity == 'HIGH':
                summary_parts.append("Further data collection is recommended to improve quality")
            elif quality_score < 85:
                summary_parts.append("Data is suitable for analysis with noted transformations")
            else:
                summary_parts.append("Data is ready for downstream analysis")
            
            return '. '.join(summary_parts) + '.'
            
        except:
            return "Data cleaning completed successfully. Review detailed report for specific findings."
    
    def _decide_strategy_with_reasoning(self, col: str) -> Dict[str, str]:
        """
        Decision-driven strategy selection with detailed reasoning.
        ENHANCED: Dynamic strategy for target columns based on distribution.
        
        Decision tree (context-aware):
        1. ID columns -> skip
        2. Time series -> forward_fill
        3. Target columns -> dynamic (skewed->median, normal->mean, heavy_outliers->median, low_missing->keep)
        4. Missing > 30% -> median (robust)
        5. Outliers > 10% -> median (robust)
        6. Skewness > 1 or < -1 -> median (handles skew)
        7. Abs(skewness) < 0.5 -> mean (normal distribution)
        8. Else -> mean (default)
        
        Args:
            col: Column name
        
        Returns:
            Dictionary with strategy, reason, and confidence
        """
        col_type = self.column_types.get(col, 'unknown')
        col_priority = self._classify_column_type(col)
        characteristics = self._analyze_column_characteristics(col)
        
        missing_pct = characteristics['missing_percentage']
        outlier_pct = characteristics['outlier_percentage']
        skewness = characteristics['skewness']
        
        # Rule 0: ID columns - skip cleaning
        if col_priority == 'identifier':
            return {
                'strategy': 'skip',
                'reason': 'ID-like column (skipped)',
                'confidence': 0.9  # High confidence in skipping
            }
        
        # Rule 1: Time series
        if self._is_timeseries_column(col):
            return {
                'strategy': 'forward_fill',
                'reason': 'time series column',
                'confidence': 0.85
            }
        
        # Rule 2: Target/important columns - DYNAMIC strategy based on distribution
        if col_priority == 'target':
            # Analyze distribution and choose best strategy
            if missing_pct < 5:
                return {
                    'strategy': 'keep_original',
                    'reason': f'target column with low missing ({missing_pct:.1f}%, keeping original)',
                    'confidence': 0.8
                }
            elif outlier_pct > 15:
                return {
                    'strategy': 'median',
                    'reason': f'target column with heavy outliers ({outlier_pct:.1f}%)',
                    'confidence': self._calculate_confidence_score(col, characteristics)
                }
            elif abs(skewness) > 1:
                return {
                    'strategy': 'median',
                    'reason': f'target column, highly skewed (skewness={skewness:.2f})',
                    'confidence': self._calculate_confidence_score(col, characteristics)
                }
            else:
                return {
                    'strategy': 'mean',
                    'reason': f'target column, near-normal distribution (skewness={skewness:.2f})',
                    'confidence': self._calculate_confidence_score(col, characteristics)
                }
        
        # Rule 3: Categorical/text
        if col_priority == 'categorical' or col_type in ['categorical', 'text']:
            return {
                'strategy': 'mode',
                'reason': 'categorical column',
                'confidence': self._calculate_confidence_score(col, characteristics)
            }
        
        # Rule 4+: Numeric columns - use decision engine
        if col_type in ['numeric', 'numeric_convertible'] or col_priority == 'numeric':
            # Calculate confidence
            confidence = self._calculate_confidence_score(col, characteristics)
            
            # Decision logic with priority
            if missing_pct > 30:
                return {
                    'strategy': 'median',
                    'reason': f'missing={missing_pct:.1f}% (>30%, robust choice)',
                    'confidence': confidence
                }
            elif outlier_pct > 10:
                return {
                    'strategy': 'median',
                    'reason': f'outliers={outlier_pct:.1f}% (>10%, robust choice)',
                    'confidence': confidence
                }
            elif abs(skewness) > 1:
                return {
                    'strategy': 'median',
                    'reason': f'skewness={skewness:.2f} (highly skewed)',
                    'confidence': confidence
                }
            elif abs(skewness) < 0.5:
                return {
                    'strategy': 'mean',
                    'reason': f'skewness={skewness:.2f} (near-normal distribution)',
                    'confidence': confidence
                }
            else:
                # Moderate skewness (0.5-1.0)
                return {
                    'strategy': 'mean',
                    'reason': f'skewness={skewness:.2f} (moderate, mean acceptable)',
                    'confidence': confidence
                }
        
        # Default fallback
        return {
            'strategy': 'median',
            'reason': 'default fallback strategy',
            'confidence': 0.6  # Lower confidence for default
        }
    
    def _infer_datetime_format(self, col: str, sample_size: int = 50) -> Optional[str]:
        """
        Improved datetime format detection with larger sample and success rate tracking.
        
        Args:
            col: Column name
            sample_size: Number of values to sample (default 50)
        
        Returns:
            Best format string or None
        """
        # Get non-null sample
        col_data = self.cleaned_df[col].dropna()
        sample = col_data.head(sample_size)
        
        if len(sample) == 0:
            return None
        
        # Formats to try (most common first)
        formats_to_try = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%d %b %Y',
            '%d %B %Y',
            '%b %d, %Y',
            '%B %d, %Y'
        ]
        
        best_format = None
        best_success_rate = 0.0
        
        for fmt in formats_to_try:
            success_count = 0
            total_count = 0
            
            for date_str in sample:
                try:
                    pd.to_datetime(str(date_str), format=fmt)
                    success_count += 1
                except:
                    pass
                total_count += 1
            
            success_rate = success_count / total_count if total_count > 0 else 0
            
            # Update best if this format has higher success rate
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_format = fmt
            
            # If we found a format with >90% success, use it
            if success_rate > 0.9:
                return fmt
        
        # Return best format if at least 80% success rate
        return best_format if best_success_rate >= 0.8 else None
    
    def _generate_business_insights(self):
        """
        Generate high-level business insights, warnings, and AI summary.
        Thinks like an analyst - explains WHAT + WHY + business impact.
        """
        print("\n=== GENERATING BUSINESS INSIGHTS ===")
        
        # 1. CRITICAL ISSUES ANALYSIS
        critical_issues = []
        warning_issues = []
        problem_columns = set()
        
        # Outlier severity analysis with business context
        if self.outlier_report:
            for col, info in self.outlier_report.items():
                outlier_pct = (info['outlier_count'] / len(self.cleaned_df) * 100)
                
                if outlier_pct > 10:
                    critical_issues.append({
                        'type': 'outliers',
                        'column': col,
                        'severity': 'critical',
                        'message': f"{outlier_pct:.1f}% extreme outliers in {col} ({info['outlier_count']} values)",
                        'business_impact': 'Statistical models will be biased - consider segmentation or transformation'
                    })
                    problem_columns.add(col)
                elif outlier_pct > 5:
                    warning_issues.append({
                        'type': 'outliers',
                        'column': col,
                        'severity': 'warning',
                        'message': f"Moderate outlier presence in {col} ({outlier_pct:.1f}%)",
                        'business_impact': 'May skew averages and totals - verify data collection process'
                    })
                    problem_columns.add(col)
        
        # Missing value patterns analysis
        if self.missing_report:
            critical_missing = []
            moderate_missing = []
            
            for col, info in self.missing_report.items():
                if info['percentage'] > 30:
                    critical_missing.append(col)
                    critical_issues.append({
                        'type': 'missing_data',
                        'column': col,
                        'severity': 'critical',
                        'message': f"Column '{col}' has {info['percentage']:.1f}% missing values ({info['count']} rows)",
                        'business_impact': 'Data source validation needed - may indicate system integration issues'
                    })
                    problem_columns.add(col)
                elif info['percentage'] > 10:
                    moderate_missing.append(col)
                    warning_issues.append({
                        'type': 'missing_data',
                        'column': col,
                        'severity': 'warning',
                        'message': f"{info['percentage']:.1f}% missing data in {col}",
                        'business_impact': 'Consider improving data collection processes'
                    })
                    problem_columns.add(col)
            
            if critical_missing:
                self.ai_insights.append(
                    f"CRITICAL: {len(critical_missing)} columns have >30% missing data ({', '.join(critical_missing[:3])}). "
                    f"This suggests systemic data collection issues that require immediate investigation."
                )
        
        # Negative values business impact
        if self.negative_values_found:
            total_negative_cols = len(self.negative_values_found)
            high_impact_negatives = [col for col, info in self.negative_values_found.items() 
                                    if info['percentage'] > 2]
            
            if high_impact_negatives:
                self.ai_insights.append(
                    f"BUSINESS ALERT: {total_negative_cols} columns contain negative values. "
                    f"High-impact columns: {', '.join(high_impact_negatives[:3])}. "
                    f"This likely represents refunds, returns, or data entry errors - verify against business operations."
                )
            else:
                self.ai_insights.append(
                    f"Negative values detected in {total_negative_cols} columns - normal for refund/return scenarios but validate against expected business patterns."
                )
        
        # Skewness impact on analysis
        if self.skewness_report:
            extreme_skew = [col for col, info in self.skewness_report.items() if info['severity'] == 'extreme']
            moderate_skew = [col for col, info in self.skewness_report.items() if info['severity'] == 'moderate']
            
            if extreme_skew:
                self.ai_insights.append(
                    f"STATISTICAL WARNING: {len(extreme_skew)} columns have extreme skewness ({', '.join(extreme_skew[:3])}). "
                    f"Standard statistical methods assume normal distribution - your analysis may be invalid without transformation."
                )
            
            if moderate_skew:
                warning_issues.append({
                    'type': 'skewness',
                    'columns': moderate_skew,
                    'severity': 'warning',
                    'message': f"{len(moderate_skew)} columns show moderate skewness",
                    'business_impact': 'Consider log transformation for more reliable statistics'
                })
        
        # Duplicate data quality
        if self.duplicates_found > 0:
            dup_pct = (self.duplicates_found / len(self.original_df) * 100)
            
            if dup_pct > 5:
                critical_issues.append({
                    'type': 'duplicates',
                    'severity': 'critical',
                    'message': f"{self.duplicates_found} duplicate records removed ({dup_pct:.1f}% of dataset)",
                    'business_impact': 'Data deduplication essential - duplicates inflate metrics and mislead analysis'
                })
                self.ai_insights.append(
                    f"DATA QUALITY: Removed {self.duplicates_found} duplicates ({dup_pct:.1f}%). "
                    f"This level of duplication suggests possible ETL issues or multiple data entry points needing review."
                )
            else:
                self.ai_insights.append(
                    f"Minor duplication detected ({self.duplicates_found} records, {dup_pct:.1f}%) - routine data cleaning applied."
                )
        
        # Datatype consistency
        if self.datatype_issues:
            date_issues = [issue for issue in self.datatype_issues if 'datetime' in issue['action'].lower()]
            if date_issues:
                warning_issues.append({
                    'type': 'datatype_inconsistency',
                    'columns': [issue['column'] for issue in date_issues],
                    'severity': 'warning',
                    'message': f"Inconsistent date formats in {len(date_issues)} columns",
                    'business_impact': 'Date parsing errors may affect time-series analysis'
                })
        
        # Add all critical and warning issues to appropriate lists
        for issue in critical_issues:
            self.warnings.insert(0, f"🔴 CRITICAL: {issue['message']} - {issue['business_impact']}")
        
        for issue in warning_issues:
            self.warnings.append(f"⚠️ WARNING: {issue['message']}")
        
        # 2. COMPREHENSIVE AI SUMMARY
        total_decisions = len(self.decision_log)
        columns_affected = len(set(d['column'] for d in self.decision_log)) if self.decision_log else 0
        total_issues = len(critical_issues) + len(warning_issues)
        
        # Enhanced risk level calculation (analyst-level intelligence)
        risk_score = 0
        
        # Critical issues weight heavily
        risk_score += len([i for i in critical_issues if i['severity'] == 'critical']) * 4
        
        # Warning issues moderate impact
        risk_score += len([i for i in warning_issues if i['severity'] == 'warning']) * 2
        
        # Missing data severity
        risk_score += sum(1 for col, info in self.missing_report.items() if info['percentage'] > 20) * 3
        
        # Outlier impact
        risk_score += sum(1 for col, info in self.outlier_report.items() 
                         if (info['outlier_count'] / len(self.cleaned_df) * 100) > 10) * 3
        
        # Negative values business impact
        risk_score += sum(1 for col, info in self.negative_values_found.items() if info['percentage'] > 5) * 2
        
        # Determine risk level with thresholds
        if risk_score >= 15:
            risk_level = 'High'
            risk_description = 'Significant data quality issues requiring immediate attention'
        elif risk_score >= 8:
            risk_level = 'Medium'
            risk_description = 'Moderate data concerns that should be addressed before advanced analytics'
        else:
            risk_level = 'Low'
            risk_description = 'Generally clean dataset with minor issues resolved during processing'
        
        # Build comprehensive AI summary
        ai_summary = {
            'total_issues_detected': total_issues,
            'critical_issues': len(critical_issues),
            'warnings': len(warning_issues),
            'total_decisions_made': total_decisions,
            'columns_affected': columns_affected,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'problem_columns': list(problem_columns)[:5],  # Top 5 problem columns
            'data_quality_assessment': 'Ready for analysis' if risk_level == 'Low' else 'Requires review',
            'recommendations': []
        }
        
        # Generate actionable recommendations based on findings
        if critical_issues:
            ai_summary['recommendations'].append(
                f"IMMEDIATE: Address {len(critical_issues)} critical issues before running predictive models"
            )
        
        if len(problem_columns) > 3:
            ai_summary['recommendations'].append(
                f"FOCUS: {len(problem_columns)} columns need attention - prioritize data quality improvement efforts"
            )
        
        if self.negative_values_found:
            ai_summary['recommendations'].append(
                "INVESTIGATE: Negative values detected - verify business logic and ensure proper categorization of refunds/returns"
            )
        
        if self.outlier_report:
            ai_summary['recommendations'].append(
                "ANALYZE: Extreme outliers present - consider segment-specific analysis or robust statistical methods"
            )
        
        if not ai_summary['recommendations']:
            ai_summary['recommendations'].append(
                "PROCEED: Data quality is acceptable for standard analytical workflows"
            )
        
        # Store AI summary in instance
        self.ai_summary = ai_summary
        
        # Insert executive summary as first insight
        executive_summary = (
            f"EXECUTIVE SUMMARY: Analyzed {len(self.original_df)} rows × {len(self.original_df.columns)} columns. "
            f"Made {total_decisions} intelligent decisions affecting {columns_affected} columns. "
            f"Detected {total_issues} issues ({len(critical_issues)} critical). "
            f"Overall Risk: {risk_level} - {risk_description}"
        )
        
        self.ai_insights.insert(0, executive_summary)
        
        print(f"  ✓ Generated {len(self.ai_insights)} insights (including executive summary)")
        print(f"  ✓ Generated {len(self.warnings)} warnings")
        print(f"  ✓ Critical Issues: {len(critical_issues)}")
        print(f"  ✓ Problem Columns: {', '.join(ai_summary['problem_columns'][:3]) if ai_summary['problem_columns'] else 'None'}")
        print(f"  ✓ Risk Level: {risk_level}")
        print(f"  ✓ Recommendations: {len(ai_summary['recommendations'])}")
    
    def auto_clean(self, convert_types: bool = True, 
                  handle_outliers: bool = True,
                  outlier_method: str = 'cap',
                  outlier_threshold: float = 1.5,
                  safe_mode: bool = False) -> Dict[str, Any]:
        """
        Intelligent decision-driven automatic cleaning pipeline.
        
        Pipeline:
        1. Detect column types
        2. Analyze missing values
        3. Auto-convert types (if enabled)
        4. Decision engine selects strategies with reasoning
        5. Detect and handle outliers (if enabled)
        6. Generate comprehensive report with decision log
        
        Args:
            convert_types: Whether to auto-convert column types
            handle_outliers: Whether to detect and handle outliers
            outlier_method: 'cap' (winsorize) or 'remove'
            outlier_threshold: IQR threshold for outlier detection
            safe_mode: If True, no row dropping, only fill + cap
        
        Returns:
            Comprehensive cleaning report with decision_log
        """
        # EXECUTION CHECKPOINT: Pipeline start
        self.log("AUTO CLEAN PIPELINE STARTED")
        self.log(f"Parameters: convert_types={convert_types}, handle_outliers={handle_outliers}, safe_mode={safe_mode}")
        
        # Initialize tracking
        self.fill_operations = {}
        self.decision_log = []  # Store decision reasoning
        self.ai_insights = []   # High-level insights
        self.warnings = []      # Critical warnings
        
        # STEP 1: Column detection
        self.log("STEP 1/6: Column type detection")
        try:
            self.detect_column_types()
            self.log(f"✓ Detected {len(self.column_types)} columns")
        except Exception as e:
            self.log(f"✗ FAILED: Column type detection - {str(e)}")
            raise
        
        # STEP 2: Missing values
        self.log("STEP 2/6: Missing value analysis")
        try:
            self.analyze_missing_values()
            if self.missing_report:
                self.log(f"✓ Found {len(self.missing_report)} columns with missing values")
            else:
                self.log("✓ No missing values detected")
        except Exception as e:
            self.log(f"✗ FAILED: Missing value analysis - {str(e)}")
            raise
        
        # STEP 3: Type conversion
        if convert_types:
            self.log("STEP 3/6: Type conversion")
            conversions = {}
            
            for col, col_type in self.column_types.items():
                if col_type == 'numeric_convertible':
                    conversions[col] = 'numeric'
                elif col_type == 'datetime_convertible':
                    inferred_format = self._infer_datetime_format(col)
                    if inferred_format:
                        conversions[col] = 'datetime'
            
            if conversions:
                try:
                    self.convert_columns(conversions)
                    self.log(f"✓ Converted {len(conversions)} columns")
                except Exception as e:
                    self.log(f"✗ FAILED: Type conversion - {str(e)}")
                    raise
            else:
                self.log("✓ No conversions needed")
        else:
            self.log("STEP 3/6: Type conversion SKIPPED (disabled)")
        
        # STEP 4: Missing value handling
        self.log("STEP 4/6: Missing value strategy selection")
        auto_strategies = {}
        
        for col in self.missing_report.keys():
            if safe_mode and self._is_id_column(col):
                self.log(f"  SKIP: {col} (ID column, safe mode)")
                continue
            
            decision = self._decide_strategy_with_reasoning(col)
            strategy = decision['strategy']
            reason = decision['reason']
            confidence = decision.get('confidence', 0.5)
            
            if safe_mode and strategy == 'drop':
                strategy = 'median'
                reason += ' (overridden to median in safe mode)'
            
            auto_strategies[col] = strategy
            
            self.decision_log.append({
                'column': col,
                'strategy': strategy,
                'reason': reason,
                'confidence': confidence
            })
        
        if auto_strategies:
            try:
                self.handle_missing_values(auto_strategies)
                self.fill_operations = auto_strategies
                self.log(f"✓ Handled missing values in {len(auto_strategies)} columns")
            except Exception as e:
                self.log(f"✗ FAILED: Missing value handling - {str(e)}")
                raise
        else:
            self.log("✓ No missing values to handle")
        
        # STEP 5: Outlier detection (with business protection)
        if handle_outliers:
            self.log(f"STEP 5/6: Outlier detection ({outlier_method})")
            try:
                self.detect_outliers_iqr(threshold=outlier_threshold)
                
                if self.outlier_report:
                    total_outliers = sum(info['outlier_count'] for info in self.outlier_report.values())
                    business_cols = [col for col, info in self.outlier_report.items() if info.get('is_business_critical')]
                    statistical_cols = [col for col, info in self.outlier_report.items() if not info.get('is_business_critical')]
                    
                    self.log(f"✓ Detected {total_outliers} outliers")
                    if business_cols:
                        self.log(f"  Business-critical: {', '.join(business_cols[:3])}")
                    if statistical_cols:
                        self.log(f"  Statistical: {', '.join(statistical_cols[:3])}")
                else:
                    self.log("✓ No outliers detected")
            except Exception as e:
                self.log(f"✗ FAILED: Outlier detection - {str(e)}")
                raise
        else:
            self.log("STEP 5/6: Outlier handling SKIPPED (disabled)")
        
        # ENHANCED INTELLIGENCE STEPS
        # STEP 5.1: Duplicate detection
        self.log("STEP 5.1/6: Duplicate detection")
        try:
            self.detect_and_handle_duplicates()
            if self.duplicates_found > 0:
                self.log(f"✓ Found {self.duplicates_found} duplicates")
            else:
                self.log("✓ No duplicates found")
        except Exception as e:
            self.log(f"✗ FAILED: Duplicate detection - {str(e)}")
        
        # STEP 5.2: Negative value detection (BEFORE outlier capping)
        self.log("STEP 5.2/6: Negative value detection (business context)")
        try:
            numeric_cols = [col for col in self.cleaned_df.columns if pd.api.types.is_numeric_dtype(self.cleaned_df[col])]
            self.detect_negative_values(numeric_cols[:5])
            if self.negative_values_found:
                self.log(f"✓ Detected negatives in {len(self.negative_values_found)} columns: {', '.join(list(self.negative_values_found.keys())[:3])}")
            else:
                self.log("✓ No negative values detected")
        except Exception as e:
            self.log(f"✗ FAILED: Negative value detection - {str(e)}")
        
        # STEP 5.3: Skewness analysis
        self.log("STEP 5.3/6: Skewness analysis")
        try:
            self.analyze_skewness(skew_threshold=1.0)
            if self.skewness_report:
                self.log(f"✓ Found {len(self.skewness_report)} skewed columns")
            else:
                self.log("✓ No significant skewness detected")
        except Exception as e:
            self.log(f"✗ FAILED: Skewness analysis - {str(e)}")
        
        # STEP 5.4: Datatype standardization
        self.log("STEP 5.4/6: Datatype standardization")
        try:
            self.standardize_datatypes()
            if self.datatype_issues:
                self.log(f"✓ Standardized {len(self.datatype_issues)} columns")
            else:
                self.log("✓ No datatype standardization needed")
        except Exception as e:
            self.log(f"✗ FAILED: Datatype standardization - {str(e)}")
        
        # STEP 5.5: Insight generation
        self.log("STEP 5.5/6: AI insight generation")
        try:
            self._generate_business_insights()
            self.log(f"✓ Generated {len(self.ai_insights)} insights, {len(self.warnings)} warnings")
        except Exception as e:
            self.log(f"✗ FAILED: Insight generation - {str(e)}")
            raise
        
        # FINAL: Report generation
        self.log("FINAL: Generating comprehensive report")
        report = self.generate_cleaning_report()
        
        # Add decision log to report
        report['decision_log'] = self.decision_log
        
        # ENHANCED INTELLIGENCE: Add insights and warnings
        report['ai_insights'] = self.ai_insights
        report['critical_warnings'] = self.warnings
        
        # Add comprehensive AI summary (analyst-level intelligence)
        if hasattr(self, 'ai_summary'):
            report['ai_summary'] = self.ai_summary
        else:
            # Fallback summary if _generate_business_insights wasn't called
            total_decisions = len(self.decision_log)
            columns_affected = len(set(d['column'] for d in self.decision_log)) if self.decision_log else 0
            risk_score = len(self.warnings) * 2 + len([i for i in self.ai_insights if 'extreme' in i.lower()]) * 3
            risk_level = 'High' if risk_score >= 15 else ('Medium' if risk_score >= 8 else 'Low')
            
            report['ai_summary'] = {
                'total_issues_detected': len(self.warnings),
                'critical_issues': len([w for w in self.warnings if 'CRITICAL' in w]),
                'warnings': len([w for w in self.warnings if 'WARNING' in w]),
                'total_decisions_made': total_decisions,
                'columns_affected': columns_affected,
                'risk_level': risk_level,
                'problem_columns': list(self.negative_values_found.keys())[:5],
                'data_quality_assessment': 'Ready for analysis' if risk_level == 'Low' else 'Requires review',
                'recommendations': ['Review critical issues before proceeding'] if risk_level != 'Low' else ['Data quality acceptable']
            }
        
        # Legacy intelligence_summary for backward compatibility
        total_decisions = len(self.decision_log)
        columns_affected = len(set(d['column'] for d in self.decision_log)) if self.decision_log else 0
        
        report['intelligence_summary'] = {
            'total_decisions': total_decisions,
            'columns_affected': columns_affected,
            'risk_level': report['ai_summary']['risk_level'],
            'insights_count': len(self.ai_insights),
            'warnings_count': len(self.warnings),
            'duplicates_removed': self.duplicates_found,
            'negative_values_found': {col: info['count'] for col, info in self.negative_values_found.items()},
            'skewed_columns': list(self.skewness_report.keys()),
            'ai_summary_available': True
        }
        
        # EXECUTION CHECKPOINT: Pipeline completion
        ai_summary = report.get('ai_summary', {})
        risk_level = ai_summary.get('risk_level', 'Unknown')
        
        self.log("="*60)
        self.log("AUTO-CLEANING PIPELINE COMPLETED")
        self.log(f"Decisions: {len(self.decision_log)}, Insights: {len(self.ai_insights)}, Warnings: {len(self.warnings)}")
        self.log(f"Risk Level: {risk_level}, Quality Score: {report['data_quality_score']}/100")
        self.log("="*60)
        
        return report
    
    def get_cleaned_data(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        return self.cleaned_df
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a quick summary of cleaning operations."""
        return {
            'shape': list(self.cleaned_df.shape),
            'column_types': self.column_types,
            'missing_columns': len(self.missing_report),
            'outlier_columns': len(self.outlier_report),
            'actions_taken': len(self.cleaning_actions)
        }
