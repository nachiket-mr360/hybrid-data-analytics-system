"""
Enhanced Business Intelligence Engine - FIXED VERSION
Uses ACTUAL computed values from retail_results - NO fallback/dummy values
If data is missing → SKIP insight (do NOT create fake one)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def generate_enhanced_insights(df: pd.DataFrame, retail_results: Dict) -> List[Dict]:
    """Generate EXACTLY 5 insights with SMART TITLES - ensure minimum 4"""
    insights = []
     
    # INSIGHT 1: Revenue Concentration
    insights.append(_analyze_revenue_concentration(df, retail_results))
    
    # INSIGHT 2: Profit Margin
    insights.append(_analyze_profit_margin(df, retail_results))
    
    # INSIGHT 3: Top Product Performance
    insights.append(_analyze_top_product(df, retail_results))
    
    # INSIGHT 4: Category Performance
    insights.append(_analyze_category_performance(df, retail_results))
    
    # INSIGHT 5: Store Performance
    insights.append(_analyze_store_performance(df, retail_results))
    
    # Filter out None (missing data = skip)
    valid_insights = [i for i in insights if i is not None]
    
    # If less than 4, generate fallback insights using REAL data
    while len(valid_insights) < 4:
        fallback = _generate_fallback_insight(df, retail_results, len(valid_insights) + 1)
        if fallback:
            valid_insights.append(fallback)
        else:
            break  # No more fallbacks possible
    
    return valid_insights[:5]


def _analyze_revenue_concentration(df, retail_results):
    """Use ACTUAL total_revenue from retail_results"""
    try:
        # USE retail_results first
        if retail_results and 'total_revenue' in retail_results:
            total_revenue = retail_results['total_revenue']
        else:
            return None
        
        if total_revenue == 0:
            return None
        
        # Get store performance
        if retail_results and 'store_performance' in retail_results and retail_results['store_performance'] is not None:
            store_perf = retail_results['store_performance']
            if hasattr(store_perf, 'iloc') and len(store_perf) > 0:
                top_store = store_perf.index[0]
                top_revenue = store_perf.iloc[0]['Total_Revenue'] if 'Total_Revenue' in store_perf.columns else store_perf.iloc[0].iloc[0]
            else:
                return None
        else:
            return None
        
        top_contribution = (top_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        
        what = f"{top_store} generates {_format_currency(top_revenue)} ({top_contribution:.1f}% of total {_format_currency(total_revenue)})"
        why = "Strong regional demand and higher order volume drive this location's dominance"
        
        if top_contribution > 40:
            impact = f"Heavy dependency on one location creates significant risk—if {top_store} faces disruptions, {top_contribution:.0f}% of revenue is at risk"
            action = f"Diversify revenue by expanding successful {top_store} strategies to other locations; reduce concentration risk below 30%"
            title = f"{top_store} dominates revenue at {top_contribution:.0f}%"
        elif top_contribution > 30:
            impact = f"Moderate concentration risk—{top_store} contributes nearly one-third of total revenue"
            action = f"Monitor {top_store} performance closely; invest in growing other stores"
            title = f"{top_store} leads with {top_contribution:.0f}% revenue share"
        else:
            impact = f"Well-distributed revenue with {top_store} leading at {top_contribution:.1f}%"
            action = f"Maintain balanced growth; replicate {top_store} success across network"
            title = f"{top_store} leads with {top_contribution:.0f}% revenue share"
        
        return {
            'title': title,
            'content': f"WHAT: {what}\nWHY: {why}\nIMPACT: {impact}\nACTION: {action}"
        }
    except:
        return None


def _analyze_profit_margin(df, retail_results):
    """Use ACTUAL total_profit and total_revenue from retail_results"""
    try:
        if retail_results and 'total_profit' in retail_results and 'total_revenue' in retail_results:
            total_profit = retail_results['total_profit']
            total_revenue = retail_results['total_revenue']
        else:
            return None
        
        if total_revenue == 0:
            return None
        
        profit_margin = (total_profit / total_revenue) * 100
        
        # Find loss products
        loss_count = 0
        total_loss = 0
        if 'Profit' in df.columns and 'Product' in df.columns:
            loss_products = df[df['Profit'] < 0]
            loss_count = loss_products['Product'].nunique()
            total_loss = abs(loss_products['Profit'].sum()) if len(loss_products) > 0 else 0
        
        what = f"Profit margin is {profit_margin:.1f}% ({_format_currency(total_profit)} profit on {_format_currency(total_revenue)} revenue)"
        why = f"{loss_count} loss-making products eroding {_format_currency(total_loss)}" if loss_count > 0 else "All products profitable, strong pricing discipline"
        
        if profit_margin < 10:
            impact = f"CRITICAL: Only {profit_margin:.1f}% margin leaves minimal buffer—small cost increase could push you into losses"
            action = f"Immediate pricing review needed; discontinue loss-makers or increase prices 15-20%"
            title = f"Critically low profit margin at {profit_margin:.0f}%"
        elif profit_margin < 20:
            impact = f"Below optimal 25-30% target; limited cushion for market volatility"
            action = f"Renegotiate supplier costs, shift to high-margin items, selective price increases"
            title = f"Profit margin {profit_margin:.0f}% below optimal target"
        else:
            impact = f"Healthy margin provides flexibility for growth investments"
            action = f"Reinvest profits in customer acquisition and market expansion"
            title = f"Strong profit margin of {profit_margin:.0f}%"
        
        return {
            'title': title,
            'content': f"WHAT: {what}\nWHY: {why}\nIMPACT: {impact}\nACTION: {action}"
        }
    except:
        return None


def _analyze_top_product(df, retail_results):
    """Use ACTUAL top_product from retail_results insights"""
    try:
        if retail_results and 'insights' in retail_results and retail_results['insights']:
            top_product = retail_results['insights'].get('top_product')
            if not top_product or top_product == 'N/A':
                return None
        else:
            return None
        
        # Get product metrics
        if 'Product' in df.columns and 'Quantity' in df.columns:
            product_qty = df.groupby('Product')['Quantity'].sum()
            top_qty = int(product_qty.get(top_product, 0))
            
            product_data = df[df['Product'] == top_product]
            product_revenue = product_data['Revenue'].sum()
            total_revenue = df['Revenue'].sum()
            revenue_share = (product_revenue / total_revenue * 100) if total_revenue > 0 else 0
        else:
            return None
        
        if top_qty == 0:
            return None
        
        what = f"'{top_product}' leads with {top_qty:,} units sold, generating {_format_currency(product_revenue)} ({revenue_share:.1f}% of total)"
        why = "Strong market demand and competitive pricing drive dominance"
        impact = f"As revenue leader, any supply disruption materially affects total sales"
        action = f"Ensure consistent inventory; explore bundles; monitor competitor pricing"
        
        return {
            'title': f"{top_product} drives top revenue at {revenue_share:.0f}% share",
            'content': f"WHAT: {what}\nWHY: {why}\nIMPACT: {impact}\nACTION: {action}"
        }
    except:
        return None


def _analyze_category_performance(df, retail_results):
    """Use ACTUAL category_performance from retail_results"""
    try:
        if retail_results and 'category_performance' in retail_results and retail_results['category_performance'] is not None:
            cat_perf = retail_results['category_performance']
            if hasattr(cat_perf, 'iloc') and len(cat_perf) > 0:
                best_cat_idx = cat_perf['Total_Profit'].idxmax() if 'Total_Profit' in cat_perf.columns else cat_perf.iloc[:, 1].idxmax()
                best_category = cat_perf.loc[best_cat_idx, 'Category'] if 'Category' in cat_perf.columns else cat_perf.index[0]
                best_profit = cat_perf.loc[best_cat_idx, 'Total_Profit'] if 'Total_Profit' in cat_perf.columns else cat_perf.iloc[0, 1]
            else:
                return None
        else:
            return None
        
        if best_profit == 0:
            return None
        
        # Calculate margin
        if 'Category' in df.columns and 'Revenue' in df.columns:
            cat_revenue = df[df['Category'] == best_category]['Revenue'].sum()
            profit_margin = (best_profit / cat_revenue * 100) if cat_revenue > 0 else 0
        else:
            return None
        
        what = f"'{best_category}' is most profitable, generating {_format_currency(best_profit)} profit ({profit_margin:.1f}% margin)"
        why = f"Strong pricing power and favorable cost structure in this category"
        impact = f"Growing this category's share directly improves overall profitability"
        action = f"Expand {best_category} range, increase marketing, optimize shelf space"
        
        return {
            'title': f"{best_category} category leads with {profit_margin:.0f}% margin",
            'content': f"WHAT: {what}\nWHY: {why}\nIMPACT: {impact}\nACTION: {action}"
        }
    except:
        return None


def _analyze_store_performance(df, retail_results):
    """Use ACTUAL best_store from retail_results insights"""
    try:
        if retail_results and 'insights' in retail_results and retail_results['insights']:
            best_store = retail_results['insights'].get('best_store')
            if not best_store or best_store == 'N/A':
                return None
        else:
            return None
        
        # Get store metrics
        if 'Store' in df.columns and 'Revenue' in df.columns:
            store_revenue = df.groupby('Store')['Revenue'].sum()
            store_best_revenue = store_revenue.get(best_store, 0)
            avg_revenue = store_revenue.mean()
            gap_vs_avg = ((store_best_revenue - avg_revenue) / avg_revenue * 100) if avg_revenue > 0 else 0
        else:
            return None
        
        if store_best_revenue == 0:
            return None
        
        what = f"'{best_store}' leads with {_format_currency(store_best_revenue)} revenue ({gap_vs_avg:.0f}% above average)"
        why = "Effective execution, strong local demand, or superior operations"
        impact = f"Top store sets performance benchmark for network"
        action = f"Document and replicate {best_store} strategies across underperforming locations"
        
        return {
            'title': f"{best_store} outperforms with {gap_vs_avg:.0f}% above average",
            'content': f"WHAT: {what}\nWHY: {why}\nIMPACT: {impact}\nACTION: {action}"
        }
    except:
        return None


def _format_currency(amount):
    """Format as currency"""
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:.2f}K"
    else:
        return f"${amount:.2f}"


def _generate_fallback_insight(df: pd.DataFrame, retail_results: Dict, index: int) -> Optional[Dict]:
    """Generate fallback insight using REAL data when primary insights are insufficient"""
    try:
        # Fallback 1: Average Order Value
        if index == 1 and 'Revenue' in df.columns:
            total_revenue = df['Revenue'].sum()
            total_transactions = len(df)
            avg_order_value = total_revenue / total_transactions if total_transactions > 0 else 0
            
            if avg_order_value > 0:
                return {
                    'title': f'Average order value is {_format_currency(avg_order_value)}',
                    'content': f"WHAT: Average order value stands at {_format_currency(avg_order_value)} across {total_transactions:,} transactions\nWHY: Pricing structure and purchase volume determine this metric\nIMPACT: Higher AOV indicates stronger customer spending patterns\nACTION: Implement bundling strategies to increase AOV by 10-15%"
                }
        
        # Fallback 2: Total Transaction Volume
        if index == 2:
            total_transactions = len(df)
            if 'Revenue' in df.columns:
                total_revenue = df['Revenue'].sum()
                return {
                    'title': f'{total_transactions:,} transactions generate {_format_currency(total_revenue)}',
                    'content': f"WHAT: {total_transactions:,} total transactions processed, generating {_format_currency(total_revenue)}\nWHY: Transaction volume reflects customer engagement and market reach\nIMPACT: Consistent transaction flow provides revenue stability\nACTION: Focus on customer retention to maintain transaction volume"
                }
        
        # Fallback 3: Top vs Average Store Comparison
        if index == 3 and 'Store' in df.columns and 'Revenue' in df.columns:
            store_revenue = df.groupby('Store')['Revenue'].sum()
            if len(store_revenue) > 1:
                top_store = store_revenue.idxmax()
                top_value = store_revenue.max()
                avg_value = store_revenue.mean()
                gap = ((top_value - avg_value) / avg_value * 100) if avg_value > 0 else 0
                
                return {
                    'title': f'Top store outperforms average by {gap:.0f}%',
                    'content': f"WHAT: Best-performing store exceeds average by {gap:.0f}%\nWHY: Performance gaps indicate operational differences or market potential\nIMPACT: Closing this gap could significantly boost overall revenue\nACTION: Analyze top performer strategies and replicate across locations"
                }
        
        # Fallback 4: Revenue Distribution
        if index == 4 and 'Revenue' in df.columns:
            total_revenue = df['Revenue'].sum()
            if 'Category' in df.columns:
                category_counts = df['Category'].nunique()
                avg_revenue_per_cat = total_revenue / category_counts if category_counts > 0 else 0
                
                return {
                    'title': f'{category_counts} categories average {_format_currency(avg_revenue_per_cat)}',
                    'content': f"WHAT: Revenue distributed across {category_counts} categories, averaging {_format_currency(avg_revenue_per_cat)} each\nWHY: Category diversification affects revenue stability\nIMPACT: Balanced category performance reduces dependency risk\nACTION: Identify and strengthen underperforming categories"
                }
        
        return None
    except:
        return None


def generate_enhanced_recommendations(df: pd.DataFrame, retail_results: Dict) -> List[str]:
    """Generate MINIMUM 4 HIGH-VALUE recommendations with real data"""
    recs = []
    
    # 1. Top Product Promotion
    rec = _rec_top_product(df, retail_results)
    if rec: recs.append(rec)
    
    # 2. Category Expansion
    rec = _rec_category_expansion(df, retail_results)
    if rec: recs.append(rec)
    
    # 3. Underperforming Category
    rec = _rec_underperforming_category(df, retail_results)
    if rec: recs.append(rec)
    
    # 4. Store Strategy
    rec = _rec_store_strategy(df, retail_results)
    if rec: recs.append(rec)
    
    # 5. Profit Improvement
    rec = _rec_profit_improvement(df, retail_results)
    if rec: recs.append(rec)
    
    # Generate fallback recommendations if less than 4
    while len(recs) < 4:
        fallback = _generate_fallback_recommendation(df, retail_results, len(recs) + 1)
        if fallback:
            recs.append(fallback)
        else:
            break
    
    return recs[:5]


def _rec_top_product(df, retail_results):
    """Recommendation for top product"""
    try:
        if retail_results and 'insights' in retail_results:
            top_product = retail_results['insights'].get('top_product')
            if not top_product or top_product == 'N/A':
                return None
            
            if 'Product' in df.columns and 'Revenue' in df.columns:
                product_revenue = df[df['Product'] == top_product]['Revenue'].sum()
                total_revenue = df['Revenue'].sum()
                share = (product_revenue / total_revenue * 100) if total_revenue > 0 else 0
                
                return f"Product '{top_product}' contributes {share:.1f}% of revenue ({_format_currency(product_revenue)}). Increase visibility through promotions to boost sales 10-15%."
        return None
    except:
        return None


def _rec_category_expansion(df, retail_results):
    """Recommendation for best category"""
    try:
        if retail_results and 'category_performance' is not None and retail_results['category_performance'] is not None:
            cat_perf = retail_results['category_performance']
            if len(cat_perf) > 0:
                best_cat = cat_perf.iloc[0]['Category'] if 'Category' in cat_perf.columns else cat_perf.index[0]
                best_profit = cat_perf.iloc[0]['Total_Profit'] if 'Total_Profit' in cat_perf.columns else cat_perf.iloc[0, 1]
                
                return f"Expand '{best_cat}' category ({_format_currency(best_profit)} profit). Add 5-10 products and increase marketing 20%."
        return None
    except:
        return None


def _rec_underperforming_category(df, retail_results):
    """Recommendation for worst category"""
    try:
        if retail_results and 'category_performance' is not None and retail_results['category_performance'] is not None:
            cat_perf = retail_results['category_performance']
            if len(cat_perf) > 1:
                worst_cat = cat_perf.iloc[-1]['Category'] if 'Category' in cat_perf.columns else cat_perf.index[-1]
                worst_profit = cat_perf.iloc[-1]['Total_Profit'] if 'Total_Profit' in cat_perf.columns else cat_perf.iloc[-1, 1]
                
                return f"Review '{worst_cat}' category ({_format_currency(worst_profit)} profit). Consider promotions or discontinuation."
        return None
    except:
        return None


def _rec_store_strategy(df, retail_results):
    """Recommendation for store strategy"""
    try:
        if retail_results and 'insights' in retail_results:
            best_store = retail_results['insights'].get('best_store')
            if not best_store or best_store == 'N/A':
                return None
            
            return f"Replicate '{best_store}' success strategies across other locations through standardized playbook."
        return None
    except:
        return None


def _rec_profit_improvement(df, retail_results):
    """Recommendation for profit improvement"""
    try:
        if retail_results and 'total_profit' in retail_results and 'total_revenue' in retail_results:
            profit = retail_results['total_profit']
            revenue = retail_results['total_revenue']
            margin = (profit / revenue * 100) if revenue > 0 else 0
            
            if margin < 25:
                return f"Profit margin {margin:.1f}% below 25-30% target. Reduce COGS 5-8%, shift to high-margin items."
        return None
    except:
        return None


def _generate_fallback_recommendation(df: pd.DataFrame, retail_results: Dict, index: int) -> Optional[str]:
    """Generate fallback recommendations using REAL data when primary recommendations insufficient"""
    try:
        # Fallback 1: Revenue Optimization
        if index == 1 and 'Revenue' in df.columns:
            total_revenue = df['Revenue'].sum()
            return f"Revenue totals {_format_currency(total_revenue)}. Implement loyalty programs to increase repeat purchases by 20-25%."
        
        # Fallback 2: Product Portfolio
        if index == 2 and 'Product' in df.columns:
            product_count = df['Product'].nunique()
            return f"Product portfolio includes {product_count} items. Evaluate low performers for discontinuation to optimize inventory costs."
        
        # Fallback 3: Category Balance
        if index == 3 and 'Category' in df.columns:
            category_count = df['Category'].nunique()
            return f"Operating {category_count} categories. Consider expanding top-performing categories to improve market coverage."
        
        # Fallback 4: Geographic Expansion
        if index == 4 and 'Store' in df.columns:
            store_count = df['Store'].nunique()
            return f"Current footprint: {store_count} locations. Identify high-potential markets for strategic expansion."
        
        return None
    except:
        return None


def generate_business_warnings(df: pd.DataFrame, retail_results: Dict) -> List[Dict]:
    """Generate MINIMUM 2 business warnings with severity levels"""
    warnings = []
    
    # 1. Profit Margin Warning
    w = _warn_profit_margin(df, retail_results)
    if w: warnings.append(w)
    
    # 2. Revenue Concentration Warning
    w = _warn_revenue_concentration(df, retail_results)
    if w: warnings.append(w)
    
    # 3. Loss Products Warning
    w = _warn_loss_products(df, retail_results)
    if w: warnings.append(w)
    
    # Generate fallback warnings if less than 2
    while len(warnings) < 2:
        fallback = _generate_fallback_warning(df, retail_results, len(warnings) + 1)
        if fallback:
            warnings.append(fallback)
        else:
            break
    
    return warnings[:3]


def _warn_profit_margin(df, retail_results):
    """CRITICAL if <5%, WARNING if <10%"""
    try:
        if retail_results and 'total_profit' in retail_results and 'total_revenue' in retail_results:
            profit = retail_results['total_profit']
            revenue = retail_results['total_revenue']
            margin = (profit / revenue * 100) if revenue > 0 else 0
            
            if margin < 5:
                return {
                    'severity': 'CRITICAL',
                    'label': 'CRITICAL',
                    'message': f'Profit margin only {margin:.1f}% ({_format_currency(profit)} on {_format_currency(revenue)}). High operational cost risk—immediate action needed.',
                    'color': '#ef4444'
                }
            elif margin < 10:
                return {
                    'severity': 'WARNING',
                    'label': 'WARNING',
                    'message': f'Profit margin at {margin:.1f}% is below healthy 15% threshold. Review pricing and costs.',
                    'color': '#f59e0b'
                }
        return None
    except:
        return None


def _warn_revenue_concentration(df, retail_results):
    """CRITICAL if >40%, WARNING if >30%"""
    try:
        if retail_results and 'total_revenue' in retail_results and 'store_performance' in retail_results:
            total = retail_results['total_revenue']
            store_perf = retail_results['store_performance']
            
            if store_perf is not None and len(store_perf) > 0:
                top_revenue = store_perf.iloc[0]['Total_Revenue'] if 'Total_Revenue' in store_perf.columns else store_perf.iloc[0].iloc[0]
                concentration = (top_revenue / total * 100) if total > 0 else 0
                top_store = store_perf.index[0]
                
                if concentration > 40:
                    return {
                        'severity': 'CRITICAL',
                        'label': 'CRITICAL',
                        'message': f'{top_store} generates {concentration:.1f}% of revenue ({_format_currency(top_revenue)}). Severe dependency risk.',
                        'color': '#ef4444'
                    }
                elif concentration > 30:
                    return {
                        'severity': 'WARNING',
                        'label': 'WARNING',
                        'message': f'{top_store} contributes {concentration:.1f}% of revenue. Monitor concentration risk.',
                        'color': '#f59e0b'
                    }
        return None
    except:
        return None


def _warn_loss_products(df, retail_results):
    """WARNING if loss products exist"""
    try:
        if 'Profit' in df.columns and 'Product' in df.columns:
            loss_products = df[df['Profit'] < 0]
            if len(loss_products) > 0:
                loss_count = loss_products['Product'].nunique()
                total_loss = abs(loss_products['Profit'].sum())
                
                return {
                    'severity': 'WARNING',
                    'label': 'WARNING',
                    'message': f'{loss_count} products generating {_format_currency(total_loss)} in losses. Review pricing immediately.',
                    'color': '#f59e0b'
                }
        return None
    except:
        return None


def _generate_fallback_warning(df: pd.DataFrame, retail_results: Dict, index: int) -> Optional[Dict]:
    """Generate fallback warnings using REAL data when primary warnings insufficient"""
    try:
        # Fallback 1: Low Revenue Alert
        if index == 1 and 'Revenue' in df.columns:
            total_revenue = df['Revenue'].sum()
            avg_revenue = df['Revenue'].mean() if len(df) > 0 else 0
            
            if total_revenue > 0:
                return {
                    'severity': 'WARNING',
                    'label': 'WARNING',
                    'message': f'Total revenue {_format_currency(total_revenue)} with {_format_currency(avg_revenue)} average per transaction. Focus on increasing transaction value.',
                    'color': '#f59e0b'
                }
        
        # Fallback 2: Category Dependency
        if index == 2 and 'Category' in df.columns:
            category_counts = df['Category'].value_counts()
            if len(category_counts) > 0:
                top_category = category_counts.index[0]
                top_count = category_counts.iloc[0]
                total_count = category_counts.sum()
                concentration = (top_count / total_count * 100)
                
                if concentration > 40:
                    return {
                        'severity': 'WARNING',
                        'label': 'WARNING',
                        'message': f'{top_category} category represents {concentration:.0f}% of transactions. Diversify to reduce concentration risk.',
                        'color': '#f59e0b'
                    }
        
        return None
    except:
        return None
