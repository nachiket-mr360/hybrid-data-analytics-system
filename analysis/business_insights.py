"""
Business Insights Engine
Generates actionable decision-level insights from retail data analysis
"""
import pandas as pd
import numpy as np


def analyze_revenue_performance(df, retail_results):
    """Analyze revenue performance and generate actionable insights"""
    insights = []
    
    # Check if we have store performance data
    if 'store_performance' in retail_results and retail_results['store_performance'] is not None:
        store_perf = retail_results['store_performance']
        
        if len(store_perf) > 0:
            # Identify top performing store
            top_store_idx = store_perf['Total_Revenue'].idxmax()
            top_store = store_perf.loc[top_store_idx, 'Store']
            top_revenue = store_perf.loc[top_store_idx, 'Total_Revenue']
            
            # Calculate average revenue
            avg_revenue = store_perf['Total_Revenue'].mean()
            
            # Generate insight for top performer
            if top_revenue > avg_revenue * 1.5:  # 50% above average
                insights.append({
                    'type': 'opportunity',
                    'category': 'Revenue Performance',
                    'message': f"Store '{top_store}' is significantly outperforming others with ${top_revenue:.2f} in revenue",
                    'recommendation': "Study successful practices at this location and replicate them across underperforming stores",
                    'impact': 'high'
                })
            
            # Identify lowest performer
            low_store_idx = store_perf['Total_Revenue'].idxmin()
            low_store = store_perf.loc[low_store_idx, 'Store']
            low_revenue = store_perf.loc[low_store_idx, 'Total_Revenue']
            
            if low_revenue < avg_revenue * 0.5:  # 50% below average
                insights.append({
                    'type': 'warning',
                    'category': 'Revenue Performance',
                    'message': f"Store '{low_store}' is underperforming with only ${low_revenue:.2f} in revenue",
                    'recommendation': "Investigate local market conditions, staffing, or operational issues at this location",
                    'impact': 'high'
                })
    
    # Analyze product performance
    if 'top_products' in retail_results and retail_results['top_products'] is not None:
        top_products = retail_results['top_products']
        
        if len(top_products) > 0:
            top_product = top_products.iloc[0]['Product']
            top_qty = top_products.iloc[0]['Quantity']
            
            insights.append({
                'type': 'opportunity',
                'category': 'Inventory Management',
                'message': f"'{top_product}' is your best-selling product with {top_qty} units sold",
                'recommendation': f"Ensure adequate inventory levels for '{top_product}' and consider bundling with complementary products",
                'impact': 'medium'
            })
            
            # Check for products that could be phased out
            if len(top_products) > 3:
                bottom_product = top_products.iloc[-1]['Product']
                bottom_qty = top_products.iloc[-1]['Quantity']
                
                if bottom_qty < top_qty * 0.2:  # Less than 20% of top seller
                    insights.append({
                        'type': 'info',
                        'category': 'Inventory Optimization',
                        'message': f"'{bottom_product}' has very low sales volume ({bottom_qty} units)",
                        'recommendation': "Consider discontinuing or replacing with higher-demand alternatives",
                        'impact': 'low'
                    })
    
    return insights


def analyze_profit_margins(df, retail_results):
    """Analyze profitability and identify margin issues"""
    insights = []
    
    # Check if profit data exists
    if 'total_profit' in retail_results and 'total_revenue' in retail_results:
        total_profit = retail_results['total_profit']
        total_revenue = retail_results['total_revenue']
        
        if total_revenue > 0:
            overall_margin = (total_profit / total_revenue) * 100
            
            if overall_margin < 10:  # Less than 10% margin
                insights.append({
                    'type': 'warning',
                    'category': 'Profitability',
                    'message': f"Overall profit margin is only {overall_margin:.1f}%",
                    'recommendation': "Review pricing strategy and supplier costs. Consider negotiating better terms or adjusting prices on high-volume items",
                    'impact': 'critical'
                })
            elif overall_margin > 30:  # More than 30% margin
                insights.append({
                    'type': 'opportunity',
                    'category': 'Profitability',
                    'message': f"Strong profit margin of {overall_margin:.1f}%",
                    'recommendation': "Consider competitive pricing adjustments to gain market share while maintaining healthy margins",
                    'impact': 'medium'
                })
    
    # Analyze category profitability
    if 'category_performance' in retail_results and retail_results['category_performance'] is not None:
        cat_perf = retail_results['category_performance']
        
        if len(cat_perf) > 0:
            # Find highest profit category
            top_cat_idx = cat_perf['Total_Profit'].idxmax()
            top_cat = cat_perf.loc[top_cat_idx, 'Category']
            top_profit = cat_perf.loc[top_cat_idx, 'Total_Profit']
            
            insights.append({
                'type': 'opportunity',
                'category': 'Category Strategy',
                'message': f"'{top_cat}' category generates highest profit (${top_profit:.2f})",
                'recommendation': f"Expand product range in '{top_cat}' and prioritize shelf space allocation",
                'impact': 'high'
            })
    
    return insights


def analyze_sales_trends(df, retail_results):
    """Analyze time-based trends and patterns"""
    insights = []
    
    if 'sales_trend' in retail_results and retail_results['sales_trend'] is not None:
        trend_data = retail_results['sales_trend']
        
        if len(trend_data) >= 2:
            # Convert to DataFrame if it's a list of dicts
            if isinstance(trend_data, list):
                trend_df = pd.DataFrame(trend_data)
            else:
                trend_df = trend_data
            
            # Sort by month
            trend_df = trend_df.sort_values('Month')
            
            # Calculate trend direction
            recent_months = trend_df.tail(3)['Total_Revenue'].mean()
            earlier_months = trend_df.head(3)['Total_Revenue'].mean()
            
            if len(trend_df) >= 6:  # At least 6 months of data
                trend_change = ((recent_months - earlier_months) / earlier_months) * 100
                
                if trend_change < -10:  # Declining by more than 10%
                    insights.append({
                        'type': 'warning',
                        'category': 'Sales Trend',
                        'message': f"Sales declining over recent periods (-{abs(trend_change):.1f}%)",
                        'recommendation': "Investigate causes: seasonal factors, competition, or changing customer preferences. Consider promotional campaigns",
                        'impact': 'high'
                    })
                elif trend_change > 15:  # Growing by more than 15%
                    insights.append({
                        'type': 'opportunity',
                        'category': 'Growth',
                        'message': f"Strong upward sales trend (+{trend_change:.1f}% growth)",
                        'recommendation': "Scale operations to meet growing demand. Ensure inventory and staffing can support continued growth",
                        'impact': 'high'
                    })
            
            # Identify peak month
            peak_month_idx = trend_df['Total_Revenue'].idxmax()
            peak_month = trend_df.loc[peak_month_idx, 'Month']
            
            insights.append({
                'type': 'info',
                'category': 'Seasonal Planning',
                'message': f"Peak sales period is {peak_month}",
                'recommendation': f"Plan inventory buildup and staffing increases before {peak_month}. Consider pre-season marketing campaigns",
                'impact': 'medium'
            })
    
    return insights


def analyze_product_category_mix(df, retail_results):
    """Analyze product and category mix optimization"""
    insights = []
    
    if 'category_performance' in retail_results and retail_results['category_performance'] is not None:
        cat_perf = retail_results['category_performance']
        
        if len(cat_perf) > 1:
            # Calculate concentration risk
            total_cat_revenue = cat_perf['Total_Profit'].sum()
            
            if total_cat_revenue > 0:
                # Check if any single category dominates
                max_cat_profit = cat_perf['Total_Profit'].max()
                concentration = (max_cat_profit / total_cat_revenue) * 100
                
                if concentration > 60:  # More than 60% dependent on one category
                    dominant_cat = cat_perf.loc[cat_perf['Total_Profit'].idxmax(), 'Category']
                    insights.append({
                        'type': 'warning',
                        'category': 'Diversification',
                        'message': f"Heavy reliance on '{dominant_cat}' category ({concentration:.0f}% of profits)",
                        'recommendation': "Diversify product portfolio to reduce risk. Invest in developing other categories",
                        'impact': 'high'
                    })
    
    return insights


def generate_business_insights(df, retail_results):
    """
    Main function to generate comprehensive business insights
    Returns structured insights with recommendations
    """
    all_insights = []
    
    # Only generate insights if we have retail results
    if retail_results:
        # Revenue and performance insights
        all_insights.extend(analyze_revenue_performance(df, retail_results))
        
        # Profitability insights
        all_insights.extend(analyze_profit_margins(df, retail_results))
        
        # Trend analysis insights
        all_insights.extend(analyze_sales_trends(df, retail_results))
        
        # Product mix insights
        all_insights.extend(analyze_product_category_mix(df, retail_results))
    
    # Sort insights by impact (critical first)
    impact_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    all_insights.sort(key=lambda x: impact_order.get(x.get('impact', 'low'), 3))
    
    return all_insights