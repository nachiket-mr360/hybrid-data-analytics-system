"""
Retail Insight Generation Engine
Transforms calculated metrics into meaningful business insights automatically.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List


def generate_retail_insights(df: pd.DataFrame) -> Dict[str, str]:
    """
    Generate comprehensive retail insights from DataFrame with business metrics.
    
    Args:
        df: DataFrame with columns: Store, Product, Category, Quantity, 
            Selling_Price, Cost_Price, Revenue, Profit
    
    Returns:
        Dictionary with insight sentences for business users
    """
    insights = {}
    
    # VALIDATION: Check if DataFrame is empty
    if df.empty:
        print("ERROR: DataFrame is empty - cannot generate insights")
        return {
            'error': "Empty DataFrame provided",
            'top_store': 'N/A - No data available',
            'low_store': 'N/A - No data available',
            'top_product': 'N/A - No data available',
            'slow_product': 'N/A - No data available',
            'best_category': 'N/A - No data available',
            'loss_products': 'N/A - No data available',
            'sales_trend': 'N/A - No data available',
            'revenue_summary': 'N/A - No data available',
            'profit_summary': 'N/A - No data available',
            'strategic_recommendation': 'N/A - No data available'
        }
    
    # VALIDATION: Print DataFrame info for debugging
    print(f"\n [INSIGHT ENGINE] Input DataFrame:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample data (first 2 rows):")
    print(df.head(2))
    
    # Ensure required columns exist
    required_cols = ['Store', 'Product', 'Category', 'Quantity', 'Revenue', 'Profit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n [INSIGHT ENGINE] WARNING: Missing required columns: {missing_cols}")
        return {
            'error': f"Missing required columns: {', '.join(missing_cols)}",
            'top_store': f'N/A - Missing column: {missing_cols[0]}' if missing_cols else 'N/A',
            'low_store': 'N/A - Missing required columns',
            'top_product': 'N/A - Missing required columns',
            'slow_product': 'N/A - Missing required columns',
            'best_category': 'N/A - Missing required columns',
            'loss_products': 'N/A - Missing required columns',
            'sales_trend': 'N/A - Missing required columns'
        }
    
    # VALIDATION: Check for null values in critical columns
    print(f"\n[INSIGHT ENGINE] Data validation:")
    for col in required_cols:
        null_count = df[col].isnull().sum()
        print(f"  {col}: {null_count} nulls out of {len(df)} rows")
    
    # Generate each insight
    print(f"\n[INSIGHT ENGINE] Generating insights...")
    insights['top_store'] = _analyze_top_store(df)
    insights['low_store'] = _analyze_low_store(df)
    insights['top_product'] = _analyze_top_product(df)
    insights['slow_product'] = _analyze_slow_product(df)
    insights['best_category'] = _analyze_best_category(df)
    insights['loss_products'] = _analyze_loss_products(df)
    insights['sales_trend'] = _analyze_sales_trend(df)
    
    # Additional strategic insights
    insights['revenue_summary'] = _generate_revenue_summary(df)
    insights['profit_summary'] = _generate_profit_summary(df)
    insights['strategic_recommendation'] = _generate_strategic_recommendation(insights)
    
    print(f"[INSIGHT ENGINE] ✓ Generated {len(insights)} insights")
    
    return insights


def _analyze_top_store(df: pd.DataFrame) -> str:
    """Find top performing store with revenue contribution"""
    try:
        # Group by store and calculate total revenue
        store_revenue = df.groupby('Store')['Revenue'].sum().reset_index()
        store_revenue.columns = ['Store', 'Total_Revenue']
        
        # Find top store
        top_store_idx = store_revenue['Total_Revenue'].idxmax()
        top_store = store_revenue.loc[top_store_idx, 'Store']
        top_revenue = store_revenue.loc[top_store_idx, 'Total_Revenue']
        
        # Calculate contribution percentage
        total_revenue = df['Revenue'].sum()
        contribution_pct = (top_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        # Find second store for comparison
        store_revenue_sorted = store_revenue.sort_values('Total_Revenue', ascending=False)
        if len(store_revenue_sorted) > 1:
            second_revenue = store_revenue_sorted.iloc[1]['Total_Revenue']
            lead_margin = top_revenue - second_revenue
            lead_pct = (lead_margin / second_revenue * 100) if second_revenue > 0 else 0
        
        # Format revenue
        formatted_revenue = _format_currency(top_revenue)
        formatted_total = _format_currency(total_revenue)
        
        # WHAT: Observation
        what = f"{top_store} leads all locations with {formatted_revenue} in revenue, contributing {contribution_pct:.1f}% of total sales."
        
        # WHY: Reason
        why = f"This strong performance indicates effective market positioning, customer loyalty, or superior operational execution at this location."
        
        # IMPACT: Business effect
        if len(store_revenue_sorted) > 1:
            impact = f"The {lead_pct:.1f}% lead over the second-best store demonstrates market dominance, but also highlights concentration risk if this location faces disruptions."
        else:
            impact = f"This store's dominance makes it critical to overall business success."
        
        # ACTION: What to do
        action = f"Action: Replicate {top_store}'s successful strategies across other locations, invest in maintaining its competitive edge, and develop contingency plans to reduce dependency on a single location."
        
        return f"📊 WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze top store: {str(e)}"


def _analyze_low_store(df: pd.DataFrame) -> str:
    """Find lowest performing store"""
    try:
        # Group by store and calculate total revenue
        store_revenue = df.groupby('Store')['Revenue'].sum().reset_index()
        store_revenue.columns = ['Store', 'Total_Revenue']
        
        # Find lowest store
        low_store_idx = store_revenue['Total_Revenue'].idxmin()
        low_store = store_revenue.loc[low_store_idx, 'Store']
        low_revenue = store_revenue.loc[low_store_idx, 'Total_Revenue']
        
        # Compare to top store and average
        top_revenue = store_revenue['Total_Revenue'].max()
        avg_revenue = store_revenue['Total_Revenue'].mean()
        gap_vs_top = ((top_revenue - low_revenue) / top_revenue * 100) if top_revenue > 0 else 0
        gap_vs_avg = ((avg_revenue - low_revenue) / avg_revenue * 100) if avg_revenue > 0 else 0
        
        formatted_revenue = _format_currency(low_revenue)
        formatted_avg = _format_currency(avg_revenue)
        
        # WHAT: Observation
        what = f"{low_store} generates only {formatted_revenue}, significantly underperforming with a {gap_vs_top:.1f}% gap versus the top store and {gap_vs_avg:.1f}% below average ({formatted_avg})."
        
        # WHY: Reason
        why = f"This performance gap suggests potential issues such as poor location visibility, inadequate staffing, local market conditions, or operational inefficiencies."
        
        # IMPACT: Business effect
        impact = f"Continued underperformance at {low_store} drags down overall profitability and may indicate systemic issues that could spread to other locations if not addressed."
        
        # ACTION: What to do
        action = f"Action: Conduct a detailed audit of {low_store} operations, analyze local competition, review marketing spend, and consider targeted promotions or operational restructuring to close the performance gap."
        
        return f"⚠️ WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze low store: {str(e)}"


def _analyze_top_product(df: pd.DataFrame) -> str:
    """Find top product by quantity sold"""
    try:
        # Group by product and calculate total quantity
        product_quantity = df.groupby('Product')['Quantity'].sum().reset_index()
        product_quantity.columns = ['Product', 'Total_Quantity']
        
        # Find top product
        top_product_idx = product_quantity['Total_Quantity'].idxmax()
        top_product = product_quantity.loc[top_product_idx, 'Product']
        top_quantity = int(product_quantity.loc[top_product_idx, 'Total_Quantity'])
        
        # Calculate revenue and profit for this product
        product_data = df[df['Product'] == top_product]
        product_revenue = product_data['Revenue'].sum()
        product_profit = product_data['Profit'].sum()
        profit_margin = (product_profit / product_revenue * 100) if product_revenue > 0 else 0
        
        # Calculate percentage of total
        total_revenue = df['Revenue'].sum()
        revenue_share = (product_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        formatted_revenue = _format_currency(product_revenue)
        formatted_profit = _format_currency(product_profit)
        
        # WHAT: Observation
        what = f"{top_product} is your bestseller with {top_quantity:,} units sold, generating {formatted_revenue} ({revenue_share:.1f}% of total revenue) and {formatted_profit} in profit ({profit_margin:.1f}% margin)."
        
        # WHY: Reason
        why = f"This product's strong performance indicates high market demand, competitive pricing, effective positioning, or brand recognition that resonates with customers."
        
        # IMPACT: Business effect
        impact = f"As your revenue leader, {top_product} significantly impacts overall business performance. Any supply disruption, price pressure, or demand shift could materially affect total sales."
        
        # ACTION: What to do
        action = f"Action: Ensure consistent inventory availability for {top_product}, explore bundle opportunities with complementary products, monitor competitor pricing, and consider expanding distribution channels to maximize this winning product."
        
        return f"🔥 WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze top product: {str(e)}"


def _analyze_slow_product(df: pd.DataFrame) -> str:
    """Find slow-moving product"""
    try:
        # Group by product and calculate total quantity
        product_quantity = df.groupby('Product')['Quantity'].sum().reset_index()
        product_quantity.columns = ['Product', 'Total_Quantity']
        
        # Filter out products with zero or negative sales
        valid_products = product_quantity[product_quantity['Total_Quantity'] > 0]
        
        if len(valid_products) == 0:
            return "🚨 WHAT: All products show minimal or no sales activity. WHY: This indicates either a new product launch phase, severe market rejection, or data quality issues. IMPACT: Zero movement ties up capital in inventory and warehouse space. ACTION: Immediately review product-market fit, pricing strategy, and marketing effectiveness for all SKUs."
        
        # Find slowest product
        slow_product_idx = valid_products['Total_Quantity'].idxmin()
        slow_product = valid_products.loc[slow_product_idx, 'Product']
        slow_quantity = int(valid_products.loc[slow_product_idx, 'Total_Quantity'])
        
        # Compare to top product and average
        top_quantity = int(product_quantity['Total_Quantity'].max())
        avg_quantity = int(valid_products['Total_Quantity'].mean())
        gap_vs_top = ((top_quantity - slow_quantity) / top_quantity * 100) if top_quantity > 0 else 0
        gap_vs_avg = ((avg_quantity - slow_quantity) / avg_quantity * 100) if avg_quantity > 0 else 0
        
        # Check if product is generating losses
        slow_product_data = df[df['Product'] == slow_product]
        slow_product_profit = slow_product_data['Profit'].sum()
        is_loss_making = slow_product_profit < 0
        
        # WHAT: Observation
        what = f"{slow_product} moves extremely slowly with only {slow_quantity:,} units sold, which is {gap_vs_top:.1f}% below your bestseller and {gap_vs_avg:.1f}% below average."
        
        # WHY: Reason
        why = f"This poor performance likely stems from weak demand, poor product-market fit, inadequate marketing, strong competition, or pricing that doesn't match perceived value."
        
        # IMPACT: Business effect
        if is_loss_making:
            impact = f"CRITICAL: This product is also generating losses, creating a double negative impact—tying up inventory capital while actively reducing profitability."
        else:
            impact = f"Slow movement ties up working capital in inventory, occupies valuable warehouse space, and increases holding costs without meaningful revenue contribution."
        
        # ACTION: What to do
        if is_loss_making:
            action = f"Action: Immediately discontinue or liquidate {slow_product} inventory through clearance sales. Reallocate resources to higher-performing products and investigate root causes before considering similar products in the future."
        else:
            action = f"Action: Consider aggressive discounting to clear {slow_product} inventory, bundle with popular products, or explore alternative sales channels. If performance doesn't improve within 60-90 days, plan for discontinuation."
        
        return f"📉 WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze slow product: {str(e)}"


def _analyze_best_category(df: pd.DataFrame) -> str:
    """Find most profitable category"""
    try:
        # Group by category and calculate total profit and revenue
        category_performance = df.groupby('Category').agg(
            Total_Profit=('Profit', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Units_Sold=('Quantity', 'sum')
        ).reset_index()
        
        # Find best category by profit
        best_category_idx = category_performance['Total_Profit'].idxmax()
        best_category = category_performance.loc[best_category_idx, 'Category']
        best_profit = category_performance.loc[best_category_idx, 'Total_Profit']
        best_revenue = category_performance.loc[best_category_idx, 'Total_Revenue']
        best_units = int(category_performance.loc[best_category_idx, 'Units_Sold'])
        
        # Calculate profit margin
        profit_margin = (best_profit / best_revenue * 100) if best_revenue > 0 else 0
        
        # Compare to other categories
        avg_profit = category_performance['Total_Profit'].mean()
        profit_lead = ((best_profit - avg_profit) / avg_profit * 100) if avg_profit > 0 else 0
        
        # Number of products in category
        products_in_category = df[df['Category'] == best_category]['Product'].nunique()
        
        formatted_profit = _format_currency(best_profit)
        formatted_revenue = _format_currency(best_revenue)
        
        # WHAT: Observation
        what = f"{best_category} is your most profitable category, generating {formatted_profit} in profit on {formatted_revenue} revenue ({profit_margin:.1f}% margin) with {best_units:,} units sold across {products_in_category} products."
        
        # WHY: Reason
        why = f"This category's profitability advantage—{profit_lead:.1f}% above average—suggests strong pricing power, favorable cost structure, high customer demand, or effective category management."
        
        # IMPACT: Business effect
        impact = f"{best_category}'s strong margins make it a key profit driver for the business. Growing this category's share of total sales would directly improve overall company profitability."
        
        # ACTION: What to do
        action = f"Action: Expand {best_category} product range, increase marketing investment, optimize shelf space or online visibility, and analyze whether successful strategies can transfer to lower-performing categories."
        
        return f"⭐ WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze best category: {str(e)}"


def _analyze_loss_products(df: pd.DataFrame) -> str:
    """Identify loss-making products (CRITICAL INSIGHT)"""
    try:
        # Find products with negative profit
        loss_products = df[df['Profit'] < 0].copy()
        
        if len(loss_products) == 0:
            return "✅ WHAT: All products in your portfolio are profitable—no loss-makers detected. WHY: Strong pricing discipline and cost management across the board. IMPACT: Healthy profit profile with no revenue leakage from unprofitable products. ACTION: Maintain current pricing strategy while monitoring margins regularly. Consider whether strategic investments in customer acquisition products might justify temporary losses on select items."
        
        # Group by product to find total loss
        product_loss = loss_products.groupby('Product').agg(
            Total_Loss=('Profit', 'sum'),
            Units_Sold=('Quantity', 'sum'),
            Revenue_Lost=('Revenue', 'sum'),
            Avg_Loss_Per_Unit=('Profit', lambda x: x.sum() / loss_products.loc[x.index, 'Quantity'].sum())
        ).reset_index()
        
        # Sort by total loss (descending - worst first)
        product_loss = product_loss.sort_values('Total_Loss', ascending=True)
        
        # Get top 3 worst loss-makers
        top_loss_products = product_loss.head(3)
        
        # Calculate total impact
        total_loss = product_loss['Total_Loss'].sum()
        total_revenue_affected = product_loss['Revenue_Lost'].sum()
        loss_product_count = len(product_loss)
        total_products = df['Product'].nunique()
        loss_pct = (loss_product_count / total_products * 100) if total_products > 0 else 0
        
        # Calculate as % of total profit
        total_company_profit = df['Profit'].sum()
        profit_impact = (abs(total_loss) / total_company_profit * 100) if total_company_profit > 0 else 0
        
        formatted_total_loss = _format_currency(abs(total_loss))
        formatted_revenue_affected = _format_currency(total_revenue_affected)
        
        # Build insight message
        loss_details = []
        for _, row in top_loss_products.iterrows():
            product = row['Product']
            loss = _format_currency(abs(row['Total_Loss']))
            units = int(row['Units_Sold'])
            loss_details.append(f"{product} ({loss} loss on {units:,} units)")
        
        # WHAT: Observation
        what = f"CRITICAL: {loss_product_count} out of {total_products} products ({loss_pct:.1f}%) are generating losses totaling {formatted_total_loss}, while still generating {formatted_revenue_affected} in revenue."
        
        # WHY: Reason
        why = f"These products are priced below total cost (including COGS, operations, and distribution), or costs have increased without corresponding price adjustments. Common causes include promotional pricing, outdated cost data, or intentional loss-leader strategies that aren't driving incremental sales."
        
        # IMPACT: Business effect
        impact = f"These loss-makers are eroding {profit_impact:.1f}% of your total profit. While some loss-leading can be strategic for customer acquisition, uncontrolled losses directly reduce company profitability and waste resources on non-viable products."
        
        # ACTION: What to do
        action = f"Action: Immediately review {', '.join([p['Product'] for _, p in top_loss_products.iterrows()])}. Options include: (1) Increase prices to profitable levels, (2) Negotiate lower supplier costs, (3) Discontinue and liquidate inventory, or (4) If strategic loss-leaders, ensure they're driving measurable cross-sell uplift to justify losses."
        
        return f"🚨 WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze loss products: {str(e)}"


def _analyze_sales_trend(df: pd.DataFrame) -> str:
    """Analyze sales trend if date column exists"""
    try:
        # Try to find date column
        date_col = None
        date_keywords = ['date', 'order_date', 'transaction_date', 'time', 'timestamp']
        
        for keyword in date_keywords:
            for col in df.columns:
                if keyword.lower() in col.lower():
                    date_col = col
                    break
            if date_col:
                break
        
        if not date_col:
            return "📈 WHAT: Time-based trend analysis is unavailable because your dataset doesn't include date columns. WHY: Trend insights require temporal data to identify patterns over time. IMPACT: Without trend visibility, you may miss seasonal patterns, growth opportunities, or early warning signs of decline. ACTION: Add date/time columns to your data exports to unlock powerful trend analysis, seasonality detection, and forecasting capabilities."
        
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        
        if len(df_copy) < 10:
            return "📈 WHAT: Insufficient historical data for meaningful trend analysis (only " + str(len(df_copy)) + " records with dates). WHY: Statistical trend detection requires adequate sample size to distinguish signal from noise. IMPACT: Limited data prevents reliable pattern identification. ACTION: Accumulate at least 30-60 days of transaction history before running trend analysis."
        
        # Sort by date
        df_copy = df_copy.sort_values(date_col)
        
        # Aggregate daily revenue
        daily_revenue = df_copy.groupby(date_col)['Revenue'].sum().reset_index()
        
        if len(daily_revenue) < 7:
            return "📈 WHAT: Only " + str(len(daily_revenue)) + " days of data available—too short for reliable trend analysis. WHY: Weekly or longer patterns require 2+ weeks of data to establish baselines. IMPACT: Short timeframes may show daily volatility but not meaningful trends. ACTION: Continue collecting data for at least 14-30 days to enable accurate trend detection."
        
        # Split into first half and second half
        mid_point = len(daily_revenue) // 2
        first_half_avg = daily_revenue.iloc[:mid_point]['Revenue'].mean()
        second_half_avg = daily_revenue.iloc[mid_point:]['Revenue'].mean()
        
        # Calculate trend
        trend_pct = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        
        # Calculate volatility (standard deviation)
        revenue_volatility = daily_revenue['Revenue'].std()
        volatility_pct = (revenue_volatility / daily_revenue['Revenue'].mean() * 100) if daily_revenue['Revenue'].mean() > 0 else 0
        
        first_date = daily_revenue.iloc[0][date_col].strftime('%Y-%m-%d')
        last_date = daily_revenue.iloc[-1][date_col].strftime('%Y-%m-%d')
        avg_daily = _format_currency(daily_revenue['Revenue'].mean())
        
        if trend_pct > 5:
            direction = "increasing"
            emoji = "📈"
            what = f"Sales show a strong upward trend of +{trend_pct:.1f}% from {first_date} to {last_date}, growing from {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall average)."
            why = f"This positive momentum suggests growing market demand, successful marketing campaigns, improved customer acquisition, or seasonal tailwinds driving increased transaction volume."
            impact = f"Sustained growth at this rate could add significant annual revenue. However, rapid growth may strain inventory, staffing, and operational capacity if not properly scaled."
            action = f"Action: Capitalize on momentum by increasing marketing spend, expanding inventory to prevent stockouts, hiring ahead of demand curve, and ensuring operational infrastructure scales with growth trajectory."
        elif trend_pct < -5:
            direction = "decreasing"
            emoji = "📉"
            what = f"Sales are declining at -{trend_pct:.1f}% from {first_date} to {last_date}, dropping from {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall average)."
            why = f"This downward trend indicates potential issues such as increased competition, market saturation, customer churn, product lifecycle maturity, economic headwinds, or diminished marketing effectiveness."
            impact = f"If this decline continues, it will materially impact quarterly/annual revenue targets and may signal deeper structural issues requiring strategic intervention."
            action = f"Action: Immediately investigate root causes—survey customers, analyze competitor activity, review marketing ROI, assess product relevance. Implement retention campaigns, refresh marketing messaging, and consider promotional activities to reverse the trend."
        else:
            direction = "stable"
            emoji = "➡️"
            what = f"Sales remain relatively flat with only {trend_pct:+.1f}% change from {first_date} to {last_date}, maintaining {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall average)."
            why = f"This stability suggests mature market conditions with balanced supply and demand, but also indicates potential growth stagnation or market saturation."
            impact = f"While stability provides predictable cash flow, it also means you're not capturing market share growth. Inflation and rising costs will gradually erode real profitability without revenue growth."
            action = f"Action: Break through stagnation by introducing new products, entering untapped segments, launching loyalty programs, or exploring new distribution channels. Consider competitive differentiation strategies to gain market share."
        
        # Add volatility context
        if volatility_pct > 40:
            action += f" Note: Daily revenue volatility is high at {volatility_pct:.0f}%, indicating inconsistent sales patterns—focus on smoothing demand through promotions or subscription models."
        
        return f"{emoji} WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not analyze sales trend: {str(e)}"


def _generate_revenue_summary(df: pd.DataFrame) -> str:
    """Generate overall revenue summary"""
    try:
        total_revenue = df['Revenue'].sum()
        total_orders = len(df)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        # Calculate revenue concentration
        store_revenue = df.groupby('Store')['Revenue'].sum().sort_values(ascending=False)
        top_store_revenue = store_revenue.iloc[0] if len(store_revenue) > 0 else 0
        top_store_share = (top_store_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        formatted_revenue = _format_currency(total_revenue)
        formatted_avg = _format_currency(avg_order_value)
        
        # Get unique stores and products
        unique_stores = df['Store'].nunique()
        unique_products = df['Product'].nunique()
        
        # WHAT: Observation
        what = f"Total revenue stands at {formatted_revenue} from {total_orders:,} transactions across {unique_stores} stores and {unique_products} products. Average order value is {formatted_avg}."
        
        # WHY: Reason
        why = f"This revenue base reflects your current market reach, customer demand, product portfolio effectiveness, and operational capacity.{' Top store contributes ' + str(round(top_store_share, 1)) + '% of total sales, indicating concentration.' if top_store_share > 40 else ' Revenue is reasonably distributed across locations.'}"
        
        # IMPACT: Business effect
        impact = f"This revenue foundation determines your market position and growth trajectory. Understanding revenue concentration helps assess business risk and expansion opportunities."
        
        # ACTION: What to do
        action = f"Action: Diversify revenue streams by expanding product offerings, entering new markets, or increasing customer frequency. Monitor top store dependency and develop strategies to balance revenue distribution."
        
        return f"📊 WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not generate revenue summary: {str(e)}"


def _generate_profit_summary(df: pd.DataFrame) -> str:
    """Generate overall profit summary"""
    try:
        total_revenue = df['Revenue'].sum()
        total_profit = df['Profit'].sum()
        total_cost = total_revenue - total_profit
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        # Benchmark against industry standards
        if profit_margin > 30:
            health = "Excellent"
            emoji = "🌟"
            benchmark = "well above the 10-20% retail industry average, indicating exceptional cost management and pricing power."
        elif profit_margin > 20:
            health = "Strong"
            emoji = "✅"
            benchmark = "above the 10-20% retail industry average, reflecting healthy pricing and cost control."
        elif profit_margin > 10:
            health = "Moderate"
            emoji = "⚠️"
            benchmark = "within the typical 10-20% retail range, but there's room for improvement through cost optimization or pricing adjustments."
        else:
            health = "Below Average"
            emoji = "🚨"
            benchmark = "below the 10-20% retail industry average, indicating significant margin pressure that requires immediate attention."
        
        formatted_revenue = _format_currency(total_revenue)
        formatted_profit = _format_currency(total_profit)
        formatted_cost = _format_currency(total_cost)
        
        # Calculate profit per transaction
        total_orders = len(df)
        profit_per_order = total_profit / total_orders if total_orders > 0 else 0
        formatted_per_order = _format_currency(profit_per_order)
        
        # WHAT: Observation
        what = f"Your profit margin is {profit_margin:.1f}%—{health} performance. From {formatted_revenue} in revenue, costs are {formatted_cost}, leaving {formatted_profit} net profit ({formatted_per_order} per order)."
        
        # WHY: Reason
        why = f"This margin profile reflects your current pricing strategy, product mix, cost structure, and operational efficiency. {benchmark}"
        
        # IMPACT: Business effect
        if profit_margin < 15:
            impact = f"At {profit_margin:.1f}% margin, you have limited buffer for cost increases, discounts, or market downturns. A small uptick in costs could push you into unprofitable territory."
        else:
            impact = f"At {profit_margin:.1f}% margin, you have healthy cushion to absorb cost fluctuations, invest in growth initiatives, and weather market volatility."
        
        # ACTION: What to do
        if profit_margin < 20:
            action = f"Action: Focus on margin improvement through (1) Renegotiating supplier contracts to reduce COGS, (2) Optimizing product mix toward higher-margin items, (3) Reducing operational waste, (4) Implementing selective price increases on price-insensitive products."
        else:
            action = f"Action: Maintain margin discipline while exploring growth opportunities. Consider reinvesting profits in customer acquisition, product development, or market expansion to scale the business."
        
        return f"{emoji} WHAT: {what} WHY: {why} IMPACT: {impact} {action}"
    except Exception as e:
        return f"Could not generate profit summary: {str(e)}"


def _generate_strategic_recommendation(insights: Dict[str, str]) -> str:
    """Generate strategic recommendation based on all insights"""
    try:
        recommendations = []
        
        # Extract key information from insights
        top_store_insight = insights.get('top_store', '')
        low_store_insight = insights.get('low_store', '')
        top_product_insight = insights.get('top_product', '')
        loss_insight = insights.get('loss_products', '')
        trend_insight = insights.get('sales_trend', '')
        profit_insight = insights.get('profit_summary', '')
        category_insight = insights.get('best_category', '')
        
        # PRIORITY 1: Address loss products (Critical)
        if 'WARNING' in loss_insight or 'CRITICAL' in loss_insight:
            recommendations.append(
                f"IMMEDIATE ACTION REQUIRED: Eliminate loss-making products to stop profit leakage. "
                f"Review pricing, negotiate supplier costs, or discontinue unprofitable items."
            )
        
        # PRIORITY 2: Address declining trends (Critical)
        if 'decreasing' in trend_insight.lower() and ('declining' in trend_insight.lower() or 'dropping' in trend_insight.lower()):
            recommendations.append(
                f"URGENT: Reverse declining sales trend through customer retention programs, "
                f"competitive analysis, and targeted marketing campaigns to regain momentum."
            )
        
        # PRIORITY 3: Improve profit margins (High)
        if 'Below Average' in profit_insight or 'Moderate' in profit_insight:
            recommendations.append(
                f"Optimize profit margins by renegotiating supplier contracts, "
                f"shifting product mix toward high-margin items, and implementing strategic price adjustments."
            )
        
        # PRIORITY 4: Leverage top performers (Strategic)
        if top_store_insight and 'contribute' in top_store_insight.lower():
            recommendations.append(
                f"Scale successful strategies from top-performing locations to underperforming stores. "
                f"Document best practices and implement standardized operational excellence across all locations."
            )
        
        # PRIORITY 5: Grow winning products (Strategic)
        if top_product_insight and 'bestseller' in top_product_insight.lower():
            recommendations.append(
                f"Maximize visibility and distribution of bestselling products through expanded marketing, "
                f"bundle promotions, and inventory prioritization to capitalize on proven demand."
            )
        
        # PRIORITY 6: Expand profitable categories (Growth)
        if category_insight and 'profitable' in category_insight.lower():
            recommendations.append(
                f"Invest in expanding the most profitable category through new product introductions, "
                f"increased marketing spend, and enhanced shelf space or online prominence."
            )
        
        # PRIORITY 7: Address underperforming locations (Operational)
        if low_store_insight and 'underperforming' in low_store_insight.lower():
            recommendations.append(
                f"Conduct comprehensive operational audit of lowest-performing location, "
                f"implement turnaround plan with specific KPIs, and set 90-day improvement milestones."
            )
        
        if recommendations:
            return f"🎯 Strategic Priorities: " + " | ".join(recommendations[:3])
        else:
            return (
                f"🎯 Strategic Priority: Maintain current performance trajectory while identifying "
                f"opportunities for market share growth, operational efficiency gains, and product portfolio optimization. "
                f"Focus on sustaining competitive advantages and building customer loyalty programs."
            )
    except Exception as e:
        return f"Could not generate strategic recommendation: {str(e)}"


def _format_currency(amount: float) -> str:
    """Format amount as currency with appropriate unit"""
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:.2f}K"
    else:
        return f"${amount:.2f}"


def get_insight_cards(insights: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Convert insights dictionary into card format for frontend display.
    
    Returns:
        List of dictionaries with icon, title, insight, and priority
    """
    cards = []
    
    # Revenue Summary (High Priority)
    if 'revenue_summary' in insights:
        cards.append({
            'icon': '💰',
            'title': 'Revenue Overview',
            'insight': insights['revenue_summary'],
            'priority': 'high',
            'color': '#4CAF50'  # Green
        })
    
    # Profit Health (High Priority)
    if 'profit_summary' in insights:
        cards.append({
            'icon': '📊',
            'title': 'Profit Health',
            'insight': insights['profit_summary'],
            'priority': 'high',
            'color': '#2196F3'  # Blue
        })
    
    # Top Store
    if 'top_store' in insights:
        cards.append({
            'icon': '🏆',
            'title': 'Top Performing Store',
            'insight': insights['top_store'],
            'priority': 'medium',
            'color': '#FF9800'  # Orange
        })
    
    # Low Store
    if 'low_store' in insights:
        cards.append({
            'icon': '⚠️',
            'title': 'Improvement Opportunity',
            'insight': insights['low_store'],
            'priority': 'medium',
            'color': '#FF5722'  # Deep Orange
        })
    
    # Top Product
    if 'top_product' in insights:
        cards.append({
            'icon': '🔥',
            'title': 'Best Seller',
            'insight': insights['top_product'],
            'priority': 'medium',
            'color': '#E91E63'  # Pink
        })
    
    # Slow Product
    if 'slow_product' in insights:
        cards.append({
            'icon': '📉',
            'title': 'Slow Mover',
            'insight': insights['slow_product'],
            'priority': 'low',
            'color': '#9E9E9E'  # Grey
        })
    
    # Best Category
    if 'best_category' in insights:
        cards.append({
            'icon': '⭐',
            'title': 'Most Profitable Category',
            'insight': insights['best_category'],
            'priority': 'medium',
            'color': '#9C27B0'  # Purple
        })
    
    # Loss Products (CRITICAL)
    if 'loss_products' in insights:
        is_critical = 'WARNING' in insights['loss_products'] or 'loss' in insights['loss_products'].lower()
        cards.append({
            'icon': '🚨',
            'title': 'Loss-Making Products',
            'insight': insights['loss_products'],
            'priority': 'critical' if is_critical else 'low',
            'color': '#F44336' if is_critical else '#4CAF50'  # Red if critical, Green if OK
        })
    
    # Sales Trend
    if 'sales_trend' in insights:
        cards.append({
            'icon': '📈',
            'title': 'Sales Trend',
            'insight': insights['sales_trend'],
            'priority': 'medium',
            'color': '#00BCD4'  # Cyan
        })
    
    # Strategic Recommendation (Always last)
    if 'strategic_recommendation' in insights:
        cards.append({
            'icon': '🎯',
            'title': 'Strategic Priority',
            'insight': insights['strategic_recommendation'],
            'priority': 'critical',
            'color': '#673AB7'  # Deep Purple
        })
    
    # Sort by priority
    priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    cards.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    return cards
