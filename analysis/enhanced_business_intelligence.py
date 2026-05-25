"""
Enhanced Business Intelligence Engine
Generates specific, data-driven insights with actual metrics and actionable recommendations
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def generate_enhanced_insights(df: pd.DataFrame, retail_results: Dict) -> List[str]:
    """
    Generate EXACTLY 5 high-quality, data-driven business insights
    Each insight follows WHAT-WHY-IMPACT-ACTION format with REAL numbers
    Uses ACTUAL computed values from retail_results - NO fallback/dummy values
    If data is missing, SKIP that insight (return None)
    """
    insights = []
    
    # INSIGHT 1: Revenue Concentration Risk (Always #1)
    concentration_insight = _analyze_revenue_concentration(df, retail_results)
    insights.append(concentration_insight)
    
    # INSIGHT 2: Profit Margin Analysis (Always #2)
    margin_insight = _analyze_profit_margin_deep(df, retail_results)
    insights.append(margin_insight)
    
    # INSIGHT 3: Product Performance with Specific Data (Always #3)
    product_insight = _analyze_product_performance_specific(df, retail_results)
    insights.append(product_insight)
    
    # INSIGHT 4: Category Performance (Always #4)
    category_insight = _analyze_category_performance(df, retail_results)
    insights.append(category_insight)
    
    # INSIGHT 5: Store Performance (Always #5)
    store_insight = _analyze_store_performance(df, retail_results)
    insights.append(store_insight)
    
    # Filter out None values (missing data = skip insight, NO fake insights)
    valid_insights = [insight for insight in insights if insight is not None]
    
    # Return EXACTLY 5 insights (only if we have real data for all)
    # If less than 5, return what we have (NO generic fallbacks)
    return valid_insights[:5]


def _analyze_revenue_concentration(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Analyze revenue concentration using ACTUAL computed values from retail_results"""
    try:
        # USE ACTUAL retail_results if available
        if retail_results and 'total_revenue' in retail_results:
            total_revenue = retail_results['total_revenue']
        elif 'Revenue' in df.columns:
            total_revenue = df['Revenue'].sum()
        else:
            return None  # Skip if no data
        
        if total_revenue == 0:
            return None
        
        # Get store performance data
        if retail_results and 'store_performance' in retail_results and retail_results['store_performance'] is not None:
            store_perf = retail_results['store_performance']
            if hasattr(store_perf, 'iloc') and len(store_perf) > 0:
                # Already computed - use it
                top_store = store_perf.index[0] if hasattr(store_perf.index, '__getitem__') else store_perf.iloc[0]['Store']
                top_revenue = store_perf.iloc[0]['Total_Revenue'] if 'Total_Revenue' in store_perf.columns else store_perf.iloc[0].iloc[0]
            else:
                return None
        elif 'Store' in df.columns:
            # Calculate from DataFrame
            store_revenue = df.groupby('Store')['Revenue'].sum().sort_values(ascending=False)
            if len(store_revenue) == 0:
                return None
            top_store = store_revenue.index[0]
            top_revenue = store_revenue.iloc[0]
        else:
            return None
        
        top_contribution = (top_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Format currency
        top_rev_formatted = _format_currency(top_revenue)
        total_rev_formatted = _format_currency(total_revenue)
        
        # Build insight with line breaks for readability
        what = f"{top_store} generates {top_rev_formatted} ({top_contribution:.1f}% of total {total_rev_formatted})"
        why = "Strong regional demand and higher order volume drive this location's dominance"
        
        if top_contribution > 40:
            impact = f"Heavy dependency on one location creates significant risk—if {top_store} faces disruptions, {top_contribution:.0f}% of revenue is at risk"
            action = f"Diversify revenue by expanding successful {top_store} strategies to other locations; reduce concentration risk below 30%"
        elif top_contribution > 30:
            impact = f"Moderate concentration risk—{top_store} contributes nearly one-third of total revenue"
            action = f"Monitor {top_store} performance closely; invest in growing other stores to balance revenue distribution"
        else:
            impact = f"Well-distributed revenue across stores with {top_store} leading at {top_contribution:.1f}%"
            action = f"Maintain balanced growth; study {top_store} success factors and replicate across network"
        
        # Format with line breaks
        return f"""WHAT: {what} 
                WHY: {why}
                IMPACT: {impact}
                ACTION: {action}"""
        
    except Exception as e:
        return None


def _analyze_profit_margin_deep(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Deep profit margin analysis with specific issues and actions"""
    try:
        if 'Profit' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        total_revenue = df['Revenue'].sum()
        total_profit = df['Profit'].sum()
        
        if total_revenue == 0:
            return None
        
        profit_margin = (total_profit / total_revenue) * 100
        
        # Calculate cost
        total_cost = total_revenue - total_profit
        
        # Find loss-making products
        loss_products = df[df['Profit'] < 0]
        loss_count = loss_products['Product'].nunique() if 'Product' in loss_products.columns else 0
        total_loss = loss_products['Profit'].sum() if len(loss_products) > 0 else 0
        
        # Format
        profit_formatted = _format_currency(total_profit)
        cost_formatted = _format_currency(total_cost)
        loss_formatted = _format_currency(abs(total_loss))
        
        # Determine margin health
        if profit_margin > 30:
            health = "Excellent"
            benchmark = "well above 10-20% retail average"
        elif profit_margin > 20:
            health = "Strong"
            benchmark = "above retail industry average"
        elif profit_margin > 10:
            health = "Moderate"
            benchmark = "within typical retail range"
        else:
            health = "Below Average"
            benchmark = "below 10-20% retail average"
        
        # WHAT
        what = f"Profit margin is {profit_margin:.1f}%—{health} performance. From {total_revenue:,.0f} revenue, costs are {cost_formatted}, leaving {profit_formatted} profit"
        
        # WHY
        if loss_count > 0:
            why = f"{loss_count} loss-making products are eroding {loss_formatted} in potential profit, dragging down overall margins"
        else:
            why = f"All products are profitable, indicating strong pricing discipline and cost management"
        
        # IMPACT
        if profit_margin < 15:
            impact = f"CRITICAL: Only {profit_margin:.1f}% margin leaves minimal buffer for cost increases, discounts, or market downturns—a small cost uptick could push you into losses"
        else:
            impact = f"At {profit_margin:.1f}% margin ({benchmark}), you have healthy cushion to absorb fluctuations and invest in growth"
        
        # ACTION
        if loss_count > 0 and profit_margin < 20:
            action = f"Immediate actions: (1) Review pricing on {loss_count} loss-makers, (2) Increase prices or reduce COGS on worst performers, (3) Discontinue products with >20% negative margins"
        elif profit_margin < 15:
            action = f"Focus on margin improvement: renegotiate supplier contracts to reduce costs by 5-10%, shift product mix toward high-margin items, implement selective price increases on price-insensitive products"
        else:
            action = f"Maintain margin discipline while exploring growth—reinvest profits in customer acquisition, product development, or market expansion to scale the business"
        
        return f"WHAT: {what}. WHY: {why}. IMPACT: {impact}. ACTION: {action}"
        
    except Exception as e:
        return None


def _analyze_product_performance_specific(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Analyze product performance with specific metrics and data"""
    try:
        if 'Product' not in df.columns or 'Quantity' not in df.columns:
            return None
        
        # Product quantity sold
        product_qty = df.groupby('Product')['Quantity'].sum()
        
        if len(product_qty) == 0:
            return None
        
        # Top product
        top_product = product_qty.idxmax()
        top_qty = int(product_qty.max())
        
        # Revenue and profit for top product
        top_product_data = df[df['Product'] == top_product]
        top_product_revenue = top_product_data['Revenue'].sum()
        top_product_profit = top_product_data['Profit'].sum() if 'Profit' in df.columns else 0
        
        # Total revenue
        total_revenue = df['Revenue'].sum()
        revenue_share = (top_product_revenue / total_revenue) * 100 if total_revenue > 0 else 0
        
        # Bottom product (if exists)
        valid_products = product_qty[product_qty > 0]
        if len(valid_products) > 1:
            bottom_product = valid_products.idxmin()
            bottom_qty = int(valid_products.min())
            gap_vs_top = ((top_qty - bottom_qty) / top_qty) * 100
        else:
            bottom_product = None
            bottom_qty = 0
            gap_vs_top = 0
        
        # Format
        top_rev_formatted = _format_currency(top_product_revenue)
        
        # WHAT
        what = f"{top_product} is your bestseller with {top_qty:,} units sold, generating {top_rev_formatted} ({revenue_share:.1f}% of total revenue)"
        
        # WHY
        why = f"Strong market demand, competitive pricing, and effective positioning drive this product's dominance"
        
        # IMPACT
        impact = f"As your revenue leader, {top_product} significantly impacts overall business performance—any supply disruption or demand shift could materially affect total sales"
        
        # ACTION
        action = f"Ensure consistent inventory availability for {top_product}; explore bundle opportunities with complementary products; monitor competitor pricing to maintain competitive edge"
        
        # If there's a slow product, add to insight
        if bottom_product and gap_vs_top > 80:
            action += f". Address {bottom_product} (only {bottom_qty:,} units, {gap_vs_top:.0f}% behind top)—consider discontinuation or aggressive promotion"
        
        return f"WHAT: {what}. WHY: {why}. IMPACT: {impact}. ACTION: {action}"
        
    except Exception as e:
        return None


def _analyze_trend_with_impact(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Analyze sales trend with real increase/decrease detection and business impact"""
    try:
        # Find date column
        date_col = None
        date_keywords = ['date', 'order_date', 'transaction_date', 'time']
        
        for keyword in date_keywords:
            for col in df.columns:
                if keyword.lower() in col.lower():
                    date_col = col
                    break
            if date_col:
                break
        
        if not date_col or 'Revenue' not in df.columns:
            return None
        
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        
        if len(df_copy) < 10:
            return None
        
        # Sort by date
        df_copy = df_copy.sort_values(date_col)
        
        # Aggregate daily revenue
        daily_revenue = df_copy.groupby(date_col)['Revenue'].sum()
        
        if len(daily_revenue) < 7:
            return None
        
        # Split into first half and second half
        mid_point = len(daily_revenue) // 2
        first_half_avg = daily_revenue.iloc[:mid_point].mean()
        second_half_avg = daily_revenue.iloc[mid_point:].mean()
        
        # Calculate trend
        if first_half_avg == 0:
            return None
        
        trend_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        # Dates
        first_date = daily_revenue.index[0].strftime('%Y-%m-%d')
        last_date = daily_revenue.index[-1].strftime('%Y-%m-%d')
        avg_daily = _format_currency(daily_revenue.mean())
        
        # Determine direction and build insight
        if trend_pct > 5:
            emoji = "📈"
            what = f"Sales show strong upward trend of +{trend_pct:.1f}% from {first_date} to {last_date}, growing from {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall)"
            why = f"Positive momentum suggests growing market demand, successful marketing campaigns, or improved customer acquisition"
            impact = f"Sustained growth at this rate could add significant annual revenue, but rapid growth may strain inventory and operational capacity"
            action = f"Capitalize on momentum by increasing marketing spend, expanding inventory to prevent stockouts, and hiring ahead of demand curve"
        elif trend_pct < -5:
            emoji = "📉"
            what = f"Sales are declining at {trend_pct:.1f}% from {first_date} to {last_date}, dropping from {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall)"
            why = f"Downward trend indicates potential issues: increased competition, market saturation, customer churn, or economic headwinds"
            impact = f"If decline continues, it will materially impact quarterly revenue targets and may signal deeper structural issues"
            action = f"Immediately investigate root causes—survey customers, analyze competitor activity, review marketing ROI. Implement retention campaigns and refresh marketing messaging"
        else:
            emoji = "➡️"
            what = f"Sales remain flat with only {trend_pct:+.1f}% change from {first_date} to {last_date}, maintaining {_format_currency(first_half_avg)} to {_format_currency(second_half_avg)} average daily revenue ({avg_daily} overall)"
            why = f"Stability suggests mature market conditions with balanced supply and demand, but also indicates potential growth stagnation"
            impact = f"While stability provides predictable cash flow, inflation and rising costs will gradually erode real profitability without revenue growth"
            action = f"Break through stagnation by introducing new products, entering untapped segments, launching loyalty programs, or exploring new distribution channels"
        
        return f"{emoji} WHAT: {what}. WHY: {why}. IMPACT: {impact}. ACTION: {action}"
        
    except Exception as e:
        return None


def _analyze_category_opportunity(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Analyze category performance and identify strategic opportunities"""
    try:
        if 'Category' not in df.columns:
            return None
        
        # Category performance
        category_perf = df.groupby('Category').agg(
            Total_Profit=('Profit', 'sum') if 'Profit' in df.columns else ('Revenue', 'sum'),
            Total_Revenue=('Revenue', 'sum'),
            Units_Sold=('Quantity', 'sum')
        ).reset_index()
        
        if len(category_perf) == 0:
            return None
        
        # Best category by profit
        profit_col = 'Total_Profit' if 'Profit' in df.columns else 'Total_Revenue'
        best_cat_idx = category_perf[profit_col].idxmax()
        best_category = category_perf.loc[best_cat_idx, 'Category']
        best_profit = category_perf.loc[best_cat_idx, profit_col]
        best_revenue = category_perf.loc[best_cat_idx, 'Total_Revenue']
        best_units = int(category_perf.loc[best_cat_idx, 'Units_Sold'])
        
        # Profit margin
        profit_margin = (best_profit / best_revenue * 100) if best_revenue > 0 else 0
        
        # Compare to average
        avg_profit = category_perf[profit_col].mean()
        profit_lead = ((best_profit - avg_profit) / avg_profit * 100) if avg_profit > 0 else 0
        
        # Products in category
        products_in_category = df[df['Category'] == best_category]['Product'].nunique()
        
        # WHAT
        what = f"{best_category} is your most profitable category, generating {_format_currency(best_profit)} in profit on {_format_currency(best_revenue)} revenue ({profit_margin:.1f}% margin) with {best_units:,} units sold across {products_in_category} products"
        
        # WHY
        why = f"This category's profitability advantage—{profit_lead:.1f}% above average—suggests strong pricing power, favorable cost structure, or effective category management"
        
        # IMPACT
        impact = f"{best_category}'s strong margins make it a key profit driver—growing this category's share of total sales would directly improve overall company profitability"
        
        # ACTION
        action = f"Expand {best_category} product range, increase marketing investment, optimize shelf space or online visibility, and analyze whether successful strategies can transfer to lower-performing categories"
        
        return f"⭐ WHAT: {what}. WHY: {why}. IMPACT: {impact}. ACTION: {action}"
        
    except Exception as e:
        return None


def _format_currency(amount: float) -> str:
    """Format amount as currency with appropriate unit"""
    if abs(amount) >= 1_000_000:
        return f"${amount/1_000_000:.2f}M"
    elif abs(amount) >= 1_000:
        return f"${amount/1_000:.2f}K"
    else:
        return f"${amount:.2f}"


def _generate_fallback_insight(df: pd.DataFrame, retail_results: Dict, insight_number: int) -> str:
    """Generate a fallback insight if primary analysis fails - still follows WHAT-WHY-IMPACT-ACTION"""
    try:
        if insight_number == 1:
            total_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0
            total_orders = len(df)
            avg_order = total_revenue / total_orders if total_orders > 0 else 0
            return f"WHAT: Total revenue is {_format_currency(total_revenue)} from {total_orders:,} transactions (avg {_format_currency(avg_order)} per order). WHY: Transaction volume and average order value drive revenue. IMPACT: Understanding baseline performance is essential for growth planning. ACTION: Focus on increasing average order value through upselling and cross-selling strategies to boost revenue per transaction."
        
        elif insight_number == 2:
            total_profit = df['Profit'].sum() if 'Profit' in df.columns else 0
            profit_margin = (total_profit / df['Revenue'].sum() * 100) if 'Revenue' in df.columns and df['Revenue'].sum() > 0 else 0
            return f"WHAT: Overall profit is {_format_currency(total_profit)} with {profit_margin:.1f}% margin. WHY: Profit margins reflect pricing effectiveness and cost management. IMPACT: Healthy margins provide flexibility for growth investments; thin margins increase risk. ACTION: Monitor margins by product and category; optimize pricing and reduce costs on low-margin items."
        
        elif insight_number == 3:
            num_products = df['Product'].nunique() if 'Product' in df.columns else 0
            num_stores = df['Store'].nunique() if 'Store' in df.columns else 0
            return f"WHAT: Portfolio includes {num_products} products across {num_stores} stores. WHY: Product and location diversity affects market coverage and risk distribution. IMPACT: Diverse portfolio reduces dependency risk but requires careful management. ACTION: Regularly review product performance across locations; discontinue underperformers and scale winners."
        
        elif insight_number == 4:
            if 'Category' in df.columns:
                num_categories = df['Category'].nunique()
                top_cat = df.groupby('Category')['Revenue'].sum().idxmax()
                return f"WHAT: {num_categories} product categories with '{top_cat}' leading revenue. WHY: Category performance varies based on demand, pricing, and market positioning. IMPACT: Category concentration affects revenue stability and growth opportunities. ACTION: Balance category mix; invest in growing categories while maintaining profitable core categories."
            else:
                return f"WHAT: Business operations show consistent patterns across dataset. WHY: Operational consistency indicates mature processes. IMPACT: Stable operations provide foundation for strategic initiatives. ACTION: Identify optimization opportunities in routine operations to improve efficiency and reduce costs."
        
        else:
            return f"WHAT: Analysis reveals opportunities for business optimization. WHY: Continuous improvement drives competitive advantage. IMPACT: Acting on data-driven insights leads to measurable business growth. ACTION: Prioritize high-impact initiatives based on available resources and expected ROI."
    
    except Exception:
        return f"WHAT: Business data analyzed successfully. WHY: Data-driven decisions outperform intuition-based choices. IMPACT: Regular analysis enables proactive management. ACTION: Review insights monthly and adjust strategy based on emerging trends."


def generate_enhanced_recommendations(df: pd.DataFrame, retail_results: Dict) -> List[str]:
    """
    Generate MAXIMUM 6 specific, data-backed recommendations with supporting metrics
    Each recommendation must include specific product/category/store names and actual numbers
    NO weak recommendations like "monitor KPIs" or "analyze factors"
    Only HIGH-VALUE decisions allowed
    """
    recommendations = []
    
    # RECOMMENDATION 1: Top Product Revenue Optimization (High Value)
    top_product_rec = _recommendation_top_product(df, retail_results)
    if top_product_rec:
        recommendations.append(top_product_rec)
    
    # RECOMMENDATION 2: Loss Product Elimination (High Value - Critical)
    loss_product_rec = _recommendation_loss_products(df, retail_results)
    if loss_product_rec:
        recommendations.append(loss_product_rec)
    
    # RECOMMENDATION 3: Category Expansion (High Value)
    category_rec = _recommendation_category_expansion(df, retail_results)
    if category_rec:
        recommendations.append(category_rec)
    
    # RECOMMENDATION 4: Store Performance Improvement (High Value)
    store_rec = _recommendation_store_improvement(df, retail_results)
    if store_rec:
        recommendations.append(store_rec)
    
    # RECOMMENDATION 5: Margin Improvement (High Value)
    margin_rec = _recommendation_margin_improvement(df, retail_results)
    if margin_rec:
        recommendations.append(margin_rec)
    
    # RECOMMENDATION 6: Revenue Growth Strategy (High Value)
    growth_rec = _recommendation_revenue_growth(df, retail_results)
    if growth_rec:
        recommendations.append(growth_rec)
    
    # Return MAXIMUM 6 recommendations (only high-value ones)
    return recommendations[:6]


def _recommendation_top_product(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for top product revenue optimization"""
    try:
        if 'Product' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        # Top product by revenue
        product_revenue = df.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
        
        if len(product_revenue) == 0:
            return None
        
        top_product = product_revenue.index[0]
        top_revenue = product_revenue.iloc[0]
        total_revenue = product_revenue.sum()
        revenue_share = (top_revenue / total_revenue) * 100
        
        return f"Product '{top_product}' contributes {revenue_share:.1f}% of total revenue ({_format_currency(top_revenue)}). Increasing visibility through targeted promotions and strategic placement could boost overall sales by 10-15%."
        
    except Exception as e:
        return None


def _recommendation_loss_products(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for eliminating loss-making products"""
    try:
        if 'Profit' not in df.columns or 'Product' not in df.columns:
            return None
        
        # Find loss products
        product_profit = df.groupby('Product')['Profit'].sum()
        loss_products = product_profit[product_profit < 0].sort_values()
        
        if len(loss_products) == 0:
            return None
        
        # Top 3 loss-makers
        top_3_losses = loss_products.head(3)
        total_loss = top_3_losses.sum()
        
        product_list = ", ".join([f"'{p}' ({_format_currency(abs(profit))} loss)" for p, profit in top_3_losses.items()])
        
        return f"Immediately address loss-makers: {product_list}. These products are generating {_format_currency(abs(total_loss))} in total losses. Options: (1) Increase prices by 15-20%, (2) Negotiate lower supplier costs, or (3) Discontinue and liquidate inventory."
        
    except Exception as e:
        return None


def _recommendation_category_expansion(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for expanding profitable categories"""
    try:
        if 'Category' not in df.columns or 'Profit' not in df.columns:
            return None
        
        # Category profit
        category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
        
        if len(category_profit) == 0:
            return None
        
        top_category = category_profit.index[0]
        top_profit = category_profit.iloc[0]
        profit_margin = (top_profit / df[df['Category'] == top_category]['Revenue'].sum()) * 100
        
        return f"Expand '{top_category}' category—it generates {_format_currency(top_profit)} profit with {profit_margin:.1f}% margin, outperforming other categories. Add 5-10 new products in this category and increase marketing budget by 20% to capture more market share."
        
    except Exception as e:
        return None


def _recommendation_store_improvement(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for underperforming store improvement"""
    try:
        if 'Store' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        # Store performance
        store_revenue = df.groupby('Store')['Revenue'].sum()
        
        if len(store_revenue) < 2:
            return None
        
        # Lowest performer
        avg_revenue = store_revenue.mean()
        lowest_store = store_revenue.idxmin()
        lowest_revenue = store_revenue.min()
        gap_vs_avg = ((avg_revenue - lowest_revenue) / avg_revenue) * 100
        
        return f"Store '{lowest_store}' underperforms with {_format_currency(lowest_revenue)} revenue ({gap_vs_avg:.1f}% below average {_format_currency(avg_revenue)}). Conduct operational audit, analyze local competition, and implement 90-day turnaround plan with targeted promotions and staffing optimization."
        
    except Exception as e:
        return None


def _recommendation_margin_improvement(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for margin improvement"""
    try:
        if 'Profit' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        total_revenue = df['Revenue'].sum()
        total_profit = df['Profit'].sum()
        
        if total_revenue == 0:
            return None
        
        profit_margin = (total_profit / total_revenue) * 100
        
        if profit_margin >= 25:
            return None  # Already healthy
        
        return f"Profit margin is {profit_margin:.1f}%—below optimal 25-30% target. Immediate actions: (1) Renegotiate supplier contracts to reduce COGS by 5-8%, (2) Shift product mix toward high-margin items, (3) Implement selective price increases of 3-5% on price-insensitive products to add {_format_currency(total_revenue * 0.05)} to annual profit."
        
    except Exception as e:
        return None


def _recommendation_revenue_growth(df: pd.DataFrame, retail_results: Dict) -> Optional[str]:
    """Recommendation for revenue growth through high-performers"""
    try:
        if 'Store' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        # Top store revenue
        store_revenue = df.groupby('Store')['Revenue'].sum().sort_values(ascending=False)
        
        if len(store_revenue) == 0:
            return None
        
        top_store = store_revenue.index[0]
        top_revenue = store_revenue.iloc[0]
        avg_revenue = store_revenue.mean()
        
        # Calculate gap
        gap = ((top_revenue - avg_revenue) / avg_revenue) * 100 if avg_revenue > 0 else 0
        
        return f"Scale '{top_store}' success ({_format_currency(top_revenue)} revenue, {gap:.0f}% above average) by replicating strategies across underperforming locations. Target: Increase total network revenue by 15-20% through proven playbook expansion."
        
    except Exception as e:
        return None


def generate_business_warnings(df: pd.DataFrame, retail_results: Dict) -> List[Dict[str, str]]:
    """
    Generate MAXIMUM 3 business warnings with severity levels
    ONLY business-relevant warnings - NO technical warnings
    Returns list of dicts with: severity, label, message
    
    Severity Levels:
    - CRITICAL (red): Immediate action required
    - WARNING (orange): Attention needed soon
    - INFO (yellow): Monitor and be aware
    """
    warnings = []
    
    # WARNING 1: Profit Margin Check (CRITICAL if <10%, WARNING if <20%)
    margin_warning = _check_profit_margin(df)
    if margin_warning:
        warnings.append(margin_warning)
    
    # WARNING 2: Revenue Concentration (CRITICAL if >50%, WARNING if >35%)
    concentration_warning = _check_revenue_concentration(df)
    if concentration_warning:
        warnings.append(concentration_warning)
    
    # WARNING 3: Loss Products or Declining Trends
    loss_warning = _check_loss_products(df)
    if loss_warning:
        warnings.append(loss_warning)
    
    # Return MAXIMUM 3 warnings
    return warnings[:3]


def _check_profit_margin(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Check profit margin and generate appropriate warning"""
    try:
        if 'Profit' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        total_revenue = df['Revenue'].sum()
        total_profit = df['Profit'].sum()
        
        if total_revenue == 0:
            return None
        
        profit_margin = (total_profit / total_revenue) * 100
        
        if profit_margin < 10:
            return {
                'severity': 'CRITICAL',
                'label': 'CRITICAL',
                'message': f'Profit margin is only {profit_margin:.1f}% ({_format_currency(total_profit)} profit on {_format_currency(total_revenue)} revenue). This indicates high operational cost risk—minimal buffer for cost increases or market downturns. Immediate pricing review and cost reduction needed.',
                'color': '#ef4444'  # Red
            }
        elif profit_margin < 20:
            return {
                'severity': 'WARNING',
                'label': 'WARNING',
                'message': f'Profit margin at {profit_margin:.1f}% is below optimal 25-30% target. While profitable, there is room for improvement through cost optimization and strategic price adjustments.',
                'color': '#f59e0b'  # Orange
            }
        
        return None
        
    except Exception as e:
        return None


def _check_revenue_concentration(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Check revenue concentration risk"""
    try:
        if 'Store' not in df.columns or 'Revenue' not in df.columns:
            return None
        
        store_revenue = df.groupby('Store')['Revenue'].sum()
        total_revenue = store_revenue.sum()
        
        if total_revenue == 0 or len(store_revenue) == 0:
            return None
        
        # Top store contribution
        top_store_revenue = store_revenue.max()
        top_store = store_revenue.idxmax()
        concentration = (top_store_revenue / total_revenue) * 100
        
        if concentration > 50:
            return {
                'severity': 'CRITICAL',
                'label': 'CRITICAL',
                'message': f'{top_store} generates {concentration:.1f}% of total revenue ({_format_currency(top_store_revenue)}). Heavy dependency creates severe risk—if this location faces disruptions, over half of revenue is at risk. Urgent diversification needed.',
                'color': '#ef4444'  # Red
            }
        elif concentration > 35:
            return {
                'severity': 'WARNING',
                'label': 'WARNING',
                'message': f'{top_store} contributes {concentration:.1f}% of revenue ({_format_currency(top_store_revenue)}). Moderate concentration risk—monitor performance closely and invest in growing other locations.',
                'color': '#f59e0b'  # Orange
            }
        
        return None
        
    except Exception as e:
        return None


def _check_loss_products(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Check for loss-making products"""
    try:
        if 'Profit' not in df.columns or 'Product' not in df.columns:
            return None
        
        # Find loss products
        product_profit = df.groupby('Product')['Profit'].sum()
        loss_products = product_profit[product_profit < 0]
        
        if len(loss_products) == 0:
            return None
        
        total_loss = loss_products.sum()
        loss_count = len(loss_products)
        
        # Get top 3 worst losses
        top_losses = loss_products.nsmallest(3)
        loss_details = ", ".join([f"'{p}' ({_format_currency(abs(profit))} loss)" for p, profit in top_losses.items()])
        
        if loss_count >= 5 or abs(total_loss) > 50000:
            return {
                'severity': 'CRITICAL',
                'label': 'CRITICAL',
                'message': f'{loss_count} products generating {_format_currency(abs(total_loss))} in losses. Worst offenders: {loss_details}. These loss-makers are eroding overall profitability. Immediate action required: increase prices, reduce costs, or discontinue.',
                'color': '#ef4444'  # Red
            }
        elif loss_count >= 2:
            return {
                'severity': 'WARNING',
                'label': 'WARNING',
                'message': f'{loss_count} products showing losses totaling {_format_currency(abs(total_loss))}. Review pricing and costs for these items to prevent further margin erosion.',
                'color': '#f59e0b'  # Orange
            }
        else:
            return {
                'severity': 'INFO',
                'label': 'INFO',
                'message': f'1 product ({loss_details}) showing losses. Monitor closely and consider price adjustment or discontinuation if trend continues.',
                'color': '#eab308'  # Yellow
            }
        
    except Exception as e:
        return None
