"""
Column Mapper Utility
This module will contain functions for mapping and identifying column types in datasets
"""

def map_columns(df, mapping_dict):
    """
    mapping_dict example:
    {
        "Date": "Order_Date",
        "Store": "Branch",
        "Product": "Item_Name",
        "Quantity": "Units",
        "Selling_Price": "SalePrice",
        "Cost_Price": "Cost"
    }
    """
    return df.rename(columns={v: k for k, v in mapping_dict.items()})