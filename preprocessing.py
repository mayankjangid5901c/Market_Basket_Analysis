"""
Data Preprocessing Module for Market Basket Analysis
Handles loading, cleaning, filtering, and transaction-basket encoding.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Non-merchandise service codes & administrative items to remove from retail basket
EXCLUDE_DESCRIPTIONS = {
    'POSTAGE',
    'DOTCOM POSTAGE',
    'MANUAL',
    'BANK CHARGES',
    'AMAZON FEE',
    'CRUK COMMISSION',
    'SAMPLES',
    'DISCOUNT',
    'PADS TO MATCH ALL CUSHIONS',
    'ADJUST',
    'ADJUSTMENT',
    'CARRIAGE',
    'LOST IN VANS',
    'DAMAGED',
    'DAMAGES',
    'CHECK',
    'FOUND',
    'UNCANELLED'
}

DEFAULT_DATA_PATHS = [
    os.path.join('data', 'online_retail.csv'),
    os.path.join('data', 'Online Retail.xlsx'),
    os.path.join('data', 'sample_retail_data.csv')
]


def find_default_dataset() -> Tuple[Optional[str], bool]:
    """
    Locates the best available dataset file.
    Returns:
        (file_path, is_sample)
    """
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            is_sample = 'sample' in os.path.basename(path).lower()
            return path, is_sample
    return None, False


def load_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads retail transaction dataset from CSV or Excel.
    
    Args:
        file_path: Optional custom path to data file.
        
    Returns:
        Tuple of (DataFrame, metadata_dictionary)
    """
    is_sample = False
    if file_path is None or not os.path.exists(file_path):
        detected_path, is_sample = find_default_dataset()
        if detected_path is None:
            raise FileNotFoundError(
                "No dataset found in 'data/' directory. Please provide online_retail.csv or sample_retail_data.csv"
            )
        file_path = detected_path
    else:
        is_sample = 'sample' in os.path.basename(file_path).lower()

    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.csv':
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='ISO-8859-1')
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Use CSV or Excel.")
    except Exception as e:
        raise RuntimeError(f"Error reading dataset at {file_path}: {str(e)}")

    metadata = {
        'file_name': os.path.basename(file_path),
        'file_path': file_path,
        'raw_rows': len(df),
        'raw_columns': df.columns.tolist(),
        'is_sample': is_sample
    }

    return df, metadata


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw retail transaction records according to project specs:
    1. Removes duplicate records.
    2. Drops rows with missing Description or InvoiceNo.
    3. Handles missing CustomerID.
    4. Removes cancelled orders (InvoiceNo starting with 'C' or 'c').
    5. Removes non-positive quantities (Quantity <= 0) and prices (UnitPrice <= 0).
    6. Strips product descriptions, standardizes uppercase, and removes service items.
    7. Formats InvoiceDate as datetime and creates analytical features.
    
    Returns:
        Tuple of (cleaned_df, cleaning_stats_dict)
    """
    initial_count = len(df)
    stats = {'initial_records': initial_count}

    # Make working copy
    df_clean = df.copy()

    # 1. Remove duplicate records
    duplicates_count = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    stats['duplicates_removed'] = int(duplicates_count)

    # 2. Handle missing InvoiceNo and Description
    missing_desc = df_clean['Description'].isna().sum()
    missing_invoice = df_clean['InvoiceNo'].isna().sum()
    df_clean = df_clean.dropna(subset=['InvoiceNo', 'Description'])
    stats['missing_descriptions_removed'] = int(missing_desc)
    stats['missing_invoices_removed'] = int(missing_invoice)

    # 3. Standardize CustomerID
    if 'CustomerID' in df_clean.columns:
        df_clean['CustomerID'] = df_clean['CustomerID'].fillna(-1).astype(int)

    # 4. Remove cancelled transactions
    df_clean['InvoiceNo'] = df_clean['InvoiceNo'].astype(str).str.strip()
    is_cancelled = df_clean['InvoiceNo'].str.upper().str.startswith('C')
    cancelled_count = is_cancelled.sum()
    df_clean = df_clean[~is_cancelled]
    stats['cancellations_removed'] = int(cancelled_count)

    # 5. Remove negative or zero quantities and unit prices
    if 'Quantity' in df_clean.columns:
        invalid_qty = (df_clean['Quantity'] <= 0).sum()
        df_clean = df_clean[df_clean['Quantity'] > 0]
        stats['invalid_quantities_removed'] = int(invalid_qty)
    else:
        stats['invalid_quantities_removed'] = 0

    if 'UnitPrice' in df_clean.columns:
        invalid_price = (df_clean['UnitPrice'] <= 0).sum()
        df_clean = df_clean[df_clean['UnitPrice'] > 0]
        stats['invalid_prices_removed'] = int(invalid_price)
    else:
        stats['invalid_prices_removed'] = 0

    # 6. Clean Product Descriptions
    df_clean['Description'] = df_clean['Description'].astype(str).str.strip().str.upper()

    # Filter out service charges & administrative non-product codes
    before_service_filter = len(df_clean)
    df_clean = df_clean[~df_clean['Description'].isin(EXCLUDE_DESCRIPTIONS)]
    # Filter descriptions that are too short (<= 2 chars) or pure numbers
    df_clean = df_clean[df_clean['Description'].apply(lambda x: len(x) > 2 and not x.isdigit())]
    stats['service_items_removed'] = int(before_service_filter - len(df_clean))

    # 7. Date parsing and feature engineering
    if 'InvoiceDate' in df_clean.columns:
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')
        df_clean = df_clean.dropna(subset=['InvoiceDate'])
        df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)
        df_clean['MonthName'] = df_clean['InvoiceDate'].dt.strftime('%b %Y')
        df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
        df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour

    if 'Quantity' in df_clean.columns and 'UnitPrice' in df_clean.columns:
        df_clean['TotalAmount'] = (df_clean['Quantity'] * df_clean['UnitPrice']).round(2)

    final_count = len(df_clean)
    stats['final_records'] = final_count
    stats['retained_percentage'] = round((final_count / initial_count * 100), 2) if initial_count > 0 else 0

    return df_clean, stats


def filter_by_country(df: pd.DataFrame, country: str = 'All') -> pd.DataFrame:
    """
    Filters cleaned dataset by Country.
    """
    if country in ['All', 'All Countries', None]:
        return df
    if 'Country' in df.columns:
        return df[df['Country'] == country]
    return df


def create_basket(
    df: pd.DataFrame,
    min_items_per_basket: int = 2,
    max_baskets: Optional[int] = None
) -> pd.DataFrame:
    """
    Converts transaction lines into a one-hot encoded boolean basket matrix.
    Rows = Invoices, Columns = Products (True if purchased in invoice, False otherwise).
    
    Args:
        df: Cleaned dataframe.
        min_items_per_basket: Minimum distinct items required in a basket (default 2).
        max_baskets: Optional maximum number of invoices to sample for fast computation.
        
    Returns:
        Boolean DataFrame indexed by InvoiceNo with Product Descriptions as columns.
    """
    if df.empty:
        return pd.DataFrame()

    # Aggregate quantities by InvoiceNo and Description
    grouped = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    
    # Filter baskets by minimum distinct items
    item_counts_per_basket = (grouped > 0).sum(axis=1)
    filtered = grouped[item_counts_per_basket >= min_items_per_basket]

    # Sample if requested
    if max_baskets is not None and len(filtered) > max_baskets:
        filtered = filtered.sample(n=max_baskets, random_state=42)

    # Convert to boolean matrix (efficient and expected by mlxtend >= 0.23)
    basket_bool = (filtered > 0).astype(bool)
    return basket_bool
