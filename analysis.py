"""
Analysis and Modeling Module for Market Basket Analysis
Includes Exploratory Data Analysis (EDA) visualizations and Apriori Association Rule Mining using Mlxtend.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional, List
from mlxtend.frequent_patterns import apriori, association_rules

# Set aesthetic styling for Matplotlib and Seaborn
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette('crest')


# ---------------------------------------------------------
# 1. KPI & Summary Metrics
# ---------------------------------------------------------
def get_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes top-level retail KPIs from transaction dataset.
    """
    if df.empty:
        return {
            'total_transactions': 0,
            'total_revenue': 0.0,
            'total_products': 0,
            'total_customers': 0,
            'avg_basket_value': 0.0,
            'avg_basket_size': 0.0,
            'countries_count': 0
        }

    total_transactions = df['InvoiceNo'].nunique()
    total_revenue = df['TotalAmount'].sum() if 'TotalAmount' in df.columns else 0.0
    total_products = df['Description'].nunique()
    
    # Active registered customers (excluding -1)
    if 'CustomerID' in df.columns:
        valid_cust = df[df['CustomerID'] > 0]
        total_customers = valid_cust['CustomerID'].nunique()
    else:
        total_customers = 0

    avg_basket_value = total_revenue / total_transactions if total_transactions > 0 else 0.0
    
    # Average distinct products per invoice
    items_per_invoice = df.groupby('InvoiceNo')['Description'].nunique()
    avg_basket_size = items_per_invoice.mean() if len(items_per_invoice) > 0 else 0.0
    countries_count = df['Country'].nunique() if 'Country' in df.columns else 0

    return {
        'total_transactions': int(total_transactions),
        'total_revenue': float(total_revenue),
        'total_products': int(total_products),
        'total_customers': int(total_customers),
        'avg_basket_value': float(avg_basket_value),
        'avg_basket_size': float(avg_basket_size),
        'countries_count': int(countries_count)
    }


# ---------------------------------------------------------
# 2. Exploratory Data Analysis (Visualizations)
# ---------------------------------------------------------
def plot_top_frequent_products(df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots top N products by purchase frequency (transaction appearances).
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    top_items = df.groupby('Description')['InvoiceNo'].nunique().nlargest(top_n).sort_values(ascending=True)

    colors = sns.color_palette("Blues_r", n_colors=top_n)
    bars = ax.barh(top_items.index, top_items.values, color=colors, edgecolor='none', height=0.65)
    
    ax.set_title(f'Top {top_n} Most Frequently Purchased Products', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Transactions', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Product Description', fontsize=11, fontweight='semibold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Add numeric labels to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (top_items.values.max() * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{int(width):,}', va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_top_products_by_quantity(df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots top N products by total units sold.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    top_qty = df.groupby('Description')['Quantity'].sum().nlargest(top_n).sort_values(ascending=True)

    colors = sns.color_palette("Purples_r", n_colors=top_n)
    bars = ax.barh(top_qty.index, top_qty.values, color=colors, edgecolor='none', height=0.65)

    ax.set_title(f'Top {top_n} Products by Sales Quantity (Units Sold)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Total Units Sold', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Product Description', fontsize=11, fontweight='semibold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + (top_qty.values.max() * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{int(width):,}', va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_monthly_sales_trend(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots monthly revenue and transaction volume trends over time.
    """
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    monthly = df.groupby('YearMonth').agg(
        Revenue=('TotalAmount', 'sum'),
        Transactions=('InvoiceNo', 'nunique')
    ).reset_index()

    # Drop potential incomplete boundary months if needed
    monthly = monthly.sort_values('YearMonth')

    color1 = '#3b82f6'
    color2 = '#10b981'

    bars = ax1.bar(monthly['YearMonth'], monthly['Revenue'] / 1000, color=color1, alpha=0.75, width=0.5, label='Revenue (£ in Thousands)')
    ax1.set_xlabel('Month / Year', fontsize=11, fontweight='semibold', labelpad=10)
    ax1.set_ylabel('Revenue (£ Thousands)', color=color1, fontsize=11, fontweight='semibold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Overlay line for transactions count
    ax2 = ax1.twinx()
    line = ax2.plot(monthly['YearMonth'], monthly['Transactions'], color=color2, marker='o', linewidth=2.5, label='Transaction Count')
    ax2.set_ylabel('Number of Transactions', color=color2, fontsize=11, fontweight='semibold')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)

    plt.title('Monthly Sales Revenue and Transaction Volume Trends', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_basket_size_distribution(df: pd.DataFrame, max_size: int = 25, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots the frequency distribution of basket sizes (number of distinct items per invoice).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    basket_sizes = df.groupby('InvoiceNo')['Description'].nunique()
    clipped_sizes = basket_sizes[basket_sizes <= max_size]

    sns.histplot(clipped_sizes, bins=range(1, max_size + 2), discrete=True, color='#6366f1', ax=ax, edgecolor='white', alpha=0.85)

    ax.set_title('Customer Basket Size Distribution (Distinct Items per Order)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Distinct Items in Basket', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Frequency (Invoices)', fontsize=11, fontweight='semibold')
    ax.set_xticks(range(1, max_size + 1, 2))
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Median and Mean lines
    mean_val = basket_sizes.mean()
    median_val = basket_sizes.median()
    ax.axvline(mean_val, color='#ef4444', linestyle='--', linewidth=1.8, label=f'Mean: {mean_val:.1f} items')
    ax.axvline(median_val, color='#f59e0b', linestyle=':', linewidth=2.0, label=f'Median: {median_val:.0f} items')
    ax.legend(frameon=True, facecolor='white', loc='upper right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_orders_by_day(df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots order distribution across days of the week.
    """
    fig, ax = plt.subplots(figsize=(9, 4.8))
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_counts = df.groupby('DayOfWeek')['InvoiceNo'].nunique().reindex(days_order).dropna()
    colors = sns.color_palette("crest", n_colors=len(day_counts))

    bars = ax.bar(day_counts.index, day_counts.values, color=colors, width=0.55)
    ax.set_title('Transaction Volume by Day of Week', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Day of Week', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Number of Transactions', fontsize=11, fontweight='semibold')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + (day_counts.values.max() * 0.015),
                f'{int(h):,}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_top_countries(df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots the top countries by number of orders.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    country_counts = df.groupby('Country')['InvoiceNo'].nunique().nlargest(top_n).sort_values(ascending=True)

    colors = sns.color_palette("mako", n_colors=top_n)
    bars = ax.barh(country_counts.index, country_counts.values, color=colors, height=0.6)

    ax.set_title(f'Top {top_n} Countries by Transaction Count', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Invoices', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Country', fontsize=11, fontweight='semibold')
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.1, bar.get_y() + bar.get_height() / 2,
                f'{int(w):,}', va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ---------------------------------------------------------
# 3. Market Basket Analysis: Apriori & Association Rules
# ---------------------------------------------------------
def run_apriori(
    basket_df: pd.DataFrame,
    min_support: float = 0.03,
    max_len: Optional[int] = None
) -> pd.DataFrame:
    """
    Executes Apriori frequent itemset mining on a boolean basket matrix.
    
    Args:
        basket_df: Boolean DataFrame of transactions.
        min_support: Minimum support threshold (0.0 to 1.0).
        max_len: Maximum length of itemsets.
        
    Returns:
        DataFrame of frequent itemsets with support and itemset length.
    """
    if basket_df.empty:
        return pd.DataFrame(columns=['support', 'itemsets', 'length', 'itemsets_display'])

    # Ensure basket is boolean type
    basket_bool = basket_df.astype(bool)

    # Run Apriori
    itemsets = apriori(basket_bool, min_support=min_support, use_colnames=True, max_len=max_len)
    
    if itemsets.empty:
        return pd.DataFrame(columns=['support', 'itemsets', 'length', 'itemsets_display'])

    # Add itemset length and display string
    itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
    itemsets['itemsets_display'] = itemsets['itemsets'].apply(lambda x: " + ".join(sorted(list(x))))
    
    # Sort descending by support
    itemsets = itemsets.sort_values(by=['support', 'length'], ascending=[False, False]).reset_index(drop=True)
    return itemsets


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    metric: str = "lift",
    min_threshold: float = 1.0,
    min_confidence: float = 0.1
) -> pd.DataFrame:
    """
    Generates association rules from mined frequent itemsets.
    
    Args:
        frequent_itemsets: Output DataFrame from run_apriori.
        metric: Metric to evaluate ("lift", "confidence", "support").
        min_threshold: Minimum threshold value for chosen metric.
        min_confidence: Additional minimum confidence filter.
        
    Returns:
        DataFrame of association rules with formatted string descriptions.
    """
    cols = ['antecedents', 'consequents', 'antecedent support', 'consequent support',
            'support', 'confidence', 'lift', 'leverage', 'conviction',
            'antecedents_str', 'consequents_str', 'rule_str']

    if frequent_itemsets.empty or len(frequent_itemsets) < 2:
        return pd.DataFrame(columns=cols)

    try:
        rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
    except Exception:
        return pd.DataFrame(columns=cols)

    if rules.empty:
        return pd.DataFrame(columns=cols)

    # Apply confidence filter
    rules = rules[rules['confidence'] >= min_confidence]

    if rules.empty:
        return pd.DataFrame(columns=cols)

    # Format antecedents and consequents as readable strings
    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ", ".join(sorted(list(x))))
    rules['consequents_str'] = rules['consequents'].apply(lambda x: ", ".join(sorted(list(x))))
    rules['rule_str'] = rules['antecedents_str'] + "  ===>  " + rules['consequents_str']

    # Round metrics for presentation
    rules['support_pct'] = (rules['support'] * 100).round(2)
    rules['confidence_pct'] = (rules['confidence'] * 100).round(2)
    rules['lift_val'] = rules['lift'].round(2)

    # Sort descending by lift
    rules = rules.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False]).reset_index(drop=True)
    return rules


# ---------------------------------------------------------
# 4. Association Rules Visualizations
# ---------------------------------------------------------
def plot_frequent_itemsets(frequent_itemsets: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plots top N frequent itemsets by support.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    top_items = frequent_itemsets.nlargest(top_n, 'support').sort_values(by='support', ascending=True)

    colors = sns.color_palette("Spectral", n_colors=len(top_items))
    bars = ax.barh(top_items['itemsets_display'], top_items['support'] * 100, color=colors, height=0.6)

    ax.set_title(f'Top {top_n} Most Frequent Itemsets (by Support %)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Support (% of All Transactions)', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Itemsets', fontsize=11, fontweight='semibold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{w:.2f}%', va='center', ha='left', fontsize=10, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_rules_scatter(rules_df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """
    Scatter plot of Support vs Confidence with point size and color mapped to Lift.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if rules_df.empty:
        ax.text(0.5, 0.5, 'No association rules found with current thresholds',
                ha='center', va='center', fontsize=12, color='gray')
        return fig

    scatter = ax.scatter(
        rules_df['support'] * 100,
        rules_df['confidence'] * 100,
        c=rules_df['lift'],
        cmap='viridis',
        s=rules_df['lift'] * 25,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.6
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Lift Ratio', fontsize=11, fontweight='semibold')

    ax.set_title('Association Rules: Support vs. Confidence (Bubble Size & Color = Lift)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Support (%)', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Confidence (%)', fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_top_rules_lift(rules_df: pd.DataFrame, top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Horizontal bar chart of top N rules ranked by Lift.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    if rules_df.empty:
        ax.text(0.5, 0.5, 'No rules available', ha='center', va='center', fontsize=12, color='gray')
        return fig

    top_rules = rules_df.nlargest(top_n, 'lift').sort_values(by='lift', ascending=True)
    colors = sns.color_palette("rocket", n_colors=len(top_rules))

    # Shorten rule labels if too long
    labels = [
        (r if len(r) <= 50 else r[:47] + '...')
        for r in top_rules['rule_str']
    ]

    bars = ax.barh(labels, top_rules['lift'], color=colors, height=0.6)

    ax.set_title(f'Top {min(top_n, len(top_rules))} Association Rules by Lift', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Lift Ratio', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Association Rule (Antecedent  ===>  Consequent)', fontsize=11, fontweight='semibold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Reference line at Lift = 1.0 (baseline independence)
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Lift = 1 (Independent)')
    ax.legend(loc='lower right', frameon=True)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + (top_rules['lift'].max() * 0.01), bar.get_y() + bar.get_height() / 2,
                f'{w:.2f}x', va='center', ha='left', fontsize=9.5, fontweight='bold', color='#1e293b')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig
