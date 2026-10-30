"""
Streamlit Web Application: Market Basket Analysis & Product Recommendation System
Interactive Dashboard for Exploratory Data Analysis, Apriori Association Rule Mining,
and Data-Driven Retail Cross-Selling Recommendations.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import preprocessing as pp
import analysis as ana
import recommendations as rec

# ---------------------------------------------------------
# Page Configuration & Modern Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Basket Analysis & Recommender",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Clean UI
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header Gradient Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* KPI Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #3b82f6;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    
    /* Recommendation Result Cards */
    .rec-card {
        background: white;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #10b981;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .rec-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }
    .badge-lift {
        background-color: #dbeafe;
        color: #1e40af;
    }
    .badge-conf {
        background-color: #dcfce7;
        color: #166534;
    }
    .badge-supp {
        background-color: #fef3c7;
        color: #92400e;
    }
    
    /* Educational Viva Guide Box */
    .concept-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.95rem;
        color: #1e3a8a;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess_dataset(data_path: str):
    """Loads raw dataset and cleans it."""
    raw_df, metadata = pp.load_data(data_path)
    clean_df, stats = pp.clean_data(raw_df)
    return raw_df, clean_df, metadata, stats


@st.cache_data(show_spinner=False)
def get_basket_matrix(df: pd.DataFrame, country: str, max_baskets: int = 5000):
    """Filters dataset by country and constructs boolean basket."""
    filtered_df = pp.filter_by_country(df, country)
    basket_df = pp.create_basket(filtered_df, min_items_per_basket=2, max_baskets=max_baskets)
    return filtered_df, basket_df


@st.cache_data(show_spinner=False)
def run_mba_pipeline(basket_df: pd.DataFrame, min_support: float, min_confidence: float, min_lift: float, max_len: int = 3):
    """Computes frequent itemsets and association rules using Apriori."""
    itemsets = ana.run_apriori(basket_df, min_support=min_support, max_len=max_len)
    rules = ana.generate_rules(itemsets, metric="lift", min_threshold=min_lift, min_confidence=min_confidence)
    return itemsets, rules


# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081840.png", width=70)
st.sidebar.title("🛒 Market Basket AI")
st.sidebar.caption("Data-Driven Cross-Selling Engine")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigate Project",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Analysis (EDA)",
        "📦 Frequent Itemsets",
        "🔗 Association Rules",
        "🎯 Product Recommendations"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Dataset & Analysis Settings")

# Dataset Source Selection
default_files = [p for p in pp.DEFAULT_DATA_PATHS if os.path.exists(p)]
dataset_options = {}
for p in default_files:
    fname = os.path.basename(p)
    if 'sample' in fname.lower():
        dataset_options[f"Sample Retail Data (Fast Demo)"] = p
    elif fname.endswith('.csv'):
        dataset_options[f"Full Online Retail (CSV - Recommended)"] = p
    else:
        dataset_options[f"Online Retail (Excel)"] = p

if not dataset_options:
    st.error("No dataset file detected in data/ directory. Please run with sample or upload.")
    st.stop()

selected_dataset_label = st.sidebar.selectbox("Select Dataset Source", list(dataset_options.keys()))
active_data_path = dataset_options[selected_dataset_label]

# Load Data
with st.spinner("Loading and preprocessing transaction data..."):
    raw_df, clean_df, meta, clean_stats = load_and_preprocess_dataset(active_data_path)

# Country Selector
country_options = ['France', 'United Kingdom', 'Germany', 'EIRE', 'All Countries']
available_countries = [c for c in country_options if c in clean_df['Country'].unique() or c == 'All Countries']
selected_country = st.sidebar.selectbox(
    "Filter by Country",
    available_countries,
    index=0,
    help="France and Germany provide high-affinity baskets ideal for association analysis. UK contains the highest volume."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Apriori Hyperparameters")

# Threshold Sliders
default_min_supp = 0.05 if selected_country in ['France', 'Germany'] else 0.02
min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.01,
    max_value=0.20,
    value=default_min_supp,
    step=0.01,
    help="Proportion of total baskets containing the itemset (e.g. 0.05 = appears in at least 5% of transactions)."
)

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05,
    help="Probability of purchasing Consequent given Antecedent is already purchased."
)

min_lift = st.sidebar.slider(
    "Minimum Lift",
    min_value=1.0,
    max_value=10.0,
    value=1.2,
    step=0.2,
    help="Strength of association beyond random chance. Lift > 1 indicates positive affinity."
)

max_itemset_len = st.sidebar.slider(
    "Max Itemset Length",
    min_value=2,
    max_value=4,
    value=3,
    help="Maximum number of items in combined basket sets."
)

# Educational Sidebar Guide (Viva/Presentation Prep)
with st.sidebar.expander("📚 Concept Guide (Formulas & Viva)"):
    st.markdown("""
    **1. Support:**  
    Frequency of item occurrence across all baskets.  
    $$\\text{Support}(A \\rightarrow B) = \\frac{\\text{Transactions containing } A \\text{ and } B}{\\text{Total Transactions}}$$
    
    **2. Confidence:**  
    Probability item $B$ is purchased when item $A$ is purchased.  
    $$\\text{Confidence}(A \\rightarrow B) = \\frac{\\text{Support}(A \\cup B)}{\\text{Support}(A)}$$
    
    **3. Lift:**  
    Strength of association over expected baseline independence.  
    $$\\text{Lift}(A \\rightarrow B) = \\frac{\\text{Confidence}(A \\rightarrow B)}{\\text{Support}(B)}$$
    - **Lift > 1**: Items boost each other's purchase.
    - **Lift = 1**: Independent items.
    - **Lift < 1**: Substitutes (rarely bought together).
    """)

# Compute Basket and MBA Pipeline
country_df, basket_df = get_basket_matrix(clean_df, selected_country)
itemsets_df, rules_df = run_mba_pipeline(basket_df, min_support, min_confidence, min_lift, max_itemset_len)


# ---------------------------------------------------------
# Page 1: Home (Project Overview & KPIs)
# ---------------------------------------------------------
if nav_selection == "🏠 Home":
    st.markdown(f"""
    <div class="hero-banner">
        <h1 class="hero-title">Market Basket Analysis & Product Recommendations</h1>
        <p class="hero-subtitle">
            Leveraging the Apriori Algorithm to discover frequent shopping patterns, compute association rules,
            and empower retailers with actionable cross-selling intelligence.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Status Badge
    is_sample_badge = "⚠️ Demonstration Sample Dataset" if meta['is_sample'] else "✅ Full UCI Online Retail Dataset"
    st.info(f"**Active Source:** `{meta['file_name']}` ({is_sample_badge}) | **Active Country Scope:** `{selected_country}`")

    # Global KPI Row
    kpis = ana.get_kpis(clean_df)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">{kpis['total_transactions']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #10b981;">
            <div class="kpi-title">Total Revenue</div>
            <div class="kpi-value">£{kpis['total_revenue']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #8b5cf6;">
            <div class="kpi-title">Unique Products</div>
            <div class="kpi-value">{kpis['total_products']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #f59e0b;">
            <div class="kpi-title">Active Customers</div>
            <div class="kpi-value">{kpis['total_customers']:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="border-top-color: #ec4899;">
            <div class="kpi-title">Avg Basket Size</div>
            <div class="kpi-value">{kpis['avg_basket_size']:.1f} items</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Current Scope KPIs
    st.subheader(f"📍 Current Analysis Scope: {selected_country}")
    scope_col1, scope_col2, scope_col3, scope_col4 = st.columns(4)
    with scope_col1:
        st.metric("Multi-Item Baskets Mined", f"{len(basket_df):,}")
    with scope_col2:
        st.metric("Unique Products in Scope", f"{basket_df.shape[1]:,}")
    with scope_col3:
        st.metric("Frequent Itemsets Mined", f"{len(itemsets_df):,}")
    with scope_col4:
        st.metric("Association Rules Formed", f"{len(rules_df):,}")

    st.markdown("---")
    
    # Project Highlights Columns
    h_col1, h_col2, h_col3 = st.columns(3)
    with h_col1:
        st.markdown("### 🛍️ Cross-Selling Strategy")
        st.write("Identifies high-affinity product pairs to recommend complementary products during checkout, increasing average order value (AOV).")
    with h_col2:
        st.markdown("### 📦 Smart Product Bundling")
        st.write("Pairs frequently co-purchased items (e.g. Tea Cups + Saucers) into discounted combo packs to boost sales velocity.")
    with h_col3:
        st.markdown("### 🏪 Inventory & Placement")
        st.write("Informs optimal store layout and e-commerce catalogue placement, placing high-lift items adjacent to each other.")

    # Top Bundles Preview on Home Page
    st.write("")
    st.subheader("🔥 Top High-Affinity Product Bundles Discovered")
    bundles = rec.get_bundle_recommendations(rules_df, top_n=3)
    if bundles:
        b_cols = st.columns(len(bundles))
        for idx, b in enumerate(bundles):
            with b_cols[idx]:
                st.markdown(f"""
                <div class="rec-card" style="border-left-color: #3b82f6;">
                    <span class="badge badge-lift">Lift: {b['lift']}x</span>
                    <span class="badge badge-conf">Conf: {b['confidence']}%</span>
                    <div class="rec-title" style="margin-top: 0.6rem; font-size: 1rem;">
                        {b['item_a']} <br>➕<br> {b['item_b']}
                    </div>
                    <p style="font-size: 0.85rem; color: #475569; margin-top: 0.5rem;">
                        Occurs in <b>{b['support']}%</b> of shopping transactions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Adjust the Minimum Support or Confidence sliders in the sidebar to discover more association bundles.")


# ---------------------------------------------------------
# Page 2: Dataset Overview
# ---------------------------------------------------------
elif nav_selection == "📊 Dataset Overview":
    st.header("📊 Dataset Overview & Data Preprocessing")
    st.write("Exploration of dataset structure, cleaning audit log, and sanitized transaction records.")

    # Data Hygiene & Cleaning Audit Metrics
    st.subheader("🧹 Data Preprocessing & Cleaning Audit")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Initial Raw Rows", f"{clean_stats['initial_records']:,}")
    with c2:
        st.metric("Duplicates Removed", f"{clean_stats['duplicates_removed']:,}")
    with c3:
        st.metric("Cancelled Orders Removed", f"{clean_stats['cancellations_removed']:,}")
    with c4:
        st.metric("Final Valid Records", f"{clean_stats['final_records']:,}", f"{clean_stats['retained_percentage']}% Retained")

    with st.expander("🔍 View Detailed Preprocessing Breakdown"):
        st.markdown(f"""
        - **Duplicates Removed:** {clean_stats['duplicates_removed']:,} records
        - **Missing Product Descriptions:** {clean_stats['missing_descriptions_removed']:,} records dropped
        - **Cancelled Transactions (`InvoiceNo` starting with 'C'):** {clean_stats['cancellations_removed']:,} records removed
        - **Invalid / Negative Quantities:** {clean_stats['invalid_quantities_removed']:,} records filtered
        - **Zero or Negative Unit Prices:** {clean_stats['invalid_prices_removed']:,} records filtered
        - **Administrative / Service Codes Removed:** {clean_stats['service_items_removed']:,} non-product records (e.g. `POSTAGE`, `MANUAL`, `BANK CHARGES`)
        - **Feature Engineering:** Added `TotalAmount = Quantity * UnitPrice`, `YearMonth`, `DayOfWeek`, and `Hour`.
        """)

    st.markdown("---")
    
    # Interactive Data Table Preview
    st.subheader(f"📋 Cleaned Transactions Sample ({clean_df.shape[0]:,} records total)")
    
    filter_col, sort_col = st.columns([2, 1])
    with filter_col:
        search_query = st.text_input("Search transactions by Product Description:", placeholder="e.g. MUG, CLOCK, BAG...")
    
    preview_df = clean_df
    if search_query:
        preview_df = preview_df[preview_df['Description'].str.contains(search_query.strip().upper(), na=False)]

    st.dataframe(
        preview_df[['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'UnitPrice', 'TotalAmount', 'CustomerID', 'Country', 'InvoiceDate']].head(100),
        use_container_width=True,
        height=320
    )

    # Summary Statistics & Schema
    stat_tab1, stat_tab2 = st.tabs(["Numerical Summary Statistics", "Column Schema & Missing Values"])
    with stat_tab1:
        st.dataframe(clean_df[['Quantity', 'UnitPrice', 'TotalAmount']].describe().round(2), use_container_width=True)
    with stat_tab2:
        schema_df = pd.DataFrame({
            'Data Type': clean_df.dtypes.astype(str),
            'Non-Null Count': clean_df.notnull().sum(),
            'Null Count': clean_df.isnull().sum(),
            'Distinct Values': clean_df.nunique()
        })
        st.dataframe(schema_df, use_container_width=True)

    # Download Cleaned Dataset
    csv_data = clean_df.head(5000).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Dataset (Top 5,000 Sample CSV)",
        data=csv_data,
        file_name="cleaned_online_retail_sample.csv",
        mime="text/csv"
    )


# ---------------------------------------------------------
# Page 3: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
elif nav_selection == "📈 Data Analysis (EDA)":
    st.header("📈 Exploratory Data Analysis")
    st.write("Visual analysis of customer shopping habits, sales volume, time trends, and transaction distributions.")

    eda_tabs = st.tabs([
        "🏆 Top Products",
        "📅 Monthly Sales Trends",
        "🛒 Basket Size Distribution",
        "⏰ Weekly Order Patterns",
        "🌍 Geographic Distribution"
    ])

    # Tab 1: Top Products
    with eda_tabs[0]:
        st.subheader("Top Selling Products")
        top_n_slider = st.slider("Select Top N Products to Display", min_value=5, max_value=25, value=10)
        
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            fig_freq = ana.plot_top_frequent_products(country_df, top_n=top_n_slider)
            st.pyplot(fig_freq)
        with t_col2:
            fig_qty = ana.plot_top_products_by_quantity(country_df, top_n=top_n_slider)
            st.pyplot(fig_qty)

    # Tab 2: Monthly Trends
    with eda_tabs[1]:
        st.subheader("Monthly Revenue and Transaction Dynamics")
        fig_monthly = ana.plot_monthly_sales_trend(clean_df)
        st.pyplot(fig_monthly)
        st.caption("Shows monthly sales revenue and transaction counts across the retail cycle. Notice peak shopping surge during Q4 holiday season.")

    # Tab 3: Basket Size Distribution
    with eda_tabs[2]:
        st.subheader("Customer Basket Size (Items per Transaction)")
        fig_basket_dist = ana.plot_basket_size_distribution(country_df)
        st.pyplot(fig_basket_dist)
        st.info("The basket size distribution is right-skewed: the majority of transactions contain 2 to 10 unique items, providing optimal variety for association rule mining.")

    # Tab 4: Weekly Patterns
    with eda_tabs[3]:
        st.subheader("Transaction Volume by Day of Week")
        fig_day = ana.plot_orders_by_day(clean_df)
        st.pyplot(fig_day)

    # Tab 5: Geographic Distribution
    with eda_tabs[4]:
        st.subheader("International Order Volume (Top Countries)")
        fig_countries = ana.plot_top_countries(clean_df, top_n=12)
        st.pyplot(fig_countries)


# ---------------------------------------------------------
# Page 4: Frequent Itemsets (Apriori)
# ---------------------------------------------------------
elif nav_selection == "📦 Frequent Itemsets":
    st.header("📦 Frequent Itemset Mining (Apriori)")
    st.write(
        f"Mined using minimum support threshold **{min_support:.2f}** ({min_support*100:.1f}%) "
        f"in scope: **{selected_country}**."
    )

    if itemsets_df.empty:
        st.warning(
            f"No frequent itemsets met the minimum support of {min_support:.2f}. "
            "Please lower the Minimum Support slider in the sidebar."
        )
    else:
        # Metrics row
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.metric("Total Itemsets Discovered", len(itemsets_df))
        with f_col2:
            st.metric("Single Items (1-Itemsets)", len(itemsets_df[itemsets_df['length'] == 1]))
        with f_col3:
            st.metric("Item Pairs (2-Itemsets)", len(itemsets_df[itemsets_df['length'] == 2]))
        with f_col4:
            st.metric("Triplets+ (3-Itemsets+)", len(itemsets_df[itemsets_df['length'] >= 3]))

        st.markdown("---")

        # Visualization of Top Frequent Itemsets
        st.subheader("Top Frequent Itemsets by Support")
        fig_itemsets = ana.plot_frequent_itemsets(itemsets_df, top_n=12)
        st.pyplot(fig_itemsets)

        st.markdown("---")
        
        # Interactive Filterable Table
        st.subheader("Filter Frequent Itemsets")
        len_filter = st.multiselect("Filter by Itemset Size (Length):", options=[1, 2, 3, 4], default=[1, 2, 3])
        filtered_itemsets = itemsets_df[itemsets_df['length'].isin(len_filter)]

        display_df = filtered_itemsets[['itemsets_display', 'length', 'support']].copy()
        display_df['Support (%)'] = (display_df['support'] * 100).round(2)
        display_df = display_df.rename(columns={'itemsets_display': 'Product Itemset', 'length': 'Item Count'})

        st.dataframe(
            display_df[['Product Itemset', 'Item Count', 'Support (%)']],
            use_container_width=True,
            height=350
        )

        # Download Itemsets
        csv_itemsets = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Frequent Itemsets (CSV)",
            data=csv_itemsets,
            file_name=f"frequent_itemsets_{selected_country.lower()}.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------
# Page 5: Association Rules
# ---------------------------------------------------------
elif nav_selection == "🔗 Association Rules":
    st.header("🔗 Association Rules Mining")
    st.write(
        f"Generated association rules with **Min Support: {min_support:.2f}**, "
        f"**Min Confidence: {min_confidence:.2f}**, and **Min Lift: {min_lift:.2f}**."
    )

    if rules_df.empty:
        st.warning(
            "No association rules met the current criteria. "
            "Try lowering Minimum Confidence or Minimum Support in the sidebar."
        )
    else:
        # Rule Summary KPIs
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            st.metric("Total Association Rules", len(rules_df))
        with r_col2:
            st.metric("Highest Lift Ratio", f"{rules_df['lift'].max():.2f}x")
        with r_col3:
            st.metric("Average Confidence", f"{rules_df['confidence'].mean()*100:.1f}%")
        with r_col4:
            st.metric("Average Lift", f"{rules_df['lift'].mean():.2f}x")

        st.markdown("---")

        # Visualizations Row
        rule_v1, rule_v2 = st.columns(2)
        with rule_v1:
            st.subheader("Support vs. Confidence vs. Lift")
            fig_scatter = ana.plot_rules_scatter(rules_df)
            st.pyplot(fig_scatter)
        with rule_v2:
            st.subheader("Top Rules Ranked by Lift")
            fig_rules = ana.plot_top_rules_lift(rules_df, top_n=10)
            st.pyplot(fig_rules)

        st.markdown("---")

        # Interactive Rules Table
        st.subheader("📋 Mined Association Rules Explorer")

        search_rule = st.text_input("Filter rules by item keyword:", placeholder="e.g. LUNCH BOX, CLOCK, TEA...")
        filtered_rules = rules_df
        if search_rule:
            filtered_rules = filtered_rules[
                filtered_rules['rule_str'].str.contains(search_rule.strip().upper(), na=False)
            ]

        # Format display dataframe
        rules_table = filtered_rules[['antecedents_str', 'consequents_str', 'support_pct', 'confidence_pct', 'lift_val']].copy()
        rules_table.columns = ['Antecedents (If Bought)', 'Consequents (Also Bought)', 'Support (%)', 'Confidence (%)', 'Lift Ratio']

        st.dataframe(
            rules_table,
            use_container_width=True,
            height=350
        )

        # Download Rules Button
        csv_rules = rules_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Association Rules (CSV)",
            data=csv_rules,
            file_name=f"association_rules_{selected_country.lower()}.csv",
            mime="text/csv"
        )


# ---------------------------------------------------------
# Page 6: Product Recommendation System
# ---------------------------------------------------------
elif nav_selection == "🎯 Product Recommendations":
    st.header("🎯 Product Recommendation Engine")
    st.write("Select any product to discover the most relevant cross-sell items based on customer shopping basket history.")

    # Get available products
    available_prods = rec.get_available_products(rules_df, country_df)

    if not available_prods:
        st.warning("No products available for recommendation under current threshold. Adjust sliders to expand rules.")
    else:
        # Product Selector & Configuration Controls
        rec_ctrl1, rec_ctrl2, rec_ctrl3 = st.columns([2, 1, 1])

        with rec_ctrl1:
            # Default to a well-known product if present
            default_index = 0
            for idx, p in enumerate(available_prods):
                if 'ALARM CLOCK' in p or 'LUNCH BOX' in p or 'BAKELIKE' in p:
                    default_index = idx
                    break

            selected_product = st.selectbox(
                "Select a Product to Recommend For:",
                available_prods,
                index=default_index,
                help="Type or select a product from the mined transaction catalogue."
            )

        with rec_ctrl2:
            sort_metric = st.selectbox(
                "Rank Recommendations By:",
                options=['lift', 'confidence', 'support'],
                format_func=lambda x: f"Highest {x.capitalize()}"
            )

        with rec_ctrl3:
            num_recs = st.slider("Top N Recommendations:", min_value=1, max_value=10, value=4)

        st.markdown("---")

        # Generate Recommendations
        recommendations_df = rec.recommend_products(
            selected_product=selected_product,
            rules_df=rules_df,
            basket_df=basket_df,
            sort_by=sort_metric,
            top_n=num_recs,
            min_confidence=0.05,
            min_lift=1.0
        )

        if recommendations_df.empty:
            st.info(f"No strong cross-selling associations found for **{selected_product}** under strict criteria. Try lowering the Lift or Support sliders.")
        else:
            st.subheader(f"✨ Recommended Cross-Sell Products for: `{selected_product}`")
            
            # Recommendation Cards
            for _, r in recommendations_df.iterrows():
                target_prod = r['Recommended Product']
                lift_val = r['Lift']
                conf_val = r['Confidence (%)']
                supp_val = r['Support (%)']
                rec_type = r['Recommendation Type']
                rationale = r['Business Rationale']

                st.markdown(f"""
                <div class="rec-card">
                    <span class="badge badge-lift">⚡ Lift: {lift_val}x</span>
                    <span class="badge badge-conf">🎯 Confidence: {conf_val}%</span>
                    <span class="badge badge-supp">📊 Support: {supp_val}%</span>
                    <span class="badge" style="background-color: #f1f5f9; color: #475569;">🏷️ {rec_type}</span>
                    <div class="rec-title" style="margin-top: 0.6rem;">{target_prod}</div>
                    <div style="font-size: 0.95rem; color: #334155; margin-top: 0.4rem;">
                        {rationale}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Tabular View Option
            with st.expander("📋 View Recommendations Data Table"):
                st.dataframe(recommendations_df, use_container_width=True)

        # Pre-packaged Strategic Bundles Section
        st.markdown("---")
        st.subheader("💡 Recommended Strategic Retail Bundles")
        st.write("Automated 2-item combo packages ready for marketing promotions and checkout discounts.")
        
        bundles = rec.get_bundle_recommendations(rules_df, top_n=6)
        if bundles:
            b_grid1, b_grid2, b_grid3 = st.columns(3)
            grids = [b_grid1, b_grid2, b_grid3]
            for idx, b in enumerate(bundles):
                target_grid = grids[idx % 3]
                with target_grid:
                    st.markdown(f"""
                    <div class="rec-card" style="border-left-color: #8b5cf6;">
                        <span class="badge badge-lift">Lift: {b['lift']}x</span>
                        <span class="badge badge-conf">Confidence: {b['confidence']}%</span>
                        <div class="rec-title" style="font-size: 0.98rem; margin-top: 0.5rem;">
                            🛍️ {b['item_a']} <br>➕<br> {b['item_b']}
                        </div>
                        <p style="font-size: 0.85rem; color: #64748b; margin-top: 0.4rem;">
                            Basket co-occurrence: <b>{b['support']}%</b> of total transactions.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
