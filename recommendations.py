"""
Product Recommendation Engine for Market Basket Analysis
Maps selected retail items to optimal cross-sell recommendations using association rules and co-occurrence patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def get_available_products(rules_df: pd.DataFrame, clean_df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Returns sorted list of distinct products available for recommendation.
    Prioritizes products that have mined association rules.
    """
    products_with_rules = set()

    if not rules_df.empty:
        for antecedents in rules_df['antecedents']:
            products_with_rules.update(antecedents)

    if products_with_rules:
        return sorted(list(products_with_rules))
    
    # Fallback to top products in clean_df if no rules exist yet
    if clean_df is not None and not clean_df.empty:
        return sorted(clean_df['Description'].dropna().unique().tolist())
        
    return []


def explain_rule(selected_item: str, recommended_item: str, confidence: float, lift: float, support: float) -> str:
    """
    Generates a clear business narrative explaining why a product is recommended.
    """
    conf_pct = round(confidence * 100, 1)
    supp_pct = round(support * 100, 2)
    lift_val = round(lift, 2)

    narrative = (
        f"Customers who placed **{selected_item}** in their basket have a **{conf_pct}% chance** "
        f"of also purchasing **{recommended_item}**. "
        f"They are **{lift_val}x more likely** to purchase it than an average customer "
        f"(basket co-occurrence frequency: {supp_pct}%)."
    )
    return narrative


def recommend_products(
    selected_product: str,
    rules_df: pd.DataFrame,
    basket_df: Optional[pd.DataFrame] = None,
    sort_by: str = 'lift',
    top_n: int = 5,
    min_confidence: float = 0.0,
    min_lift: float = 1.0
) -> pd.DataFrame:
    """
    Finds product recommendations for a selected product using mined association rules.
    If no association rules match, automatically falls back to empirical basket co-occurrence.
    
    Args:
        selected_product: The product description selected by the user.
        rules_df: association_rules DataFrame.
        basket_df: Optional boolean basket DataFrame for fallback co-occurrence.
        sort_by: Metric to order recommendations ('lift', 'confidence', 'support').
        top_n: Maximum number of recommendations to return.
        min_confidence: Minimum confidence threshold filter.
        min_lift: Minimum lift threshold filter.
        
    Returns:
        DataFrame with columns ['Recommended Product', 'Support (%)', 'Confidence (%)', 'Lift', 'Recommendation Type', 'Business Rationale']
    """
    output_cols = [
        'Recommended Product',
        'Support (%)',
        'Confidence (%)',
        'Lift',
        'Recommendation Type',
        'Business Rationale'
    ]

    matched_records = []

    # 1. Search in Association Rules
    if not rules_df.empty:
        for _, row in rules_df.iterrows():
            antecedents = list(row['antecedents'])
            consequents = list(row['consequents'])

            if selected_product in antecedents:
                # Find consequents that are not the selected product
                for target_item in consequents:
                    if target_item != selected_product:
                        conf = float(row['confidence'])
                        lift = float(row['lift'])
                        supp = float(row['support'])

                        if conf >= min_confidence and lift >= min_lift:
                            matched_records.append({
                                'Recommended Product': target_item,
                                'Support (%)': round(supp * 100, 2),
                                'Confidence (%)': round(conf * 100, 1),
                                'Lift': round(lift, 2),
                                'raw_support': supp,
                                'raw_confidence': conf,
                                'raw_lift': lift,
                                'Recommendation Type': 'Association Rule (Apriori)',
                                'Business Rationale': explain_rule(selected_product, target_item, conf, lift, supp)
                            })

    # Deduplicate recommendations by picking best score for each unique recommended product
    if matched_records:
        rec_df = pd.DataFrame(matched_records)
        
        sort_field = f'raw_{sort_by}' if f'raw_{sort_by}' in rec_df.columns else 'raw_lift'
        rec_df = rec_df.sort_values(by=sort_field, ascending=False)
        rec_df = rec_df.drop_duplicates(subset=['Recommended Product']).head(top_n)
        return rec_df[output_cols]

    # 2. Fallback: Empirical Co-occurrence from basket matrix
    if basket_df is not None and not basket_df.empty and selected_product in basket_df.columns:
        product_series = basket_df[selected_product]
        selected_count = int(product_series.sum())

        if selected_count > 0:
            # Transactions containing selected product
            matching_baskets = basket_df[product_series]
            total_baskets = len(basket_df)

            co_counts = matching_baskets.sum(axis=0).drop(labels=[selected_product], errors='ignore')
            top_co = co_counts.nlargest(top_n)

            fallback_records = []
            for item, count in top_co.items():
                if count > 0:
                    supp = count / total_baskets
                    conf = count / selected_count
                    overall_item_freq = basket_df[item].sum() / total_baskets
                    lift = conf / overall_item_freq if overall_item_freq > 0 else 1.0

                    fallback_records.append({
                        'Recommended Product': item,
                        'Support (%)': round(supp * 100, 2),
                        'Confidence (%)': round(conf * 100, 1),
                        'Lift': round(lift, 2),
                        'Recommendation Type': 'Empirical Co-occurrence',
                        'Business Rationale': explain_rule(selected_product, item, conf, lift, supp)
                    })

            if fallback_records:
                fb_df = pd.DataFrame(fallback_records)
                fb_df = fb_df.sort_values(by=sort_by.capitalize() if sort_by.capitalize() in fb_df.columns else 'Lift', ascending=False)
                return fb_df[output_cols]

    return pd.DataFrame(columns=output_cols)


def get_bundle_recommendations(rules_df: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Identifies top pre-packaged bundles (cross-sell packages) based on strongest Lift & Confidence.
    """
    if rules_df.empty:
        return []

    bundles = []
    seen_pairs = set()

    sorted_rules = rules_df.sort_values(by=['lift', 'confidence'], ascending=[False, False])

    for _, row in sorted_rules.iterrows():
        ant = row['antecedents_str']
        con = row['consequents_str']
        pair_key = tuple(sorted([ant, con]))

        if pair_key not in seen_pairs:
            seen_pairs.add(pair_key)
            bundles.append({
                'item_a': ant,
                'item_b': con,
                'lift': round(row['lift'], 2),
                'confidence': round(row['confidence'] * 100, 1),
                'support': round(row['support'] * 100, 2),
                'description': f"Bundle: {ant} + {con}"
            })

            if len(bundles) >= top_n:
                break

    return bundles
