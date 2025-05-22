#!/usr/bin/env python3
"""
Console Summary for LLM Guardrail Benchmark Results
Provides a comprehensive text-based summary of test results
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
import numpy as np

def load_run_data(run_dir):
    """Load all available data from a test run directory"""
    data = {}
    
    # Load analytics CSV
    analytics_path = os.path.join(run_dir, "analytics.csv")
    if os.path.exists(analytics_path):
        data['analytics'] = pd.read_csv(analytics_path)
        print(f"✓ Loaded analytics data: {len(data['analytics'])} rows")
    else:
        print(f"✗ Analytics file not found: {analytics_path}")
        return None
    
    # Load summary JSON
    summary_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            data['summary'] = json.load(f)
        print(f"✓ Loaded summary data")
    else:
        print(f"⚠ Summary file not found: {summary_path}")
        data['summary'] = None
    
    # Load metadata
    metadata_path = os.path.join(run_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
        print(f"✓ Loaded metadata")
    else:
        print(f"⚠ Metadata file not found: {metadata_path}")
        data['metadata'] = None
    
    return data

def print_header(title, char="=", width=80):
    """Print a formatted header"""
    print()
    print(char * width)
    print(f" {title} ".center(width, char))
    print(char * width)

def print_section(title, char="-", width=80):
    """Print a formatted section header"""
    print()
    print(f" {title} ".center(width, char))
    print()

def print_table(headers, rows, col_widths=None):
    """Print a formatted table"""
    if not col_widths:
        # Auto-calculate column widths
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)
    
    # Print header
    header_row = "│"
    separator = "├"
    for i, (header, width) in enumerate(zip(headers, col_widths)):
        header_row += f" {header:<{width-1}}│"
        separator += "─" * width + "┼" if i < len(headers) - 1 else "─" * width + "┤"
    
    print("┌" + "─" * (sum(col_widths) + len(headers) - 1) + "┐")
    print(header_row)
    print(separator)
    
    # Print rows
    for row in rows:
        row_str = "│"
        for i, (cell, width) in enumerate(zip(row, col_widths)):
            row_str += f" {str(cell):<{width-1}}│"
        print(row_str)
    
    print("└" + "─" * (sum(col_widths) + len(headers) - 1) + "┘")

def format_percentage(value):
    """Format a decimal as a percentage"""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1%}"

def format_number(value):
    """Format a number with commas"""
    if pd.isna(value):
        return "N/A"
    return f"{value:,}"

def print_executive_summary(data):
    """Print executive summary"""
    print_header("🛡️ LLM GUARDRAIL BENCHMARK - EXECUTIVE SUMMARY")
    
    df = data['analytics']
    metadata = data.get('metadata', {})
    
    # Basic stats
    models = df['model'].unique()
    total_tests = len(df['test_id'].unique()) if 'test_id' in df.columns else len(df[df['model'] == models[0]])
    categories = df['category'].unique() if 'category' in df.columns else []
    overall_refusal = df['contains_refusal'].mean()
    
    print(f"📅 Test Run Date: {metadata.get('timestamp', 'Unknown')}")
    print(f"🤖 Models Tested: {', '.join(models)} ({len(models)} total)")
    print(f"🧪 Total Tests: {total_tests}")
    print(f"📂 Categories: {len([c for c in categories if pd.notna(c)]) if len(categories) > 0 else 0}")
    print(f"🚫 Overall Refusal Rate: {format_percentage(overall_refusal)}")
    
    if len(categories) > 0:
        print(f"📋 Test Categories: {', '.join([c for c in categories if pd.notna(c)])}")

def print_model_comparison(data):
    """Print detailed model comparison"""
    print_section("📊 MODEL PERFORMANCE COMPARISON")
    
    df = data['analytics']
    models = df['model'].unique()
    
    # Create comparison table
    headers = ["Model", "Tests", "Refusal Rate", "Complete Refusal", "Refusal w/ Alt", "Cautious", "Full Compliance"]
    rows = []
    
    for model in models:
        model_df = df[df['model'] == model]
        refusal_rate = model_df['contains_refusal'].mean()
        response_counts = model_df['classification'].value_counts()
        
        row = [
            model,
            format_number(len(model_df)),
            format_percentage(refusal_rate),
            format_number(response_counts.get('complete_refusal', 0)),
            format_number(response_counts.get('refusal_with_alternative', 0)),
            format_number(response_counts.get('cautious_compliance', 0)),
            format_number(response_counts.get('full_compliance', 0))
        ]
        rows.append(row)
    
    print_table(headers, rows)
    
    # Print insights
    if len(models) > 1:
        model_refusal = df.groupby('model')['contains_refusal'].mean()
        most_restrictive = model_refusal.idxmax()
        least_restrictive = model_refusal.idxmin()
        difference = model_refusal.max() - model_refusal.min()
        
        print(f"🔍 Key Insights:")
        print(f"   • Most restrictive: {most_restrictive} ({format_percentage(model_refusal[most_restrictive])})")
        print(f"   • Least restrictive: {least_restrictive} ({format_percentage(model_refusal[least_restrictive])})")
        print(f"   • Difference: {format_percentage(difference)}")
        
        # Check for unique response strategies
        for model in models:
            model_df = df[df['model'] == model]
            response_counts = model_df['classification'].value_counts()
            
            if response_counts.get('refusal_with_alternative', 0) > 0:
                alt_rate = response_counts.get('refusal_with_alternative', 0) / len(model_df)
                print(f"   • {model} offers alternatives in {format_percentage(alt_rate)} of refusals")

def print_category_analysis(data):
    """Print category-wise analysis"""
    df = data['analytics']
    
    if 'category' not in df.columns or len(df['category'].unique()) <= 1:
        print_section("📂 CATEGORY ANALYSIS")
        print("No category data available for analysis.")
        return
    
    print_section("📂 CATEGORY ANALYSIS")
    
    categories = [c for c in df['category'].unique() if pd.notna(c)]
    models = df['model'].unique()
    
    # Create category comparison table
    headers = ["Category", "Overall Refusal"]
    for model in models:
        headers.append(f"{model} Refusal")
    headers.append("Difference")
    
    rows = []
    category_insights = []
    
    for category in sorted(categories):
        cat_df = df[df['category'] == category]
        overall_refusal = cat_df['contains_refusal'].mean()
        
        row = [category, format_percentage(overall_refusal)]
        model_rates = []
        
        for model in models:
            model_cat_df = cat_df[cat_df['model'] == model]
            if len(model_cat_df) > 0:
                rate = model_cat_df['contains_refusal'].mean()
                model_rates.append(rate)
                row.append(format_percentage(rate))
            else:
                model_rates.append(0)
                row.append("N/A")
        
        # Calculate difference between models
        if len(model_rates) > 1:
            diff = max(model_rates) - min(model_rates)
            row.append(format_percentage(diff))
            
            # Store for insights
            category_insights.append({
                'category': category,
                'overall': overall_refusal,
                'difference': diff,
                'model_rates': dict(zip(models, model_rates))
            })
        else:
            row.append("N/A")
        
        rows.append(row)
    
    print_table(headers, rows)
    
    # Print insights
    if category_insights:
        print(f"🔍 Category Insights:")
        
        # Highest refusal rate category
        highest_cat = max(category_insights, key=lambda x: x['overall'])
        print(f"   • Highest refusal rate: {highest_cat['category']} ({format_percentage(highest_cat['overall'])})")
        
        # Lowest refusal rate category
        lowest_cat = min(category_insights, key=lambda x: x['overall'])
        print(f"   • Lowest refusal rate: {lowest_cat['category']} ({format_percentage(lowest_cat['overall'])})")
        
        # Biggest model difference
        biggest_diff = max(category_insights, key=lambda x: x['difference'])
        print(f"   • Biggest model disagreement: {biggest_diff['category']} ({format_percentage(biggest_diff['difference'])} difference)")
        
        # Categories where models agree
        agreement_threshold = 0.1  # 10% difference
        agreeable_cats = [cat for cat in category_insights if cat['difference'] < agreement_threshold]
        if agreeable_cats:
            print(f"   • Models agree most on: {', '.join([cat['category'] for cat in agreeable_cats])}")

def print_variation_analysis(data):
    """Print variation dimension analysis"""
    df = data['analytics']
    
    if 'variation_type' not in df.columns:
        print_section("🔄 VARIATION ANALYSIS")
        print("No variation data available for analysis.")
        return
    
    print_section("🔄 VARIATION DIMENSION ANALYSIS")
    
    variation_types = [vt for vt in df['variation_type'].unique() 
                      if vt not in ('baseline', 'sequential', 'unknown') and pd.notna(vt)]
    
    if not variation_types:
        print("No variation dimension data available.")
        return
    
    models = df['model'].unique()
    
    for var_type in variation_types:
        print(f"\n📈 {var_type.title()} Dimension:")
        
        var_df = df[(df['variation_type'] == var_type) & (df['variation_level'] > 0)]
        
        if len(var_df) < 3:
            print(f"   Insufficient data for analysis ({len(var_df)} samples)")
            continue
        
        # Create level-wise breakdown
        levels = sorted(var_df['variation_level'].unique())
        
        headers = ["Level", "Description"] + [f"{model}" for model in models] + ["Avg Refusal"]
        rows = []
        
        for level in levels:
            level_df = var_df[var_df['variation_level'] == level]
            
            # Get description from first sample
            description = level_df['variation_description'].iloc[0] if 'variation_description' in level_df.columns else f"Level {level}"
            if pd.isna(description):
                description = f"Level {level}"
            
            row = [str(level), description[:20] + "..." if len(description) > 20 else description]
            
            model_rates = []
            for model in models:
                model_level_df = level_df[level_df['model'] == model]
                if len(model_level_df) > 0:
                    rate = model_level_df['contains_refusal'].mean()
                    model_rates.append(rate)
                    row.append(format_percentage(rate))
                else:
                    row.append("N/A")
            
            # Average across models
            if model_rates:
                avg_rate = sum(model_rates) / len(model_rates)
                row.append(format_percentage(avg_rate))
            else:
                row.append("N/A")
            
            rows.append(row)
        
        print_table(headers, rows, col_widths=[6, 25] + [12] * len(models) + [12])
        
        # Print trend analysis
        if len(levels) > 1:
            # Calculate trend for each model
            trends = []
            for model in models:
                model_var_df = var_df[var_df['model'] == model]
                if len(model_var_df) > 1:
                    level_rates = model_var_df.groupby('variation_level')['contains_refusal'].mean()
                    if len(level_rates) > 1:
                        trend = level_rates.iloc[-1] - level_rates.iloc[0]  # Last - First
                        trends.append(f"{model}: {'+' if trend > 0 else ''}{format_percentage(trend)}")
            
            if trends:
                print(f"   📊 Trend (Level {min(levels)} → {max(levels)}): {', '.join(trends)}")

def print_sequential_analysis(data):
    """Print sequential refinement analysis"""
    df = data['analytics']
    
    if 'sequence_position' not in df.columns:
        print_section("🔄 SEQUENTIAL REFINEMENT ANALYSIS")
        print("No sequential data available for analysis.")
        return
    
    seq_df = df[df['sequence_position'] > 0]
    
    if len(seq_df) < 5:
        print_section("🔄 SEQUENTIAL REFINEMENT ANALYSIS")
        print(f"Insufficient sequential data for analysis ({len(seq_df)} samples)")
        return
    
    print_section("🔄 SEQUENTIAL REFINEMENT ANALYSIS")
    
    models = seq_df['model'].unique()
    positions = sorted(seq_df['sequence_position'].unique())
    
    # Create position-wise breakdown
    headers = ["Position"] + [f"{model}" for model in models] + ["Average"]
    rows = []
    
    position_insights = []
    
    for pos in positions:
        pos_df = seq_df[seq_df['sequence_position'] == pos]
        
        row = [f"Step {pos}"]
        model_rates = []
        
        for model in models:
            model_pos_df = pos_df[pos_df['model'] == model]
            if len(model_pos_df) > 0:
                rate = model_pos_df['contains_refusal'].mean()
                model_rates.append(rate)
                row.append(format_percentage(rate))
            else:
                row.append("N/A")
        
        # Average across models
        if model_rates:
            avg_rate = sum(model_rates) / len(model_rates)
            row.append(format_percentage(avg_rate))
            position_insights.append({
                'position': pos,
                'average': avg_rate,
                'model_rates': dict(zip(models, model_rates))
            })
        else:
            row.append("N/A")
        
        rows.append(row)
    
    print_table(headers, rows)
    
    # Print sequential insights
    if len(position_insights) > 1:
        print(f"🔍 Sequential Insights:")
        
        first_step = position_insights[0]
        last_step = position_insights[-1]
        
        # Overall trend
        overall_change = last_step['average'] - first_step['average']
        direction = "increases" if overall_change > 0 else "decreases"
        print(f"   • Refusal rate {direction} by {format_percentage(abs(overall_change))} from step 1 to {len(positions)}")
        
        # Model-specific trends
        for model in models:
            if model in first_step['model_rates'] and model in last_step['model_rates']:
                change = last_step['model_rates'][model] - first_step['model_rates'][model]
                if abs(change) > 0.1:  # Only show significant changes
                    direction = "increases" if change > 0 else "decreases"
                    print(f"   • {model} {direction} by {format_percentage(abs(change))}")
        
        # Find turning points
        if len(position_insights) > 2:
            max_pos = max(position_insights, key=lambda x: x['average'])
            min_pos = min(position_insights, key=lambda x: x['average'])
            
            if max_pos['position'] != last_step['position']:
                print(f"   • Peak refusal at step {max_pos['position']} ({format_percentage(max_pos['average'])})")
            if min_pos['position'] != first_step['position']:
                print(f"   • Lowest refusal at step {min_pos['position']} ({format_percentage(min_pos['average'])})")

def print_risk_analysis(data):
    """Print risk level analysis"""
    df = data['analytics']
    
    if 'risk_level' not in df.columns or len(df['risk_level'].unique()) <= 1:
        return  # Skip if no risk data
    
    print_section("⚠️ RISK LEVEL ANALYSIS")
    
    risk_levels = [r for r in df['risk_level'].unique() if pd.notna(r)]
    models = df['model'].unique()
    
    # Create risk level comparison table
    headers = ["Risk Level", "Overall Refusal"] + [f"{model}" for model in models]
    rows = []
    
    for risk in sorted(risk_levels):
        risk_df = df[df['risk_level'] == risk]
        overall_refusal = risk_df['contains_refusal'].mean()
        
        row = [risk, format_percentage(overall_refusal)]
        
        for model in models:
            model_risk_df = risk_df[risk_df['model'] == model]
            if len(model_risk_df) > 0:
                rate = model_risk_df['contains_refusal'].mean()
                row.append(format_percentage(rate))
            else:
                row.append("N/A")
        
        rows.append(row)
    
    print_table(headers, rows)

def print_detailed_insights(data):
    """Print detailed insights and recommendations"""
    print_section("💡 DETAILED INSIGHTS & RECOMMENDATIONS")
    
    df = data['analytics']
    models = df['model'].unique()
    
    insights = []
    
    # Model comparison insights
    if len(models) > 1:
        model_refusal = df.groupby('model')['contains_refusal'].mean()
        model_response_strategies = {}
        
        for model in models:
            model_df = df[df['model'] == model]
            strategies = model_df['classification'].value_counts(normalize=True)
            model_response_strategies[model] = strategies
        
        # Find unique strategies
        for model in models:
            strategies = model_response_strategies[model]
            alt_rate = strategies.get('refusal_with_alternative', 0)
            
            if alt_rate > 0.1:  # More than 10% of responses offer alternatives
                insights.append(f"🔄 {model} frequently offers alternatives when refusing ({format_percentage(alt_rate)})")
            elif alt_rate == 0:
                insights.append(f"🚫 {model} never offers alternatives when refusing - uses binary approach")
    
    # Category-specific insights
    if 'category' in df.columns:
        categories = [c for c in df['category'].unique() if pd.notna(c)]
        
        for category in categories:
            cat_df = df[df['category'] == category]
            cat_refusal = cat_df['contains_refusal'].mean()
            
            if cat_refusal > 0.7:
                insights.append(f"🔒 {category} has very high refusal rates ({format_percentage(cat_refusal)}) - strong guardrails")
            elif cat_refusal < 0.2:
                insights.append(f"⚠️ {category} has low refusal rates ({format_percentage(cat_refusal)}) - potential vulnerability")
    
    # Variation insights
    if 'variation_type' in df.columns:
        variation_types = [vt for vt in df['variation_type'].unique() 
                          if vt not in ('baseline', 'sequential', 'unknown') and pd.notna(vt)]
        
        for var_type in variation_types:
            var_df = df[df['variation_type'] == var_type]
            if len(var_df) > 5:
                # Check if variation level affects refusal rates
                level_effect = var_df.groupby('variation_level')['contains_refusal'].mean()
                if len(level_effect) > 1:
                    max_effect = level_effect.max() - level_effect.min()
                    if max_effect > 0.3:
                        insights.append(f"📈 {var_type.title()} variations significantly affect refusal rates (up to {format_percentage(max_effect)} difference)")
    
    # Print insights
    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
    else:
        print("No specific patterns identified in the current dataset.")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    
    if len(models) > 1:
        # Compare refusal rates
        model_refusal = df.groupby('model')['contains_refusal'].mean()
        most_restrictive = model_refusal.idxmax()
        least_restrictive = model_refusal.idxmin()
        
        print(f"1. Consider why {most_restrictive} is more restrictive than {least_restrictive}")
        print(f"2. Evaluate if {least_restrictive}'s lower refusal rate indicates vulnerability")
        
        # Check for alternative offering
        for model in models:
            model_df = df[df['model'] == model]
            alt_count = (model_df['classification'] == 'refusal_with_alternative').sum()
            if alt_count == 0:
                print(f"3. Consider implementing 'refusal with alternative' strategy for {model}")
    
    if 'category' in df.columns:
        categories = [c for c in df['category'].unique() if pd.notna(c)]
        cat_refusal = df.groupby('category')['contains_refusal'].mean()
        
        # Find categories with low refusal rates
        vulnerable_cats = [cat for cat in categories if cat_refusal.get(cat, 0) < 0.3]
        if vulnerable_cats:
            print(f"4. Strengthen guardrails for: {', '.join(vulnerable_cats)}")
    
    print(f"5. Expand testing with more variation dimensions and sequential refinement")
    print(f"6. Consider testing additional models for comparison")

def main():
    if len(sys.argv) < 2:
        print("❌ Please provide the run directory path as an argument.")
        print("Example: python3.11 console_summary.py results/run_20250521-173746")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    if not os.path.exists(run_dir):
        print(f"❌ Run directory not found: {run_dir}")
        sys.exit(1)
    
    print(f"🔍 Loading data from: {run_dir}")
    
    # Load data
    data = load_run_data(run_dir)
    if data is None:
        print("❌ Failed to load data")
        sys.exit(1)
    
    # Generate summary sections
    print_executive_summary(data)
    print_model_comparison(data)
    print_category_analysis(data)
    print_variation_analysis(data)
    print_sequential_analysis(data)
    print_risk_analysis(data)
    print_detailed_insights(data)
    
    print_header("🎯 SUMMARY COMPLETE")
    print(f"📊 Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Data source: {run_dir}")

if __name__ == "__main__":
    main()