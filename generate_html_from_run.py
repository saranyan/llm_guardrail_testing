#!/usr/bin/env python3
"""
HTML Report Generator for LLM Guardrail Benchmark Results
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up matplotlib to work without display
import matplotlib
matplotlib.use('Agg')

# Set up the Seaborn style
sns.set_theme(style="whitegrid")

def generate_html_report(run_dir: str):
    """Generate HTML report with visualizations for a test run"""
    print(f"Starting report generation for: {run_dir}")
    
    # Create visualizations directory
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"Created visualizations directory: {vis_dir}")
    
    # Load the data
    analytics_path = os.path.join(run_dir, "analytics.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    
    if not os.path.exists(analytics_path):
        raise FileNotFoundError(f"Analytics file not found: {analytics_path}")
    
    print(f"Loading data from: {analytics_path}")
    df = pd.read_csv(analytics_path)
    print(f"Loaded {len(df)} rows of data")
    print(f"Columns: {list(df.columns)}")
    
    # Load summary if available
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        print("Loaded summary data")
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Response categories by model
    print("Creating response categories plot...")
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(x='classification', hue='model', data=df)
    plt.title('Response Categories by Model', fontsize=16)
    plt.xlabel('Response Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add count labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d')
    
    # Save the plot
    plt.savefig(os.path.join(vis_dir, 'model_response_categories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Response categories plot saved")
    
    # 2. Refusal rates by category and model (if category field exists)
    if 'category' in df.columns:
        print("Creating category refusal rates plot...")
        # Remove any rows where category is NaN
        cat_df = df[df['category'].notna()]
        if len(cat_df) > 0:
            plt.figure(figsize=(14, 8))
            
            # Create the pivot table
            category_model_refusal = cat_df.groupby(['category', 'model'])['contains_refusal'].mean().unstack(fill_value=0)
            
            # Check if we have data
            if len(category_model_refusal) > 0:
                ax = category_model_refusal.plot(kind='bar', figsize=(14, 8))
                plt.title('Refusal Rates by Category and Model', fontsize=16)
                plt.xlabel('Category', fontsize=14)
                plt.ylabel('Refusal Rate', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.ylim(0, 1)
                plt.legend(title='Model')
                plt.tight_layout()
                
                # FIXED: Add percentage labels properly
                for container in ax.containers:
                    # Get the actual height values from the Rectangle objects
                    heights = [rect.get_height() for rect in container]
                    labels = [f'{h:.0%}' if h > 0.05 else '' for h in heights]
                    ax.bar_label(container, labels=labels, label_type='center', fontsize=9)
                    
                plt.savefig(os.path.join(vis_dir, 'category_refusal_rates.png'), dpi=300, bbox_inches='tight')
                print("✓ Category refusal rates plot saved")
            plt.close()
    else:
        print("No 'category' column found, skipping category analysis")
    
    # 3. Risk level analysis (if risk_level field exists)
    if 'risk_level' in df.columns and len(df['risk_level'].unique()) > 1:
        print("Creating risk level analysis plot...")
        risk_df = df[df['risk_level'].notna()]
        if len(risk_df) > 0:
            plt.figure(figsize=(10, 6))
            risk_model_refusal = risk_df.groupby(['risk_level', 'model'])['contains_refusal'].mean().unstack(fill_value=0)
            
            ax = risk_model_refusal.plot(kind='bar', figsize=(10, 6))
            plt.title('Refusal Rates by Risk Level and Model', fontsize=16)
            plt.xlabel('Risk Level', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            plt.legend(title='Model')
            
            # FIXED: Add percentage labels properly
            for container in ax.containers:
                # Get the actual height values from the Rectangle objects
                heights = [rect.get_height() for rect in container]
                labels = [f'{h:.0%}' if h > 0.05 else '' for h in heights]
                ax.bar_label(container, labels=labels, label_type='center', fontsize=9)
                
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'risk_level_refusal_rates.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Risk level analysis plot saved")
    else:
        print("No 'risk_level' column found or insufficient data, skipping risk level analysis")
    
    # 4. Variation analysis (if variation_type field exists)
    if 'variation_type' in df.columns:
        print("Creating variation analysis plots...")
        var_df = df[df['variation_type'].notna()]
        variation_types = [vt for vt in var_df['variation_type'].unique() 
                         if vt not in ('baseline', 'sequential', 'unknown') and pd.notna(vt)]
        
        print(f"Found variation types: {variation_types}")
        
        for var_type in variation_types:
            print(f"Processing {var_type} variation...")
            type_df = var_df[(var_df['variation_type'] == var_type) & (var_df['variation_level'] > 0)]
            
            if len(type_df) < 3:  # Skip if not enough data
                print(f"Skipping {var_type} - not enough data ({len(type_df)} rows)")
                continue
                
            plt.figure(figsize=(12, 7))
            
            # Group by level and model
            level_model_refusal = type_df.groupby(['variation_level', 'model'])['contains_refusal'].mean().unstack(fill_value=0)
            
            if len(level_model_refusal) < 2:  # Skip if not enough levels
                print(f"Skipping {var_type} - not enough variation levels")
                plt.close()
                continue
                
            # Plot line chart
            ax = level_model_refusal.plot(kind='line', marker='o', figsize=(12, 7), linewidth=3, markersize=8)
            plt.title(f'Refusal Rates by {var_type.title()} Level and Model', fontsize=16)
            plt.xlabel(f'{var_type.title()} Level', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # Set integer x-ticks
            plt.xticks(level_model_refusal.index)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{var_type}_refusal_progression.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ {var_type} variation plot saved")
    else:
        print("No 'variation_type' column found, skipping variation analysis")
    
    # 5. Sequential progression if available
    if 'sequence_position' in df.columns:
        print("Creating sequential progression plot...")
        seq_df = df[df['sequence_position'] > 0]
        if len(seq_df) >= 5:
            plt.figure(figsize=(12, 7))
            
            # Group by position and model
            position_model_refusal = seq_df.groupby(['sequence_position', 'model'])['contains_refusal'].mean().unstack(fill_value=0)
            
            # Plot line chart
            ax = position_model_refusal.plot(kind='line', marker='o', figsize=(12, 7), linewidth=3, markersize=8)
            plt.title('Refusal Rate Progression in Sequential Tests', fontsize=16)
            plt.xlabel('Sequence Position', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            
            # Format y-axis as percentages
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            
            # Set integer x-ticks
            plt.xticks(position_model_refusal.index)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'sequential_refusal_progression.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Sequential progression plot saved")
        else:
            print(f"Not enough sequential data ({len(seq_df)} rows), skipping sequential analysis")
    else:
        print("No 'sequence_position' column found, skipping sequential analysis")
    
    # Now generate the HTML report
    print("Generating HTML report...")
    
    # Get metadata
    models = df['model'].unique()
    test_count = len(df['test_id'].unique()) if 'test_id' in df.columns else len(df[df['model'] == models[0]])
    categories = df['category'].unique() if 'category' in df.columns else []
    
    # Get overall refusal rate
    overall_refusal = df['contains_refusal'].mean()
    
    # Create model comparison table
    model_comparison = ""
    for model in models:
        model_df = df[df['model'] == model]
        refusal_rate = model_df['contains_refusal'].mean()
        response_types = model_df['classification'].value_counts().to_dict()
        
        model_comparison += f"""
        <tr>
            <td>{model}</td>
            <td>{refusal_rate:.1%}</td>
            <td>{response_types.get('complete_refusal', 0)}</td>
            <td>{response_types.get('refusal_with_alternative', 0)}</td>
            <td>{response_types.get('cautious_compliance', 0)}</td>
            <td>{response_types.get('full_compliance', 0)}</td>
        </tr>
        """
    
    # Create category comparison table
    category_comparison = ""
    if len(categories) > 0:
        for category in categories:
            if pd.isna(category):
                continue
            cat_df = df[df['category'] == category]
            overall_rate = cat_df['contains_refusal'].mean()
            
            model_rates = []
            for model in models:
                model_cat_df = cat_df[cat_df['model'] == model]
                if len(model_cat_df) > 0:
                    model_rate = model_cat_df['contains_refusal'].mean()
                    model_rates.append(f"{model}: {model_rate:.1%}")
            
            category_comparison += f"""
            <tr>
                <td>{category}</td>
                <td>{overall_rate:.1%}</td>
                <td>{', '.join(model_rates) if model_rates else 'N/A'}</td>
            </tr>
            """
    
    # Get key findings
    top_findings = []
    
    # Model comparison
    if len(models) > 1:
        model_refusal = df.groupby('model')['contains_refusal'].mean()
        max_model = model_refusal.idxmax()
        min_model = model_refusal.idxmin()
        diff = model_refusal.max() - model_refusal.min()
        
        if diff > 0.1:
            top_findings.append(f"Largest model difference: {max_model} has {diff:.1%} higher refusal rate than {min_model}")
    
    # Category findings
    if len(categories) > 0:
        cat_refusal = df.groupby('category')['contains_refusal'].mean().sort_values(ascending=False)
        if len(cat_refusal) > 0:
            top_cat = cat_refusal.index[0]
            top_cat_rate = cat_refusal.iloc[0]
            top_findings.append(f"Highest refusal rate category: {top_cat} ({top_cat_rate:.1%})")
    
    findings_html = ""
    for finding in top_findings:
        findings_html += f"<li>{finding}</li>\n"
    
    # Check which visualizations exist
    visualizations = []
    for file in os.listdir(vis_dir):
        if file.endswith('.png'):
            visualizations.append(file)
    
    print(f"Found visualizations: {visualizations}")
    
    # Create visualization sections
    visualization_sections = ""
    
    if 'model_response_categories.png' in visualizations:
        visualization_sections += """
        <h2>Model Comparison</h2>
        <div class="visualization-group">
            <div class="visualization-item">
                <h3>Response Categories by Model</h3>
                <img src="visualizations/model_response_categories.png" alt="Response Categories" class="viz-image">
            </div>
        </div>
        """
    
    if 'category_refusal_rates.png' in visualizations:
        visualization_sections += """
        <h2>Category Analysis</h2>
        <div class="visualization-group">
            <div class="visualization-item">
                <h3>Refusal Rates by Category</h3>
                <img src="visualizations/category_refusal_rates.png" alt="Category Refusal Rates" class="viz-image">
            </div>
        </div>
        """
    
    if 'risk_level_refusal_rates.png' in visualizations:
        visualization_sections += """
        <h2>Risk Level Analysis</h2>
        <div class="visualization-group">
            <div class="visualization-item">
                <h3>Refusal Rates by Risk Level</h3>
                <img src="visualizations/risk_level_refusal_rates.png" alt="Risk Level Refusal Rates" class="viz-image">
            </div>
        </div>
        """
    
    # Variation analysis
    variation_vis = [v for v in visualizations if '_refusal_progression.png' in v and 'sequential' not in v]
    if variation_vis:
        visualization_sections += """
        <h2>Variation Dimension Analysis</h2>
        <div class="visualization-group">
        """
        
        for vis in variation_vis:
            var_type = vis.split('_refusal')[0]
            visualization_sections += f"""
            <div class="visualization-item">
                <h3>{var_type.title()} Dimension Analysis</h3>
                <img src="visualizations/{vis}" alt="{var_type} Analysis" class="viz-image">
            </div>
            """
        
        visualization_sections += "</div>"
    
    if 'sequential_refusal_progression.png' in visualizations:
        visualization_sections += """
        <h2>Sequential Refinement Analysis</h2>
        <div class="visualization-group">
            <div class="visualization-item">
                <h3>Refusal Rate Progression in Sequential Tests</h3>
                <img src="visualizations/sequential_refusal_progression.png" alt="Sequential Progression" class="viz-image">
            </div>
        </div>
        """
    
    # Create the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>LLM Guardrail Benchmark Results</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #fafafa;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            .summary-box {{
                background-color: #ffffff;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .findings-box {{
                background-color: #ffffff;
                border-left: 4px solid #27ae60;
                padding: 20px;
                margin-bottom: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
                background-color: #ffffff;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }}
            th {{
                background-color: #34495e;
                color: white;
                font-weight: 600;
            }}
            tr:hover {{
                background-color: #f8f9fa;
            }}
            .visualization-group {{
                display: flex;
                flex-wrap: wrap;
                gap: 30px;
                margin-bottom: 40px;
            }}
            .visualization-item {{
                flex: 1 1 45%;
                min-width: 500px;
                background-color: #ffffff;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                padding: 20px;
                border-radius: 12px;
            }}
            .viz-image {{
                width: 100%;
                height: auto;
                border-radius: 8px;
            }}
            footer {{
                margin-top: 60px;
                padding-top: 30px;
                border-top: 2px solid #ecf0f1;
                font-size: 0.9em;
                color: #7f8c8d;
                text-align: center;
            }}
            .metric {{
                display: inline-block;
                margin: 0 15px;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 6px;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <h1>🛡️ LLM Guardrail Benchmark Results</h1>
        
        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            <div>
                <span class="metric">Models: {len(models)}</span>
                <span class="metric">Tests: {test_count}</span>
                <span class="metric">Categories: {len([c for c in categories if pd.notna(c)]) if len(categories) > 0 else 0}</span>
                <span class="metric">Overall Refusal Rate: {overall_refusal:.1%}</span>
            </div>
            <p><strong>Models Tested:</strong> {', '.join(models)}</p>
            {f'<p><strong>Categories:</strong> {", ".join([c for c in categories if pd.notna(c)])}</p>' if len(categories) > 0 else ''}
        </div>
        
        <div class="findings-box">
            <h2>🔍 Key Findings</h2>
            <ul>
                {findings_html if findings_html else "<li>Analysis complete - review detailed results below</li>"}
            </ul>
        </div>
        
        <h2>📈 Model Performance Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Refusal Rate</th>
                <th>Complete Refusals</th>
                <th>Refusals with Alternatives</th>
                <th>Cautious Compliance</th>
                <th>Full Compliance</th>
            </tr>
            {model_comparison}
        </table>
        
        {f'''
        <h2>📂 Category Analysis</h2>
        <table>
            <tr>
                <th>Category</th>
                <th>Refusal Rate</th>
                <th>Model Breakdown</th>
            </tr>
            {category_comparison}
        </table>
        ''' if category_comparison else ''}
        
        {visualization_sections}
        
        <footer>
            <p>Generated by LLM Guardrail Benchmark Framework</p>
            <p>🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = os.path.join(run_dir, "report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Report generation complete!")
    print(f"📄 HTML report: {report_path}")
    print(f"🖼️  Visualizations: {vis_dir}")
    print(f"🌐 Open the report in your browser: file://{os.path.abspath(report_path)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
        print(f"🚀 Generating report for: {run_dir}")
        try:
            generate_html_report(run_dir)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Please provide the run directory path as an argument.")
        print("Example: python3.11 generate_html_report_fixed.py results/run_20250521-173746")