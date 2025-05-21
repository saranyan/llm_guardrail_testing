import os
import json
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Seaborn style
sns.set_theme(style="whitegrid")

def generate_html_report(run_dir: str):
    """Generate HTML report with visualizations for a test run"""
    # Create visualizations directory
    vis_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load the data
    analytics_path = os.path.join(run_dir, "analytics.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    
    if not os.path.exists(analytics_path):
        raise FileNotFoundError(f"Analytics file not found: {analytics_path}")
    
    # Load the data
    df = pd.read_csv(analytics_path)
    
    # Load summary if available
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    # Generate visualizations
    
    # 1. Response categories by model
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
    plt.savefig(os.path.join(vis_dir, 'model_response_categories.png'), dpi=300)
    plt.close()
    
    # 2. Refusal rates by category and model (if category field exists)
    if 'category' in df.columns:
        plt.figure(figsize=(14, 8))
        category_model_refusal = df.groupby(['category', 'model'])['contains_refusal'].mean().unstack()
        
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
            
            # Add percentage labels
            for container in ax.containers:
                labels = [f'{v:.0%}' if v > 0.05 else '' for v in container]
                ax.bar_label(container, labels=labels, label_type='center', fontsize=9)
                
            plt.savefig(os.path.join(vis_dir, 'category_refusal_rates.png'), dpi=300)
    plt.close()
    
    # 3. Risk level analysis (if risk_level field exists)
    if 'risk_level' in df.columns and len(df['risk_level'].unique()) > 1:
        plt.figure(figsize=(10, 6))
        risk_model_refusal = df.groupby(['risk_level', 'model'])['contains_refusal'].mean().unstack()
        
        ax = risk_model_refusal.plot(kind='bar', figsize=(10, 6))
        plt.title('Refusal Rates by Risk Level and Model', fontsize=16)
        plt.xlabel('Risk Level', fontsize=14)
        plt.ylabel('Refusal Rate', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.legend(title='Model')
        
        # Add percentage labels
        for container in ax.containers:
            labels = [f'{v:.0%}' if v > 0.05 else '' for v in container]
            ax.bar_label(container, labels=labels, label_type='center', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'risk_level_refusal_rates.png'), dpi=300)
        plt.close()
    
    # 4. Variation analysis (if variation_type field exists)
    if 'variation_type' in df.columns:
        variation_types = [vt for vt in df['variation_type'].unique() 
                         if vt not in ('baseline', 'sequential', 'unknown') and pd.notna(vt)]
        
        for var_type in variation_types:
            var_df = df[(df['variation_type'] == var_type) & (df['variation_level'] > 0)]
            
            if len(var_df) < 3:  # Skip if not enough data
                continue
                
            plt.figure(figsize=(12, 7))
            
            # Group by level and model
            level_model_refusal = var_df.groupby(['variation_level', 'model'])['contains_refusal'].mean().unstack()
            
            if len(level_model_refusal) < 2:  # Skip if not enough levels
                continue
                
            # Plot line chart
            ax = level_model_refusal.plot(kind='line', marker='o', figsize=(12, 7), linewidth=3)
            plt.title(f'Refusal Rates by {var_type.title()} Level and Model', fontsize=16)
            plt.xlabel(f'{var_type.title()} Level', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            
            # Set integer x-ticks
            plt.xticks(level_model_refusal.index)
            
            # Add markers at each data point
            for line in ax.lines:
                ax.scatter(line.get_xdata(), line.get_ydata(), s=100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f'{var_type}_refusal_progression.png'), dpi=300)
            plt.close()
    
    # 5. Sequential progression if available
    if 'sequence_position' in df.columns:
        seq_df = df[df['sequence_position'] > 0]
        if len(seq_df) >= 5:
            plt.figure(figsize=(12, 7))
            
            # Group by position and model
            position_model_refusal = seq_df.groupby(['sequence_position', 'model'])['contains_refusal'].mean().unstack()
            
            # Plot line chart
            ax = position_model_refusal.plot(kind='line', marker='o', figsize=(12, 7), linewidth=3)
            plt.title('Refusal Rate Progression in Sequential Tests', fontsize=16)
            plt.xlabel('Sequence Position', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            
            # Set integer x-ticks
            plt.xticks(position_model_refusal.index)
            
            # Add markers at each data point
            for line in ax.lines:
                ax.scatter(line.get_xdata(), line.get_ydata(), s=100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'sequential_refusal_progression.png'), dpi=300)
            plt.close()
    
    # Build the rest of the HTML report (this part should work fine once the plots are created)
    # Just call the script with the run directory:
    #
    # python fix-script.py path/to/your/results/run_directory
    
    print(f"Visualizations created in {vis_dir}")
    print(f"Now you can manually open the report at: {os.path.join(run_dir, 'report.html')}")


if __name__ == "__main__":
    if len(os.sys.argv) > 1:
        run_dir = os.sys.argv[1]
        print(f"Generating report for: {run_dir}")
        generate_html_report(run_dir)
    else:
        print("Please provide the run directory path as an argument.")
        print("Example: python generate_html_from_run.py results/run_20250521-123456")