#!/usr/bin/env python3
"""
Visualization tool for LLM Guardrail Benchmark results
"""

import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path


class BenchmarkVisualizer:
    """Visualizes LLM Guardrail Benchmark results"""
    
    def __init__(self, run_dir):
        """Initialize with the results directory path"""
        self.run_dir = run_dir
        self.results_data = None
        self.analytics_df = None
        self.summary = None
        self.metadata = None
        self.output_dir = os.path.join(run_dir, 'visualizations')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load all data sources from the run directory"""
        # Load analytics CSV
        analytics_path = os.path.join(self.run_dir, 'analytics.csv')
        if os.path.exists(analytics_path):
            self.analytics_df = pd.read_csv(analytics_path)
            print(f"Loaded analytics data: {len(self.analytics_df)} rows")
        else:
            print(f"Warning: Analytics file not found at {analytics_path}")
        
        # Load JSON summary
        summary_path = os.path.join(self.run_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            print(f"Loaded summary data")
        else:
            print(f"Warning: Summary file not found at {summary_path}")
        
        # Load metadata
        metadata_path = os.path.join(self.run_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded run metadata")
        else:
            print(f"Warning: Metadata file not found at {metadata_path}")
        
        # Load all results
        results_path = os.path.join(self.run_dir, 'all_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                self.results_data = json.load(f)
            print(f"Loaded raw results data: {len(self.results_data)} tests")
        else:
            print(f"Warning: Results file not found at {results_path}")
    
    def generate_visualizations(self):
        """Generate all visualizations"""
        if self.analytics_df is None:
            print("Error: Analytics data not loaded. Cannot generate visualizations.")
            return
        
        print("Generating visualizations...")
        
        # Create model comparison plots
        self.plot_model_comparison()
        
        # Create category refusal rates
        self.plot_category_refusal_rates()
        
        # Create variation analysis plots
        self.plot_variation_analysis()
        
        # Create sequential refusal progression
        self.plot_sequential_progression()
        
        # Create risk level analysis
        self.plot_risk_level_analysis()
        
        # Generate HTML report
        self.generate_html_report()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def plot_model_comparison(self):
        """Plot model comparison metrics"""
        print("Creating model comparison plots...")
        
        # Filter out error responses if any
        df = self.analytics_df[self.analytics_df['classification'] != 'error']
        
        # Plot 1: Response categories by model
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
        plt.savefig(os.path.join(self.output_dir, 'model_response_categories.png'), dpi=300)
        plt.close()
        
        # Plot 2: Overall refusal rates by model
        plt.figure(figsize=(10, 6))
        refusal_by_model = df.groupby('model')['contains_refusal'].mean()
        ax = refusal_by_model.plot(kind='bar', figsize=(10, 6), color=sns.color_palette("Set2"))
        plt.title('Overall Refusal Rate by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Refusal Rate', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        
        # Add percentage labels on top of bars
        for i, v in enumerate(refusal_by_model):
            ax.text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_refusal_rates.png'), dpi=300)
        plt.close()
    
    def plot_category_refusal_rates(self):
        """Plot refusal rates by category"""
        print("Creating category analysis plots...")
        
        df = self.analytics_df
        
        # Avoid duplicate rows from sequential tests by using only distinct test_id/model combinations
        if 'sequence_position' in df.columns:
            # For sequential tests, use only the last position to represent the final outcome
            seq_df = df[df['sequence_position'] > 0].copy()
            if not seq_df.empty:
                max_positions = seq_df.groupby(['test_id', 'model'])['sequence_position'].transform('max')
                seq_df = seq_df[seq_df['sequence_position'] == max_positions]
                
                # Combine with non-sequential tests
                non_seq_df = df[df['sequence_position'] == 0].copy()
                df = pd.concat([non_seq_df, seq_df])
            
        # Plot: Refusal rates by category and model
        plt.figure(figsize=(14, 8))
        category_model_refusal = df.groupby(['category', 'model'])['contains_refusal'].mean().unstack()
        
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
        
        plt.savefig(os.path.join(self.output_dir, 'category_refusal_rates.png'), dpi=300)
        plt.close()
        
        # If we have subcategories, plot them too
        if len(df['subcategory'].unique()) > 1:
            # Get top subcategories by count
            subcategory_counts = df['subcategory'].value_counts()
            top_subcategories = subcategory_counts[subcategory_counts >= 3].index.tolist()
            
            if top_subcategories:
                plt.figure(figsize=(16, 10))
                
                # Filter for top subcategories
                subcat_df = df[df['subcategory'].isin(top_subcategories)]
                
                subcat_model_refusal = subcat_df.groupby(['subcategory', 'model'])['contains_refusal'].mean().unstack()
                
                ax = subcat_model_refusal.plot(kind='bar', figsize=(16, 10))
                plt.title('Refusal Rates by Subcategory and Model', fontsize=16)
                plt.xlabel('Subcategory', fontsize=14)
                plt.ylabel('Refusal Rate', fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.ylim(0, 1)
                plt.legend(title='Model')
                plt.tight_layout()
                
                plt.savefig(os.path.join(self.output_dir, 'subcategory_refusal_rates.png'), dpi=300)
                plt.close()
    
    def plot_variation_analysis(self):
        """Plot analysis of variation dimensions"""
        print("Creating variation analysis plots...")
        
        df = self.analytics_df
        
        # Get unique variation types excluding 'baseline' and 'sequential'
        variation_types = [vt for vt in df['variation_type'].unique() 
                          if vt not in ('baseline', 'sequential', 'unknown') and pd.notna(vt)]
        
        for var_type in variation_types:
            # Try to get data from summary if available
            if (self.summary and "variation_analysis" in self.summary and 
                var_type in self.summary["variation_analysis"] and 
                "refusal_by_level" in self.summary["variation_analysis"][var_type]):
                
                refusal_data = self.summary["variation_analysis"][var_type]["refusal_by_level"]
                level_data = {}
                
                # Parse the string keys back into components
                for key, value in refusal_data.items():
                    parts = key.split('_')
                    if len(parts) >= 2:
                        level = int(parts[0])
                        model = '_'.join(parts[1:])  # In case model has underscores
                        if level not in level_data:
                            level_data[level] = {}
                        level_data[level][model] = value
                
                # Convert to DataFrame
                levels = sorted(level_data.keys())
                models = list(sorted(set().union(*[d.keys() for d in level_data.values()])))
                
                data = []
                for level in levels:
                    for model in models:
                        if model in level_data.get(level, {}):
                            data.append({
                                'level': level,
                                'model': model,
                                'refusal_rate': level_data[level][model]
                            })
                
                if len(data) < 3:  # Skip if not enough data
                    continue
                
                level_model_df = pd.DataFrame(data)
                level_model_refusal = level_model_df.pivot(index='level', columns='model', values='refusal_rate')
                
            else:
                # Filter for this variation type from analytics data
                var_df = df[(df['variation_type'] == var_type) & (df['variation_level'] > 0)]
                
                if len(var_df) < 3:  # Skip if not enough data
                    continue
                    
                # Group by level and model
                level_model_refusal = var_df.groupby(['variation_level', 'model'])['contains_refusal'].mean().unstack()
                
                if len(level_model_refusal) < 2:  # Skip if not enough levels
                    continue
            
            plt.figure(figsize=(12, 7))
            
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
            plt.savefig(os.path.join(self.output_dir, f'{var_type}_refusal_progression.png'), dpi=300)
            plt.close()
    
    def plot_sequential_progression(self):
        """Plot progression of refusals in sequential tests"""
        print("Creating sequential progression plots...")
        
        df = self.analytics_df
        
        # Filter for sequential tests
        seq_df = df[df['sequence_position'] > 0]
        
        if len(seq_df) < 5:  # Skip if not enough sequential data
            print("Not enough sequential test data for visualization")
            return
            
        plt.figure(figsize=(12, 7))
        
        # Group by position and model
        # We need to modify how we read the data from summary if it's in string format
        if self.summary and "variation_analysis" in self.summary and "sequential" in self.summary["variation_analysis"]:
            # Get data from summary instead
            refusal_data = self.summary["variation_analysis"]["sequential"]["refusal_by_position"]
            position_data = {}
            
            # Parse the string keys back into components
            for key, value in refusal_data.items():
                parts = key.split('_')
                if len(parts) >= 2:
                    position = int(parts[0])
                    model = '_'.join(parts[1:])  # In case model has underscores
                    if position not in position_data:
                        position_data[position] = {}
                    position_data[position][model] = value
            
            # Convert to DataFrame
            positions = sorted(position_data.keys())
            models = list(sorted(set().union(*[d.keys() for d in position_data.values()])))
            
            data = []
            for position in positions:
                for model in models:
                    if model in position_data.get(position, {}):
                        data.append({
                            'position': position,
                            'model': model,
                            'refusal_rate': position_data[position][model]
                        })
            
            if data:
                position_model_df = pd.DataFrame(data)
                position_model_refusal = position_model_df.pivot(index='position', columns='model', values='refusal_rate')
                
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
                plt.savefig(os.path.join(self.output_dir, 'sequential_refusal_progression.png'), dpi=300)
                plt.close()
            
        else:
            # Use the analytics data directly
            position_model_refusal = seq_df.groupby(['sequence_position', 'model'])['contains_refusal'].mean().unstack()
            
            if len(position_model_refusal) > 0:
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
                plt.savefig(os.path.join(self.output_dir, 'sequential_refusal_progression.png'), dpi=300)
                plt.close()
        
        # Plot by category if we have multiple categories
        categories = seq_df['category'].unique()
        if len(categories) > 1:
            for category in categories:
                cat_seq_df = seq_df[seq_df['category'] == category]
                
                if len(cat_seq_df) < 5:  # Skip if not enough data
                    continue
                    
                plt.figure(figsize=(12, 7))
                
                # Group by position and model
                cat_position_model_refusal = cat_seq_df.groupby(['sequence_position', 'model'])['contains_refusal'].mean().unstack()
                
                if len(cat_position_model_refusal) < 2:  # Skip if not enough positions
                    continue
                
                # Plot line chart
                ax = cat_position_model_refusal.plot(kind='line', marker='o', figsize=(12, 7), linewidth=3)
                plt.title(f'Refusal Rate Progression in {category} Sequential Tests', fontsize=16)
                plt.xlabel('Sequence Position', fontsize=14)
                plt.ylabel('Refusal Rate', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim(0, 1)
                
                # Set integer x-ticks
                plt.xticks(cat_position_model_refusal.index)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, f'{category.lower().replace(" ", "_")}_sequential_progression.png'), dpi=300)
                plt.close()
    
    def plot_risk_level_analysis(self):
        """Plot refusal rates by risk level"""
        print("Creating risk level analysis plots...")
        
        df = self.analytics_df
        
        if 'risk_level' not in df.columns or len(df['risk_level'].unique()) <= 1:
            print("Not enough risk level data for visualization")
            return
            
        # Try to get data from summary if available
        if self.summary and "risk_level_analysis" in self.summary:
            refusal_data = self.summary["risk_level_analysis"]
            risk_data = {}
            
            # Parse the string keys back into components
            for key, value in refusal_data.items():
                parts = key.split('_')
                if len(parts) >= 2:
                    risk = parts[0]
                    model = '_'.join(parts[1:])  # In case model has underscores
                    if risk not in risk_data:
                        risk_data[risk] = {}
                    risk_data[risk][model] = value
            
            # Convert to DataFrame
            risks = sorted(risk_data.keys())
            models = list(sorted(set().union(*[d.keys() for d in risk_data.values()])))
            
            data = []
            for risk in risks:
                for model in models:
                    if model in risk_data.get(risk, {}):
                        data.append({
                            'risk_level': risk,
                            'model': model,
                            'refusal_rate': risk_data[risk][model]
                        })
            
            if data:
                risk_model_df = pd.DataFrame(data)
                risk_model_refusal = risk_model_df.pivot(index='risk_level', columns='model', values='refusal_rate')
                
                plt.figure(figsize=(10, 6))
                # Plot bar chart
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
                    ax.bar_label(container, labels=labels, label_type='center', padding=3)
                    
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'risk_level_refusal_rates.png'), dpi=300)
                plt.close()
                
                # Skip category-risk analysis if we're using summary data
                return
        
        # Use analytics data directly if summary data not available or incomplete
        plt.figure(figsize=(10, 6))
        
        # Group by risk level and model
        risk_model_refusal = df.groupby(['risk_level', 'model'])['contains_refusal'].mean().unstack()
        
        # Plot bar chart
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
            ax.bar_label(container, labels=labels, label_type='center', padding=3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'risk_level_refusal_rates.png'), dpi=300)
        plt.close()
        
        # Plot risk level by category
        plt.figure(figsize=(14, 8))
        
        # Group by category, risk level, and get mean refusal rate
        category_risk_refusal = df.groupby(['category', 'risk_level'])['contains_refusal'].mean().unstack()
        
        if len(category_risk_refusal) > 0:
            ax = category_risk_refusal.plot(kind='bar', figsize=(14, 8))
            plt.title('Refusal Rates by Category and Risk Level', fontsize=16)
            plt.xlabel('Category', fontsize=14)
            plt.ylabel('Refusal Rate', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.ylim(0, 1)
            plt.legend(title='Risk Level')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'category_risk_refusal_rates.png'), dpi=300)
            plt.close()
    
    def generate_html_report(self):
        """Generate HTML report with all visualizations and insights"""
        print("Generating HTML report...")
        
        # Get available visualizations
        visualizations = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.png'):
                visualizations.append(file)
        
        # Get metadata
        models = self.analytics_df['model'].unique() if self.analytics_df is not None else []
        test_count = len(self.analytics_df['test_id'].unique()) if self.analytics_df is not None else 0
        categories = self.analytics_df['category'].unique() if self.analytics_df is not None else []
        
        # Get overall refusal rate
        overall_refusal = self.analytics_df['contains_refusal'].mean() if self.analytics_df is not None else 0
        
        # Create model comparison table
        model_comparison = ""
        if self.analytics_df is not None:
            model_stats = []
            for model in models:
                model_df = self.analytics_df[self.analytics_df['model'] == model]
                refusal_rate = model_df['contains_refusal'].mean()
                response_types = model_df['classification'].value_counts().to_dict()
                
                model_stats.append({
                    'model': model,
                    'refusal_rate': refusal_rate,
                    'response_types': response_types
                })
            
            # Create table rows
            for stat in model_stats:
                responses = stat['response_types']
                model_comparison += f"""
                <tr>
                    <td>{stat['model']}</td>
                    <td>{stat['refusal_rate']:.1%}</td>
                    <td>{responses.get('complete_refusal', 0)}</td>
                    <td>{responses.get('refusal_with_alternative', 0)}</td>
                    <td>{responses.get('cautious_compliance', 0)}</td>
                    <td>{responses.get('full_compliance', 0)}</td>
                </tr>
                """
        
        # Create category comparison table
        category_comparison = ""
        if self.analytics_df is not None:
            for category in categories:
                cat_df = self.analytics_df[self.analytics_df['category'] == category]
                overall_rate = cat_df['contains_refusal'].mean()
                
                model_rates = []
                for model in models:
                    model_cat_df = cat_df[cat_df['model'] == model]
                    model_rate = model_cat_df['contains_refusal'].mean()
                    model_rates.append(f"{model}: {model_rate:.1%}")
                
                category_comparison += f"""
                <tr>
                    <td>{category}</td>
                    <td>{overall_rate:.1%}</td>
                    <td>{', '.join(model_rates)}</td>
                </tr>
                """
        
        # Get top findings
        top_findings = []
        
        # If we have analytical data
        if self.analytics_df is not None:
            df = self.analytics_df
            
            # Get highest refusal rate categories
            cat_refusal = df.groupby('category')['contains_refusal'].mean().sort_values(ascending=False)
            if len(cat_refusal) > 0:
                top_cat = cat_refusal.index[0]
                top_cat_rate = cat_refusal.iloc[0]
                top_findings.append(f"Highest refusal rate category: {top_cat} ({top_cat_rate:.1%})")
            
            # Compare model performance
            if len(models) > 1:
                model_refusal = df.groupby('model')['contains_refusal'].mean()
                max_model = model_refusal.idxmax()
                min_model = model_refusal.idxmin()
                diff = model_refusal.max() - model_refusal.min()
                
                if diff > 0.1:  # Only note if difference is substantial
                    top_findings.append(f"Largest model difference: {max_model} has {diff:.1%} higher refusal rate than {min_model}")
            
            # Check variation impacts
            for var_type in df['variation_type'].unique():
                if var_type not in ('baseline', 'sequential', 'unknown') and pd.notna(var_type):
                    var_df = df[df['variation_type'] == var_type]
                    if len(var_df) < 6:  # Skip if not enough data
                        continue
                        
                    # Check effect of increasing levels
                    level_refusal = var_df.groupby('variation_level')['contains_refusal'].mean()
                    if len(level_refusal) > 1:
                        level_diff = level_refusal.max() - level_refusal.min()
                        if level_diff > 0.2:  # Only note if difference is substantial
                            top_findings.append(f"Increasing {var_type} level raises refusal rate by {level_diff:.1%}")
            
            # Check sequential progression
            seq_df = df[df['sequence_position'] > 0]
            if len(seq_df) > 5:
                first_pos = seq_df[seq_df['sequence_position'] == 1]['contains_refusal'].mean()
                last_pos = seq_df[seq_df['sequence_position'] == seq_df['sequence_position'].max()]['contains_refusal'].mean()
                seq_diff = last_pos - first_pos
                
                if abs(seq_diff) > 0.2:  # Only note if difference is substantial
                    direction = "increases" if seq_diff > 0 else "decreases"
                    top_findings.append(f"Refusal rate {direction} by {abs(seq_diff):.1%} during sequential refinement")
            
            # Check risk level impact
            if 'risk_level' in df.columns and len(df['risk_level'].unique()) > 1:
                risk_refusal = df.groupby('risk_level')['contains_refusal'].mean()
                if len(risk_refusal) > 0 and (risk_refusal.max() - risk_refusal.min()) > 0.1:
                    top_findings.append(f"Risk level significantly affects refusal rates (up to {risk_refusal.max():.1%} for highest risk)")
        
        # Compile findings list
        findings_html = ""
        for finding in top_findings:
            findings_html += f"<li>{finding}</li>\n"
        
        # Create visualization sections
        visualization_sections = ""
        vis_groups = {
            'model_': 'Model Comparison',
            'category_': 'Category Analysis',
            'sequential_': 'Sequential Progression',
            'risk_': 'Risk Level Analysis'
        }
        
        for prefix, title in vis_groups.items():
            matching_vis = [v for v in visualizations if v.startswith(prefix)]
            
            if matching_vis:
                visualization_sections += f"""
                <h2>{title}</h2>
                <div class="visualization-group">
                """
                
                for vis in matching_vis:
                    vis_title = ' '.join(vis.replace('.png', '').replace('_', ' ').split()[1:]).title()
                    visualization_sections += f"""
                    <div class="visualization-item">
                        <h3>{vis_title}</h3>
                        <img src="{vis}" alt="{vis_title}" class="viz-image">
                    </div>
                    """
                
                visualization_sections += "</div>\n"
        
        # For variation analysis, create a separate section
        variation_vis = [v for v in visualizations if any(v.startswith(vt + '_') for vt in 
                         ['directness', 'technical', 'contextual', 'authority', 'role'])]
        
        if variation_vis:
            visualization_sections += """
            <h2>Variation Dimension Analysis</h2>
            <div class="visualization-group">
            """
            
            for vis in variation_vis:
                vis_title = ' '.join(vis.replace('.png', '').replace('_', ' ').title().split())
                visualization_sections += f"""
                <div class="visualization-item">
                    <h3>{vis_title}</h3>
                    <img src="{vis}" alt="{vis_title}" class="viz-image">
                </div>
                """
            
            visualization_sections += "</div>\n"
        
        # Create the HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Guardrail Benchmark Results</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #4285f4;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                .findings-box {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #34a853;
                    padding: 15px;
                    margin-bottom: 20px;
                    border-radius: 4px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .visualization-group {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .visualization-item {{
                    flex: 1 1 45%;
                    min-width: 500px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    border-radius: 4px;
                }}
                .viz-image {{
                    width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    font-size: 0.9em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <h1>LLM Guardrail Benchmark Results</h1>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Models Tested:</strong> {', '.join(models)}</p>
                <p><strong>Tests Run:</strong> {test_count}</p>
                <p><strong>Categories:</strong> {', '.join(categories)}</p>
                <p><strong>Overall Refusal Rate:</strong> {overall_refusal:.1%}</p>
            </div>
            
            <div class="findings-box">
                <h2>Key Findings</h2>
                <ul>
                    {findings_html}
                </ul>
            </div>
            
            <h2>Model Comparison</h2>
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
            
            <h2>Category Analysis</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Refusal Rate</th>
                    <th>Model Breakdown</th>
                </tr>
                {category_comparison}
            </table>
            
            {visualization_sections}
            
            <footer>
                <p>Generated by LLM Guardrail Benchmark Visualizer</p>
                <p>Report date: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </footer>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(self.output_dir, 'report.html'), 'w') as f:
            f.write(html)
        
        print(f"HTML report saved to {os.path.join(self.output_dir, 'report.html')}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='LLM Guardrail Benchmark Results Visualizer')
    parser.add_argument('--run-dir', type=str, required=True, help='Path to results run directory')
    args = parser.parse_args()
    
    # Validate path
    if not os.path.exists(args.run_dir):
        print(f"Error: Run directory {args.run_dir} not found")
        return
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(args.run_dir)
    
    # Load data
    visualizer.load_data()
    
    # Generate visualizations
    visualizer.generate_visualizations()
    
    print(f"Visualization complete! Open the report at: {os.path.join(args.run_dir, 'visualizations', 'report.html')}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(os.path.join(args.run_dir, 'visualizations', 'report.html'))}")
    except:
        pass


if __name__ == "__main__":
    main()