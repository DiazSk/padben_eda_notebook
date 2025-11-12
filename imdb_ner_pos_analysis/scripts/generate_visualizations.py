#!/usr/bin/env python
"""
Generate enhanced multi-panel visualizations for IMDB NER and POS analysis.
Creates professional dashboard-style charts similar to Claude's output.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


def create_pos_dashboard(data_dir: Path, output_path: Path):
    """Create a 4-panel POS distribution dashboard."""
    # Load data
    pos_df = pd.read_csv(data_dir / "pos_tag_distribution.csv")
    
    # Prepare data
    pos_pivot = pos_df.pivot(index="pos_tag", columns="sentiment", values="proportion")
    pos_counts = pos_df.pivot(index="pos_tag", columns="sentiment", values="count")
    
    # Calculate differences and percentages
    pos_pivot['difference'] = (pos_pivot['Positive'] - pos_pivot['Negative']) * 100
    
    # Create figure with 4 subplots with better spacing
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35, 
                          top=0.93, bottom=0.05, left=0.08, right=0.98)
    
    # Define colors
    color_pos = '#5B9BD5'  # Blue for positive
    color_neg = '#ED7D31'  # Orange for negative
    color_green = '#70AD47'  # Green
    
    # Top 10 tags
    top_tags = pos_counts.sum(axis=1).nlargest(10).index
    
    # ============ Panel 1: Side-by-side comparison (Top Left) ============
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(top_tags))
    width = 0.35
    
    neg_vals = [pos_counts.loc[tag, 'Negative'] for tag in top_tags]
    pos_vals = [pos_counts.loc[tag, 'Positive'] for tag in top_tags]
    
    bars1 = ax1.bar(x - width/2, pos_vals, width, label='Positive', color=color_pos, alpha=0.8)
    bars2 = ax1.bar(x + width/2, neg_vals, width, label='Negative', color=color_neg, alpha=0.8)
    
    ax1.set_xlabel('POS Tag', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('POS Tags: Positive vs Negative Reviews', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_tags, rotation=45, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 50000:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height/1000)}k',
                        ha='center', va='bottom', fontsize=8)
    
    # ============ Panel 2: Top 10 in Positive (Top Right) ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    pos_only = pos_counts['Positive'].nlargest(10).sort_values()
    
    bars = ax2.barh(range(len(pos_only)), pos_only.values, color=color_pos, alpha=0.8)
    ax2.set_yticks(range(len(pos_only)))
    ax2.set_yticklabels(pos_only.index)
    ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 POS Tags in Positive Reviews', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(pos_only.items()):
        ax2.text(val, i, f' {int(val/1000)}k', va='center', fontsize=9)
    
    # ============ Panel 3: Top 10 in Negative (Bottom Left) ============
    ax3 = fig.add_subplot(gs[1, 0])
    
    neg_only = pos_counts['Negative'].nlargest(10).sort_values()
    
    bars = ax3.barh(range(len(neg_only)), neg_only.values, color=color_neg, alpha=0.8)
    ax3.set_yticks(range(len(neg_only)))
    ax3.set_yticklabels(neg_only.index)
    ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Top 10 POS Tags in Negative Reviews', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(neg_only.items()):
        ax3.text(val, i, f' {int(val/1000)}k', va='center', fontsize=9)
    
    # ============ Panel 4: Normalized Percentage Distribution (Bottom Right) ============
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate percentages within each sentiment
    all_tags = pos_pivot.index
    x = np.arange(len(all_tags))
    width = 0.35
    
    neg_pct = pos_pivot['Negative'] * 100
    pos_pct = pos_pivot['Positive'] * 100
    
    bars1 = ax4.bar(x - width/2, pos_pct, width, label='Positive', color=color_pos, alpha=0.8)
    bars2 = ax4.bar(x + width/2, neg_pct, width, label='Negative', color=color_neg, alpha=0.8)
    
    ax4.set_xlabel('POS Tag', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax4.set_title('POS Tags: Normalized Distribution', fontsize=13, fontweight='bold', pad=15)
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_tags, rotation=45, ha='right', fontsize=9)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    
    # Main title
    fig.suptitle('Part-of-Speech (POS) Tagging Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Enhanced POS dashboard saved: {output_path}")


def create_ner_dashboard(data_dir: Path, output_path: Path):
    """Create a 4-panel NER distribution dashboard."""
    # Load data
    ner_df = pd.read_csv(data_dir / "ner_label_distribution.csv")
    
    # Prepare data
    ner_pivot = ner_df.pivot(index="entity_label", columns="sentiment", values="proportion")
    ner_counts = ner_df.pivot(index="entity_label", columns="sentiment", values="count")
    
    # Calculate differences
    ner_pivot['difference'] = (ner_pivot['Positive'] - ner_pivot['Negative']) * 100
    
    # Create figure with 4 subplots with better spacing
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35, 
                          top=0.93, bottom=0.05, left=0.08, right=0.98)
    
    # Define colors
    color_pos = '#70AD47'  # Green for positive
    color_neg = '#E67C73'  # Red/coral for negative
    
    # Top entities
    top_entities = ner_counts.sum(axis=1).nlargest(12).index
    
    # ============ Panel 1: Side-by-side comparison (Top Left) ============
    ax1 = fig.add_subplot(gs[0, 0])
    
    x = np.arange(len(top_entities))
    width = 0.35
    
    neg_vals = [ner_counts.loc[ent, 'Negative'] for ent in top_entities]
    pos_vals = [ner_counts.loc[ent, 'Positive'] for ent in top_entities]
    
    bars1 = ax1.bar(x - width/2, pos_vals, width, label='Positive', color=color_pos, alpha=0.8)
    bars2 = ax1.bar(x + width/2, neg_vals, width, label='Negative', color=color_neg, alpha=0.8)
    
    ax1.set_xlabel('Entity Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('NER Tags: Positive vs Negative Reviews', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_entities, rotation=45, ha='right')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on taller bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 10000:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height/1000)}k',
                        ha='center', va='bottom', fontsize=8)
    
    # ============ Panel 2: Top 10 in Positive (Top Right) ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    pos_only = ner_counts['Positive'].nlargest(10).sort_values()
    
    bars = ax2.barh(range(len(pos_only)), pos_only.values, color=color_pos, alpha=0.8)
    ax2.set_yticks(range(len(pos_only)))
    ax2.set_yticklabels(pos_only.index)
    ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 NER Tags in Positive Reviews', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(pos_only.items()):
        ax2.text(val, i, f' {int(val/1000) if val > 1000 else int(val)}{"k" if val > 1000 else ""}', 
                va='center', fontsize=9)
    
    # ============ Panel 3: Top 10 in Negative (Bottom Left) ============
    ax3 = fig.add_subplot(gs[1, 0])
    
    neg_only = ner_counts['Negative'].nlargest(10).sort_values()
    
    bars = ax3.barh(range(len(neg_only)), neg_only.values, color=color_neg, alpha=0.8)
    ax3.set_yticks(range(len(neg_only)))
    ax3.set_yticklabels(neg_only.index)
    ax3.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Top 10 NER Tags in Negative Reviews', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, val) in enumerate(neg_only.items()):
        ax3.text(val, i, f' {int(val/1000) if val > 1000 else int(val)}{"k" if val > 1000 else ""}', 
                va='center', fontsize=9)
    
    # ============ Panel 4: Sentiment Preference (Bottom Right) ============
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate preference (positive - negative as percentage)
    differences = ner_pivot['difference'].sort_values()
    
    colors = [color_neg if x < 0 else color_pos for x in differences.values]
    
    bars = ax4.barh(range(len(differences)), differences.values, color=colors, alpha=0.8)
    ax4.set_yticks(range(len(differences)))
    ax4.set_yticklabels(differences.index, fontsize=9)
    ax4.set_xlabel('% Difference (Positive - Negative)', fontsize=11, fontweight='bold')
    ax4.set_title('NER Tags: Sentiment Preference', fontsize=13, fontweight='bold', pad=15)
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add value labels for significant differences
    for i, (idx, val) in enumerate(differences.items()):
        if abs(val) > 2:
            ax4.text(val, i, f' {val:.1f}%', va='center', fontsize=8, 
                    ha='left' if val > 0 else 'right')
    
    # Main title
    fig.suptitle('Named Entity Recognition (NER) Distribution Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Enhanced NER dashboard saved: {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating enhanced visualizations...")
    print("-" * 60)
    
    # Create POS dashboard
    pos_output = figures_dir / "pos_tag_distribution_enhanced.png"
    create_pos_dashboard(data_dir, pos_output)
    
    # Create NER dashboard
    ner_output = figures_dir / "ner_label_distribution_enhanced.png"
    create_ner_dashboard(data_dir, ner_output)
    
    print("-" * 60)
    print("[SUCCESS] All enhanced visualizations generated successfully!")
    print(f"\nOutputs saved in: {figures_dir}")


if __name__ == "__main__":
    main()

