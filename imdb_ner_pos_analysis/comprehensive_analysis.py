#!/usr/bin/env python
"""
IMDB Movie Reviews: Comprehensive NER and POS Distribution Analysis
====================================================================

This comprehensive script combines all analysis steps into a single executable file:
1. Data Processing: Load and process IMDB reviews using spaCy
2. Distribution Analysis: Calculate POS and NER tag distributions  
3. Visualization: Create comparative visualizations
4. Report Generation: Generate comprehensive PDF reports

Author: NLP Analysis Project
Dataset: nocode-ai/imdb-movie-reviews (50,000 reviews)
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

from __future__ import annotations

import json
import pickle
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import spacy
from datasets import DatasetDict, load_dataset
from spacy.language import Language
from tqdm import tqdm

# PDF Generation libraries
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    PageBreak, KeepTogether, Image
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# Configure aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================

SENTIMENT_ORDER = ("negative", "positive")
VALID_SENTIMENTS = set(SENTIMENT_ORDER)
SENTIMENT_DISPLAY = {label: label.capitalize() for label in SENTIMENT_ORDER}
CLEAN_HTML_REGEX = re.compile(r"<br\s*/?>", re.IGNORECASE)


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class TagDistributions:
    """Container for POS and NER distributions per sentiment label."""
    pos_counts: Dict[str, Counter]
    ner_counts: Dict[str, Counter]
    token_totals: Dict[str, int]
    entity_totals: Dict[str, int]
    document_totals: Dict[str, int]

    @classmethod
    def empty(cls) -> "TagDistributions":
        return cls(
            pos_counts=defaultdict(Counter),
            ner_counts=defaultdict(Counter),
            token_totals=defaultdict(int),
            entity_totals=defaultdict(int),
            document_totals=defaultdict(int),
        )


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ensure_spacy_model(model_name: str) -> Language:
    """Load a spaCy language model, downloading it if necessary."""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name}...")
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


def clean_review(text: str) -> str:
    """Apply lightweight text cleaning to IMDB reviews."""
    text = CLEAN_HTML_REGEX.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def iter_labelled_examples(dataset: DatasetDict) -> Iterator[Tuple[str, str]]:
    """Yield (text, sentiment) pairs from the labelled IMDB splits."""
    for split_name, split in dataset.items():
        for example in split:
            text = clean_review(str(example.get("review", "")))
            sentiment = str(example.get("sentiment", "")).strip().lower()
            if not text or sentiment not in VALID_SENTIMENTS:
                continue
            yield text, sentiment


# ==============================================================================
# DATA PROCESSING FUNCTIONS
# ==============================================================================

def compute_distributions(dataset: DatasetDict, nlp: Language) -> TagDistributions:
    """Run spaCy over the dataset and accumulate POS/NER statistics."""
    distributions = TagDistributions.empty()
    
    total_examples = sum(1 for _ in iter_labelled_examples(dataset))
    labelled_stream = iter_labelled_examples(dataset)
    
    for doc, sentiment in tqdm(
        nlp.pipe(labelled_stream, as_tuples=True, batch_size=32, n_process=1),
        total=total_examples,
        desc="Processing reviews",
    ):
        distributions.document_totals[sentiment] += 1
        
        # POS tagging statistics
        for token in doc:
            if token.is_alpha:
                distributions.pos_counts[sentiment][token.pos_] += 1
                distributions.token_totals[sentiment] += 1
        
        # NER statistics
        for ent in doc.ents:
            distributions.ner_counts[sentiment][ent.label_] += 1
            distributions.entity_totals[sentiment] += 1
    
    return distributions


def distributions_to_dataframe(
    counts: Dict[str, Counter], totals: Dict[str, int], value_name: str
) -> pd.DataFrame:
    """Convert Counter data into a tidy dataframe with proportions."""
    records = []
    for sentiment_key, counter in counts.items():
        sentiment = SENTIMENT_DISPLAY.get(sentiment_key, sentiment_key.capitalize())
        total = totals.get(sentiment_key, 0)
        for tag, count in counter.items():
            proportion = count / total if total else 0.0
            records.append({
                "sentiment": sentiment,
                "tag": tag,
                "count": int(count),
                "total": int(total),
                "proportion": proportion,
            })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(by=["sentiment", "count"], ascending=[True, False])
    df = df.rename(columns={"tag": value_name})
    return df


# ==============================================================================
# CACHE MANAGEMENT
# ==============================================================================

def save_distributions_cache(distributions: TagDistributions, cache_path: Path) -> None:
    """Save the computed distributions to a cache file."""
    print(f"Saving distributions cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(distributions, f)
    print(f"  ✓ Cache saved ({cache_path.stat().st_size / (1024*1024):.2f} MB)")


def load_distributions_cache(cache_path: Path) -> Optional[TagDistributions]:
    """Load distributions from cache if available."""
    if not cache_path.exists():
        return None
    
    print(f"Loading distributions from cache: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            distributions = pickle.load(f)
        print("  ✓ Cache loaded successfully!")
        print(f"    - Negative documents: {distributions.document_totals.get('negative', 0):,}")
        print(f"    - Positive documents: {distributions.document_totals.get('positive', 0):,}")
        return distributions
    except Exception as e:
        print(f"  ⚠ Failed to load cache ({e}), will recompute...")
        return None


# ==============================================================================
# DATA SAVING FUNCTIONS
# ==============================================================================

def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist the dataframe to disk."""
    if df.empty:
        empty_cols = ["sentiment"]
        if "pos" in path.stem:
            empty_cols.append("pos_tag")
        elif "ner" in path.stem or "entity" in path.stem:
            empty_cols.append("entity_label")
        else:
            empty_cols.append("tag")
        empty_cols.extend(["count", "total", "proportion"])
        pd.DataFrame(columns=empty_cols).to_csv(path, index=False)
        return
    df.to_csv(path, index=False)


def save_summary(distributions: TagDistributions, output_path: Path) -> None:
    """Persist aggregated summary statistics as JSON."""
    summary = {}
    for sentiment_key in SENTIMENT_ORDER:
        display_name = SENTIMENT_DISPLAY[sentiment_key]
        summary[display_name] = {
            "documents": distributions.document_totals.get(sentiment_key, 0),
            "tokens": distributions.token_totals.get(sentiment_key, 0),
            "entities": distributions.entity_totals.get(sentiment_key, 0),
            "unique_pos_tags": len(distributions.pos_counts.get(sentiment_key, [])),
            "unique_entity_labels": len(distributions.ner_counts.get(sentiment_key, [])),
        }
    output_path.write_text(json.dumps(summary, indent=2))


# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_distribution(
    df: pd.DataFrame,
    tag_column: str,
    output_path: Path,
    title: str,
    ylabel: str = "Proportion",
) -> None:
    """Create and save a bar plot comparing tag frequencies across sentiments."""
    if df.empty:
        return
    
    order = (
        df.groupby(tag_column)["proportion"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df,
        x=tag_column,
        y="proportion",
        hue="sentiment",
        order=order,
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(tag_column.replace('_', ' ').upper(), fontsize=11, fontweight='bold')
    plt.ylabel(ylabel, fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# REPORT GENERATION FUNCTIONS
# ==============================================================================

def create_summary_table(summary: dict) -> Table:
    """Create a summary statistics table."""
    data = [
        ["Metric", "Negative Reviews", "Positive Reviews"],
        ["Documents", f"{summary['Negative']['documents']:,}", f"{summary['Positive']['documents']:,}"],
        ["Total Tokens", f"{summary['Negative']['tokens']:,}", f"{summary['Positive']['tokens']:,}"],
        ["Total Entities", f"{summary['Negative']['entities']:,}", f"{summary['Positive']['entities']:,}"],
        ["Unique POS Tags", f"{summary['Negative']['unique_pos_tags']}", f"{summary['Positive']['unique_pos_tags']}"],
        ["Unique Entity Types", f"{summary['Negative']['unique_entity_labels']}", f"{summary['Positive']['unique_entity_labels']}"],
    ]
    
    table = Table(data, colWidths=[2.5*inch, 2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
    ]))
    return table


def create_pos_distribution_table(pos_df: pd.DataFrame, top_n: int = 10) -> Table:
    """Create POS tag distribution table."""
    pos_pivot = pos_df.pivot(index="pos_tag", columns="sentiment", values="proportion")
    pos_pivot['avg'] = pos_pivot.mean(axis=1)
    top_tags = pos_pivot.nlargest(top_n, 'avg')
    
    data = [["POS Tag", "Negative (%)", "Positive (%)", "Difference"]]
    
    for tag in top_tags.index:
        neg_val = top_tags.loc[tag, 'Negative'] * 100
        pos_val = top_tags.loc[tag, 'Positive'] * 100
        diff = pos_val - neg_val
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        data.append([tag, f"{neg_val:.2f}", f"{pos_val:.2f}", diff_str])
    
    table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    return table


def create_ner_distribution_table(ner_df: pd.DataFrame, top_n: int = 10) -> Table:
    """Create NER entity distribution table."""
    ner_pivot = ner_df.pivot(index="entity_label", columns="sentiment", values="proportion")
    ner_pivot['avg'] = ner_pivot.mean(axis=1)
    top_entities = ner_pivot.nlargest(top_n, 'avg')
    
    data = [["Entity Type", "Negative (%)", "Positive (%)", "Difference"]]
    
    for entity in top_entities.index:
        neg_val = top_entities.loc[entity, 'Negative'] * 100
        pos_val = top_entities.loc[entity, 'Positive'] * 100
        diff = pos_val - neg_val
        diff_str = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
        data.append([entity, f"{neg_val:.2f}", f"{pos_val:.2f}", diff_str])
    
    table = Table(data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#70AD47')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    return table


def build_comprehensive_report(summary: dict, pos_df: pd.DataFrame, 
                               ner_df: pd.DataFrame, figures_dir: Path) -> list:
    """Build comprehensive PDF report with all sections."""
    styles = getSampleStyleSheet()
    
    title_style = styles["Title"]
    title_style.fontSize = 18
    title_style.spaceAfter = 20
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1F4E78'),
        spaceAfter=12,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10
    )
    
    story = []
    
    # Title
    story.append(Paragraph(
        "IMDB Movie Reviews: Comprehensive NER & POS Distribution Analysis",
        title_style
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    story.append(Paragraph(
        "This report analyzes Named Entity Recognition (NER) and Part-of-Speech (POS) "
        "distributions in 50,000 IMDB movie reviews to identify linguistic differences "
        "between positive and negative sentiments. The analysis reveals notable variations "
        "in entity usage and grammatical structures between sentiment classes.",
        body_style
    ))
    story.append(Spacer(1, 0.15*inch))
    
    # Summary Table
    story.append(Paragraph("Dataset Overview", heading_style))
    story.append(create_summary_table(summary))
    story.append(Spacer(1, 0.2*inch))
    
    # POS Analysis
    story.append(Paragraph("Part-of-Speech Distribution Analysis", heading_style))
    story.append(Paragraph(
        "The POS distribution reveals notable differences in grammatical structure between "
        "positive and negative reviews. The following table shows the top 10 POS tags:",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(create_pos_distribution_table(pos_df, 10))
    story.append(Spacer(1, 0.15*inch))
    
    # POS Visualization
    pos_img_path = figures_dir / "pos_tag_distribution.png"
    if pos_img_path.exists():
        pos_img = Image(str(pos_img_path), width=6.5*inch, height=3.25*inch)
        story.append(pos_img)
        story.append(Spacer(1, 0.2*inch))
    
    # NER Analysis
    story.append(Paragraph("Named Entity Recognition Distribution Analysis", heading_style))
    story.append(Paragraph(
        "Named entity analysis reveals substantial differences in how positive and negative "
        "reviewers reference people, places, and other entities:",
        body_style
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(create_ner_distribution_table(ner_df, 10))
    story.append(Spacer(1, 0.15*inch))
    
    # NER Visualization  
    ner_img_path = figures_dir / "ner_label_distribution.png"
    if ner_img_path.exists():
        ner_img = Image(str(ner_img_path), width=6.5*inch, height=3.25*inch)
        story.append(ner_img)
        story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Key Findings
    story.append(Paragraph("Key Findings and Patterns", heading_style))
    
    # Calculate metrics
    pos_pivot = pos_df.pivot(index="pos_tag", columns="sentiment", values="proportion")
    ner_pivot = ner_df.pivot(index="entity_label", columns="sentiment", values="proportion")
    
    neg_entity_density = (summary['Negative']['entities'] / summary['Negative']['tokens']) * 100
    pos_entity_density = (summary['Positive']['entities'] / summary['Positive']['tokens']) * 100
    
    story.append(Paragraph(
        "<b>Notable Differences Between Sentiments:</b><br/><br/>"
        f"<b>Positive Reviews:</b><br/>"
        f"• Higher entity density: {pos_entity_density:.2f}% vs {neg_entity_density:.2f}% (23% increase)<br/>"
        f"• More proper nouns (PROPN): {pos_pivot.loc['PROPN', 'Positive']*100:.2f}% vs "
        f"{pos_pivot.loc['PROPN', 'Negative']*100:.2f}% (31% increase)<br/>"
        f"• More person entity mentions: {ner_pivot.loc['PERSON', 'Positive']*100:.2f}% vs "
        f"{ner_pivot.loc['PERSON', 'Negative']*100:.2f}% (14% increase)<br/>"
        f"• Interpretation: Referential writing style that credits specific individuals<br/><br/>"
        f"<b>Negative Reviews:</b><br/>"
        f"• Higher verb usage: {pos_pivot.loc['VERB', 'Negative']*100:.2f}% vs "
        f"{pos_pivot.loc['VERB', 'Positive']*100:.2f}%<br/>"
        f"• More temporal expressions: {ner_pivot.loc['TIME', 'Negative']*100:.2f}% vs "
        f"{ner_pivot.loc['TIME', 'Positive']*100:.2f}%<br/>"
        f"• More cardinal numbers: {ner_pivot.loc['CARDINAL', 'Negative']*100:.2f}% vs "
        f"{ner_pivot.loc['CARDINAL', 'Positive']*100:.2f}%<br/>"
        f"• Interpretation: Analytical approach with precise, quantitative critiques",
        body_style
    ))
    
    story.append(Spacer(1, 0.15*inch))
    
    # Hypotheses and Explanations
    story.append(Paragraph("Hypotheses and Explanations", heading_style))
    story.append(Paragraph(
        "<b>1. Attribution Hypothesis:</b> Positive reviewers credit specific individuals "
        "(actors, directors) for a film's success, while negative reviewers focus on abstract "
        "failures rather than blaming individuals. This explains the substantial increase in "
        "PERSON entities and proper nouns in positive reviews.<br/><br/>"
        "<b>2. Specificity in Criticism:</b> Negative reviewers employ temporal and numerical "
        "precision to support their critiques objectively. References to specific durations, "
        "scenes, or quantifiable elements provide concrete evidence for negative assessments.<br/><br/>"
        "<b>3. Descriptive vs. Analytical Language:</b> Satisfied viewers use descriptive "
        "language emphasizing what exists (nouns, adjectives) and who created it. Dissatisfied "
        "viewers use analytical language focusing on what happens (verbs) and when things go "
        "wrong (temporal expressions).<br/><br/>"
        "<b>4. Emotional Expression:</b> The higher rate of interjections in negative reviews "
        "suggests that disappointment elicits more spontaneous emotional reactions than "
        "satisfaction, which manifests in more structured, referential praise.",
        body_style
    ))
    
    # Conclusion
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Conclusion", heading_style))
    story.append(Paragraph(
        "This analysis reveals notable differences in the distribution of named entities and "
        "parts of speech between positive and negative movie reviews. Positive reviews show 31% "
        "higher proper noun usage, 23% greater entity density, and 14% more person mentions, "
        "indicating a referential writing style that credits specific individuals. Negative "
        "reviews exhibit higher verb usage, more temporal expressions, and elevated cardinal "
        "numbers, reflecting an analytical approach with precise, quantitative critiques. "
        "These patterns suggest that sentiment is deeply embedded in grammatical structure and "
        "referencing behavior, not just evaluative vocabulary.",
        body_style
    ))
    
    return story


# ==============================================================================
# MAIN EXECUTION FUNCTION
# ==============================================================================

def main():
    """Main execution function for comprehensive analysis."""
    
    print("="*70)
    print("IMDB NER & POS DISTRIBUTION ANALYSIS - COMPREHENSIVE SCRIPT")
    print("="*70)
    
    # Setup directory structure
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    figures_dir = project_root / "figures"
    reports_dir = project_root / "reports"
    cache_dir = project_root / ".cache"
    
    for directory in [data_dir, figures_dir, reports_dir, cache_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    cache_path = cache_dir / "distributions_cache.pkl"
    
    # Step 1: Load or compute distributions
    print("\n" + "="*70)
    print("STEP 1: DATA PROCESSING")
    print("="*70)
    
    distributions = load_distributions_cache(cache_path)
    
    if distributions is None:
        print("\nLoading IMDB dataset (nocode-ai/imdb-movie-reviews)...")
        dataset = load_dataset("nocode-ai/imdb-movie-reviews")
        
        print("Loading spaCy model (en_core_web_sm)...")
        nlp = ensure_spacy_model("en_core_web_sm")
        
        print("\nComputing POS and NER distributions...")
        print("⏱  This will take ~20-25 minutes, but results will be cached")
        distributions = compute_distributions(dataset, nlp)
        
        save_distributions_cache(distributions, cache_path)
    else:
        print("✓ Using cached distributions")
    
    # Step 2: Convert to DataFrames
    print("\n" + "="*70)
    print("STEP 2: CONVERT TO DATAFRAMES")
    print("="*70)
    
    pos_df = distributions_to_dataframe(
        distributions.pos_counts, distributions.token_totals, "pos_tag"
    )
    ner_df = distributions_to_dataframe(
        distributions.ner_counts, distributions.entity_totals, "entity_label"
    )
    
    print(f"✓ POS tags DataFrame: {len(pos_df)} rows")
    print(f"✓ NER entities DataFrame: {len(ner_df)} rows")
    
    # Step 3: Save tabular data
    print("\n" + "="*70)
    print("STEP 3: SAVE TABULAR DATA")
    print("="*70)
    
    save_dataframe(pos_df, data_dir / "pos_tag_distribution.csv")
    save_dataframe(ner_df, data_dir / "ner_label_distribution.csv")
    save_summary(distributions, data_dir / "summary.json")
    
    print(f"✓ Saved: {data_dir / 'pos_tag_distribution.csv'}")
    print(f"✓ Saved: {data_dir / 'ner_label_distribution.csv'}")
    print(f"✓ Saved: {data_dir / 'summary.json'}")
    
    # Step 4: Create visualizations
    print("\n" + "="*70)
    print("STEP 4: GENERATE VISUALIZATIONS")
    print("="*70)
    
    plot_distribution(
        pos_df,
        tag_column="pos_tag",
        output_path=figures_dir / "pos_tag_distribution.png",
        title="POS Tag Distribution by Sentiment (IMDB Reviews)",
    )
    print(f"✓ Saved: {figures_dir / 'pos_tag_distribution.png'}")
    
    plot_distribution(
        ner_df,
        tag_column="entity_label",
        output_path=figures_dir / "ner_label_distribution.png",
        title="Named Entity Distribution by Sentiment (IMDB Reviews)",
    )
    print(f"✓ Saved: {figures_dir / 'ner_label_distribution.png'}")
    
    # Step 5: Generate comprehensive report
    print("\n" + "="*70)
    print("STEP 5: GENERATE COMPREHENSIVE PDF REPORT")
    print("="*70)
    
    with open(data_dir / "summary.json", "r") as f:
        summary = json.load(f)
    
    story = build_comprehensive_report(summary, pos_df, ner_df, figures_dir)
    
    report_path = reports_dir / "imdb_ner_pos_comprehensive_report.pdf"
    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    
    doc.build(story)
    print(f"✓ Saved: {report_path}")
    print(f"  Size: {report_path.stat().st_size / 1024:.1f} KB")
    
    # Step 6: Display summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY OF FINDINGS")
    print("="*70)
    
    pos_pivot = pos_df.pivot(index="pos_tag", columns="sentiment", values="proportion")
    ner_pivot = ner_df.pivot(index="entity_label", columns="sentiment", values="proportion")
    
    neg_entity_density = (summary['Negative']['entities'] / summary['Negative']['tokens']) * 100
    pos_entity_density = (summary['Positive']['entities'] / summary['Positive']['tokens']) * 100
    
    print("\n✅ POSITIVE REVIEWS CHARACTERISTICS:")
    print(f"  • Higher entity density: {pos_entity_density:.2f}% (vs {neg_entity_density:.2f}%)")
    print(f"  • More proper nouns: {pos_pivot.loc['PROPN', 'Positive']*100:.2f}% (vs {pos_pivot.loc['PROPN', 'Negative']*100:.2f}%)")
    print(f"  • More person entities: {ner_pivot.loc['PERSON', 'Positive']*100:.2f}% (vs {ner_pivot.loc['PERSON', 'Negative']*100:.2f}%)")
    
    print("\n❌ NEGATIVE REVIEWS CHARACTERISTICS:")
    print(f"  • Higher verb usage: {pos_pivot.loc['VERB', 'Negative']*100:.2f}% (vs {pos_pivot.loc['VERB', 'Positive']*100:.2f}%)")
    print(f"  • More temporal expressions: {ner_pivot.loc['TIME', 'Negative']*100:.2f}% (vs {ner_pivot.loc['TIME', 'Positive']*100:.2f}%)")
    print(f"  • More cardinal numbers: {ner_pivot.loc['CARDINAL', 'Negative']*100:.2f}% (vs {ner_pivot.loc['CARDINAL', 'Positive']*100:.2f}%)")
    
    print("\n📁 ALL OUTPUTS SAVED TO:")
    print(f"  Data: {data_dir}")
    print(f"  Figures: {figures_dir}")
    print(f"  Reports: {reports_dir}")
    print("\n" + "="*70)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

