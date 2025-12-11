#!/usr/bin/env python
"""
Process the IMDB movie reviews dataset to perform part-of-speech (POS) tagging
and named entity recognition (NER) using spaCy. The script aggregates the
distribution of POS and NER tags for positive and negative reviews, saves the
results as tabular data, and creates visualizations for further analysis.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Tuple, Optional

import matplotlib

# Use a non-interactive backend to allow rendering without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from datasets import DatasetDict, load_dataset
from spacy.language import Language
from tqdm import tqdm


SENTIMENT_ORDER = ("negative", "positive")
VALID_SENTIMENTS = set(SENTIMENT_ORDER)
SENTIMENT_DISPLAY = {label: label.capitalize() for label in SENTIMENT_ORDER}
CLEAN_HTML_REGEX = re.compile(r"<br\s*/?>", re.IGNORECASE)

# Configure seaborn aesthetics globally.
sns.set_theme(style="whitegrid")


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


def ensure_spacy_model(model_name: str) -> Language:
    """Load a spaCy language model, downloading it if necessary."""
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli import download

        download(model_name)
        return spacy.load(model_name)


def clean_review(text: str) -> str:
    """
    Apply lightweight text cleaning to IMDB reviews.

    - Replaces HTML break tags with spaces.
    - Collapses repeated whitespace.
    """
    text = CLEAN_HTML_REGEX.sub(" ", text)
    # Keep punctuation for POS tagging, but normalise whitespace.
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

        # POS tagging statistics.
        for token in doc:
            if token.is_alpha:
                distributions.pos_counts[sentiment][token.pos_] += 1
                distributions.token_totals[sentiment] += 1

        # NER statistics.
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
            records.append(
                {
                    "sentiment": sentiment,
                    "tag": tag,
                    "count": int(count),
                    "total": int(total),
                    "proportion": proportion,
                }
            )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    df = df.sort_values(by=["sentiment", "count"], ascending=[True, False])
    df = df.rename(columns={"tag": value_name})
    return df


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Persist the dataframe to disk if it is not empty."""
    if df.empty:
        empty_cols = ["sentiment"]
        # When the dataframe is empty we infer the remaining column names from the target path.
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
    plt.title(title)
    plt.xlabel(tag_column.upper())
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_summary(distributions: TagDistributions, output_path: Path) -> None:
    """Persist aggregated summary statistics as JSON."""
    summary = {}
    seen = set()
    ordered_sentiments = list(SENTIMENT_ORDER)
    for sentiment_key in distributions.document_totals.keys():
        if sentiment_key not in seen and sentiment_key not in VALID_SENTIMENTS:
            ordered_sentiments.append(sentiment_key)
    for sentiment_key in ordered_sentiments:
        seen.add(sentiment_key)
        display_name = SENTIMENT_DISPLAY.get(sentiment_key, sentiment_key.capitalize())
        summary[display_name] = {
            "documents": distributions.document_totals.get(sentiment_key, 0),
            "tokens": distributions.token_totals.get(sentiment_key, 0),
            "entities": distributions.entity_totals.get(sentiment_key, 0),
            "unique_pos_tags": len(distributions.pos_counts.get(sentiment_key, [])),
            "unique_entity_labels": len(distributions.ner_counts.get(sentiment_key, [])),
        }
    output_path.write_text(json.dumps(summary, indent=2))


def save_distributions_cache(distributions: TagDistributions, cache_path: Path) -> None:
    """Save the computed distributions to a cache file."""
    print(f"Saving distributions cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(distributions, f)
    print(f"  Cache saved successfully ({cache_path.stat().st_size / (1024*1024):.2f} MB)")


def load_distributions_cache(cache_path: Path) -> Optional[TagDistributions]:
    """Load distributions from cache if available."""
    if not cache_path.exists():
        return None
    
    print(f"Loading distributions from cache: {cache_path}")
    try:
        with open(cache_path, 'rb') as f:
            distributions = pickle.load(f)
        print("  Cache loaded successfully!")
        print(f"    - Negative documents: {distributions.document_totals.get('negative', 0):,}")
        print(f"    - Positive documents: {distributions.document_totals.get('positive', 0):,}")
        return distributions
    except Exception as e:
        print(f"  Warning: Failed to load cache ({e}), will recompute...")
        return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figures_dir = project_root / "figures"
    reports_dir = project_root / "reports"
    cache_dir = project_root / ".cache"

    for directory in (data_dir, figures_dir, reports_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "distributions_cache.pkl"

    # Try to load from cache first
    distributions = load_distributions_cache(cache_path)

    if distributions is None:
        # Cache miss - need to compute from scratch
        print("Loading IMDB dataset (nocode-ai/imdb-movie-reviews)...")
        dataset = load_dataset("nocode-ai/imdb-movie-reviews")

        print("Loading spaCy model (en_core_web_sm)...")
        nlp = ensure_spacy_model("en_core_web_sm")

        print("Computing POS and NER distributions...")
        print("  (This will take ~20-25 minutes, but results will be cached)")
        distributions = compute_distributions(dataset, nlp)

        # Save to cache for next time
        save_distributions_cache(distributions, cache_path)
    else:
        print("Using cached distributions (to recompute, delete .cache/distributions_cache.pkl)")

    print("\nConverting distributions to dataframes...")
    pos_df = distributions_to_dataframe(distributions.pos_counts, distributions.token_totals, "pos_tag")
    ner_df = distributions_to_dataframe(distributions.ner_counts, distributions.entity_totals, "entity_label")

    print("Saving tabular outputs...")
    save_dataframe(pos_df, data_dir / "pos_tag_distribution.csv")
    save_dataframe(ner_df, data_dir / "ner_label_distribution.csv")
    save_summary(distributions, data_dir / "summary.json")

    print("Creating visualisations...")
    plot_distribution(
        pos_df,
        tag_column="pos_tag",
        output_path=figures_dir / "pos_tag_distribution.png",
        title="POS Tag Distribution by Sentiment (IMDB Reviews)",
    )
    plot_distribution(
        ner_df,
        tag_column="entity_label",
        output_path=figures_dir / "ner_label_distribution.png",
        title="Named Entity Distribution by Sentiment (IMDB Reviews)",
    )

    print("\n" + "="*60)
    print("Analysis complete. Outputs saved to:")
    print("="*60)
    print(f"  POS distribution: {data_dir / 'pos_tag_distribution.csv'}")
    print(f"  NER distribution: {data_dir / 'ner_label_distribution.csv'}")
    print(f"  Summary:          {data_dir / 'summary.json'}")
    print(f"  POS figure:       {figures_dir / 'pos_tag_distribution.png'}")
    print(f"  NER figure:       {figures_dir / 'ner_label_distribution.png'}")
    print(f"  Cache:            {cache_path}")
    print("="*60)


if __name__ == "__main__":
    main()

