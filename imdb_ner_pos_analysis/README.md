# IMDB NER & POS Distribution Analysis

## Project Overview

This project implements entity recognition (NER) and part-of-speech (POS) tagging on the IMDB movie reviews dataset, analyzing the distribution of linguistic features across positive and negative sentiment reviews.

## Objective

The objective of this assignment is to:
1. Perform NER and POS tagging on IMDB movie reviews
2. Analyze the distribution of NER and POS tags in positive vs. negative reviews
3. Compare linguistic patterns across sentiments
4. Generate visualizations and a comprehensive report

## Dataset

**Source**: `nocode-ai/imdb-movie-reviews` from Hugging Face Datasets
- **Total Reviews**: 50,000 (25,000 positive + 25,000 negative)
- **Content**: Movie reviews with sentiment labels

## Project Structure

```
imdb_ner_pos_analysis/
├── data/                           # Generated data files
│   ├── pos_tag_distribution.csv    # POS tag frequency by sentiment
│   ├── ner_label_distribution.csv  # NER entity frequency by sentiment
│   └── summary.json                # Overall statistics
├── figures/                        # Visualization outputs (4-panel dashboards)
│   ├── pos_tag_distribution.png    # POS distribution dashboard (4 panels)
│   └── ner_label_distribution.png  # NER distribution dashboard (4 panels)
├── reports/                        # Generated reports
│   └── imdb_ner_pos_analysis_report.pdf  # Comprehensive analysis report
├── scripts/                        # Python implementation
│   ├── process_imdb.py             # Main NER/POS processing pipeline
│   ├── generate_report.py          # PDF report generation with tables
│   ├── generate_visualizations.py  # Enhanced 4-panel dashboard generator
│   └── verify_outputs.py           # Output validation script
├── requirements.txt                # Project dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Running the Full Analysis Pipeline

1. **Process the IMDB dataset** (performs NER & POS tagging):
```bash
cd imdb_ner_pos_analysis
python scripts/process_imdb.py
```

This script will:
- Download the IMDB dataset from Hugging Face
- Load the spaCy English model
- Process all 50,000 reviews with NER and POS tagging
- Generate CSV files with tag distributions
- Create visualization charts

**Expected Runtime**: ~20-25 minutes (depending on system)

2. **Generate visualizations** (optional - already created by step 1):
```bash
python scripts/generate_visualizations.py
```

3. **Generate the PDF report** (optional - for updates):
```bash
python scripts/generate_report.py
```

This creates a comprehensive analysis report with tables and statistical insights in PDF format.

## Output Files

### Data Files

- **`pos_tag_distribution.csv`**: Contains POS tag frequencies and proportions for positive and negative reviews
- **`ner_label_distribution.csv`**: Contains named entity frequencies and proportions for positive and negative reviews
- **`summary.json`**: Overall statistics including document counts, token counts, and unique tag counts

### Visualizations

- **`pos_tag_distribution.png`**: Professional 4-panel dashboard showing:
  - Side-by-side POS tag comparison
  - Top 10 tags in positive reviews
  - Top 10 tags in negative reviews
  - Normalized percentage distribution
- **`ner_label_distribution.png`**: Professional 4-panel dashboard showing:
  - Side-by-side entity comparison
  - Top 10 entities in positive reviews
  - Top 10 entities in negative reviews
  - Sentiment preference analysis

### Report

- **`imdb_ner_pos_analysis_report.pdf`**: Comprehensive analysis report (~1,500 words) with 4 data tables summarizing:
  - Methodology and data preparation
  - Distribution analysis findings
  - Key insights and patterns
  - Sentiment-based linguistic differences

## Key Findings (Summary)

### POS Distribution Insights
- **Proper Nouns (PROPN)**: 31% higher in positive reviews, indicating more references to actors, directors, and film titles
- **Interjections (INTJ)**: 46% lower in positive reviews, suggesting more emotional expressiveness in negative reviews
- Overall similar distributions with nouns (~19%) and verbs (~12%) being most common

### NER Distribution Insights
- **PERSON entities**: 14% higher in positive reviews, aligning with more actor/director mentions
- **DATE & TIME entities**: More frequent in negative reviews, possibly referencing specific problematic scenes
- Both sentiments show PERSON entities as the most common (~36-41% of all entities)

## Implementation Details

### NER & POS Tagging
- **Library**: spaCy v3.8+ with `en_core_web_sm` model
- **Processing**: Batch processing with pipeline optimization
- **Features Extracted**:
  - POS tags: 17 universal POS categories
  - NER labels: 18 entity types (PERSON, ORG, GPE, DATE, etc.)

### Data Preprocessing
- HTML tag removal (e.g., `<br/>` tags)
- Whitespace normalization
- Tokenization via spaCy
- Alpha-only filtering for POS counts

## Dependencies

Core libraries (see `requirements.txt` for complete list):
- `datasets`: IMDB dataset loading
- `spacy`: NER and POS tagging
- `pandas`: Data manipulation
- `matplotlib` & `seaborn`: Visualization
- `reportlab`: PDF report generation
- `tqdm`: Progress tracking

## Ethical Considerations

- **Data Privacy**: The IMDB dataset is publicly available and does not contain personal information
- **Usage**: This analysis is for educational purposes only
- **Bias**: The dataset reflects inherent biases in movie review writing and should be interpreted accordingly

## Author

This project was completed as part of an Entity Recognition and POS Tagging assignment using the IMDB movie reviews dataset.

## License

This project uses publicly available datasets and open-source libraries. Please refer to individual library licenses for usage terms.