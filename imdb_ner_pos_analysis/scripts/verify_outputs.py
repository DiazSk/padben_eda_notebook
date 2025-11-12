#!/usr/bin/env python
"""
Verification script to check that all expected outputs exist and are valid.
"""

from pathlib import Path
import pandas as pd
import json


def main():
    project_root = Path(__file__).resolve().parents[1]
    
    print("=" * 60)
    print("IMDB NER/POS Analysis - Output Verification")
    print("=" * 60)
    
    # Check data directory
    data_dir = project_root / "data"
    print("\n[DATA FILES]")
    
    pos_csv = data_dir / "pos_tag_distribution.csv"
    if pos_csv.exists():
        pos_df = pd.read_csv(pos_csv)
        print(f"  [OK] pos_tag_distribution.csv ({len(pos_df)} rows)")
    else:
        print(f"  [MISSING] pos_tag_distribution.csv")
    
    ner_csv = data_dir / "ner_label_distribution.csv"
    if ner_csv.exists():
        ner_df = pd.read_csv(ner_csv)
        print(f"  [OK] ner_label_distribution.csv ({len(ner_df)} rows)")
    else:
        print(f"  [MISSING] ner_label_distribution.csv")
    
    summary_json = data_dir / "summary.json"
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
        print(f"  [OK] summary.json ({len(summary)} sentiments)")
        print(f"    - Negative: {summary.get('Negative', {}).get('documents', 0)} docs")
        print(f"    - Positive: {summary.get('Positive', {}).get('documents', 0)} docs")
    else:
        print(f"  [MISSING] summary.json")
    
    # Check figures directory
    figures_dir = project_root / "figures"
    print("\n[VISUALIZATION FILES]")
    
    pos_fig = figures_dir / "pos_tag_distribution.png"
    if pos_fig.exists():
        size_kb = pos_fig.stat().st_size / 1024
        print(f"  [OK] pos_tag_distribution.png ({size_kb:.1f} KB)")
    else:
        print(f"  [MISSING] pos_tag_distribution.png")
    
    ner_fig = figures_dir / "ner_label_distribution.png"
    if ner_fig.exists():
        size_kb = ner_fig.stat().st_size / 1024
        print(f"  [OK] ner_label_distribution.png ({size_kb:.1f} KB)")
    else:
        print(f"  [MISSING] ner_label_distribution.png")
    
    # Check reports directory
    reports_dir = project_root / "reports"
    print("\n[REPORT FILES]")
    
    report_pdf = reports_dir / "imdb_ner_pos_analysis_report.pdf"
    if report_pdf.exists():
        size_kb = report_pdf.stat().st_size / 1024
        print(f"  [OK] imdb_ner_pos_analysis_report.pdf ({size_kb:.1f} KB)")
    else:
        print(f"  [MISSING] imdb_ner_pos_analysis_report.pdf")
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

