# 📊 Type 1 → Type 4 Analysis (Single-Iteration Paraphrasing)

## 🎯 What This Analysis Covers

**Comparison:** Human Original Text (Type 1) vs LLM Paraphrased Text (Type 4)

**Purpose:** Evaluate semantic drift in single-step LLM paraphrasing

**Dataset:** 5000 samples → 4895 after filtering

---

## 📁 Files in This Directory

### **1. step1_preprocessing_comparison.png**
**4 subplots showing:**
- Top-left: SDS distribution (full vs filtered)
- Top-right: Drift category percentages
- Bottom-left: Individual metric comparison
- Bottom-right: Statistical summary table

**Key Finding:** Filtering removed 2.1% samples with zero impact on distribution (Δ = 0.000)

---

### **2. step2_parascore_validation.png**
**4 subplots showing:**
- Top-left: Threshold alignment with ParaScore (0.35)
- Top-right: Distance effect (SDS 0.17 → 0.56)
- Bottom-left: Dual-criteria scatter plot (Jaccard vs Edit Distance)
- Bottom-right: Validation summary text box

**Key Finding:** Our 0.35 threshold independently matches ParaScore's quality collapse boundary

---

## 🔑 **Key Metrics (Type 1 → 4)**

| Metric | Value |
|--------|-------|
| Mean SDS | 0.510 |
| Std Dev | 0.089 |
| Low Drift (<0.35) | 4.1% |
| Medium Drift (0.35-0.45) | 20.6% |
| High Drift (≥0.45) | **74.7%** |

---

## ✅ **ParaScore Integration**

**Validated findings:**
1. ✅ Threshold alignment (0.35 matches published research)
2. ✅ Distance effect confirmed (linear SDS increase with divergence)
3. ✅ Reference-free evaluation (comparing original to paraphrase directly)
4. ✅ Dual-criteria approach (semantic + lexical metrics combined)

---

## 💡 **What This Tells Us**

**Main Conclusion:** LLM paraphrasing creates significant semantic drift in single-step scenarios. 74.7% of samples exceed the medium-drift threshold, indicating that paraphrase attacks cause measurable semantic displacement.

**Validation:** Alignment with ParaScore's published findings (Shen et al., EMNLP 2022) provides theoretical foundation for our 0.35 threshold.

**Implication:** Single-iteration LLM paraphrasing is detectable through drift analysis, but creates sufficient variation to potentially evade simple detectors.

---

## 📌 **For Presentation**

**Slide 1:** Use Figure 1 for "Data Quality & Methodology"
**Slide 2:** Use Figure 2 for "Validation Against ParaScore Framework"

**Key message:** "Our framework validated against established NLP research"
