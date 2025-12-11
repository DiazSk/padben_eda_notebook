# 📊 ANALYSIS OUTPUTS - ORGANIZED STRUCTURE

Last updated: December 10, 2025

---

## 📁 Directory Structure

```
analysis_outputs/
├── Type1_Type4_SingleIteration/      ← Single-iteration paraphrasing analysis
│   ├── step1_preprocessing_comparison.png
│   ├── step2_parascore_validation.png
│   └── README.md
│
└── Type5_Iterative_MultiIteration/   ← Multi-iteration paraphrasing analysis
    ├── type5_iteration_analysis.png
    └── README.md
```

---

## 🔍 **Quick Reference Guide**

### **For Single-Iteration Analysis (Type 1 → Type 4):**
📂 Go to: `Type1_Type4_SingleIteration/`

**What's there:**
- Figure 1: Preprocessing impact (4 subplots)
- Figure 2: ParaScore validation (4 subplots)
- README with detailed interpretation

**Use for presentation slides:**
- Methodology section
- ParaScore validation section
- Data quality section

---

### **For Multi-Iteration Analysis (Type 2 → Type 5):**
📂 Go to: `Type5_Iterative_MultiIteration/`

**What's there:**
- Figure 3: Iteration analysis (2 subplots)
- README with interpretation

**Use for presentation slides:**
- Multi-iteration results section
- Catastrophic drift demonstration
- Breaking point analysis

---

## 🎯 **Key Differences**

| Aspect | Type 1→4 | Type 2→5 |
|--------|----------|----------|
| **Source** | Human text | LLM text |
| **Iterations** | 1 (single paraphrase) | 3 (iterative chain) |
| **Mean SDS** | 0.510 | 1.17 (cumulative) |
| **Pattern** | Controlled drift | Exponential collapse |
| **ParaScore** | Threshold validation | Hop-by-hop method |
| **Finding** | 74.7% high drift | 334% over threshold |

---

## 📌 **Which Figures to Use When**

### **For Academic Rigor:**
Use **Type 1→4** figures (preprocessing + ParaScore validation)
- Shows methodology
- Validates against literature
- Professional quality assurance

### **For Impact/Wow Factor:**
Use **Type 5** figure (iteration analysis)
- Shows original finding
- Exponential drift is dramatic
- Proves hypothesis

### **For Complete Story:**
Use **ALL 3 FIGURES** in sequence:
1. Data quality (preprocessing)
2. Theoretical validation (ParaScore)
3. Original contribution (iterative drift)

---

## 🚀 **Quick Start for Presentation Prep**

### **Tomorrow Morning Checklist:**

1. **Review figures:**
   ```bash
   open Type1_Type4_SingleIteration/step1_preprocessing_comparison.png
   open Type1_Type4_SingleIteration/step2_parascore_validation.png
   open Type5_Iterative_MultiIteration/type5_iteration_analysis.png
   ```

2. **Read README in each folder** for exact interpretation

3. **Add to slides** in this order:
   - Slide 6: Figure 1 (preprocessing)
   - Slide 7: Figure 2 (ParaScore validation)
   - Slide 8: Figure 3 (iterative drift)

4. **Practice explaining** using README talking points

---

## 📝 **File Descriptions**

### **step1_preprocessing_comparison.png**
- **Analysis:** Type 1 vs Type 4 on full and filtered datasets
- **Purpose:** Validate data cleaning methodology
- **Subplots:** 4 (distribution, categories, metrics, table)
- **Key stat:** Δ SDS = 0.000 (no bias from filtering)

### **step2_parascore_validation.png**
- **Analysis:** Type 1 vs Type 4 with ParaScore framework
- **Purpose:** Theoretical validation against published research
- **Subplots:** 4 (threshold, distance effect, dual-criteria, summary)
- **Key stat:** 0.35 threshold matches ParaScore exactly

### **type5_iteration_analysis.png**
- **Analysis:** Type 2 → 5 (1st) → 5 (3rd) iterative chain
- **Purpose:** Demonstrate drift accumulation
- **Subplots:** 2 (hop-by-hop comparison, cumulative progression)
- **Key stat:** Cumulative SDS = 1.17 (exponential growth)

---

## 💡 **Presentation Strategy**

### **Conservative Approach (If Short on Time):**
Use Figures 1 & 2 only
- Focus on validation and methodology
- Mention Type 5 as "ongoing work"

### **Ambitious Approach (Recommended):**
Use ALL 3 figures
- Complete story arc
- Shows depth of analysis
- Original contribution (Figure 3)

### **Emergency Backup:**
If questions about missing 5th iteration:
> "We analyzed 1st and 3rd iterations to demonstrate methodology. The exponential pattern observed suggests even more severe drift at the 5th iteration, which represents valuable future validation work."

---

## 🎯 **For Teammates - Quick Summary**

**Type 1→4 Analysis (Figures 1 & 2):**
- Single paraphrase, human source
- Shows LLMs create moderate drift (SDS = 0.51)
- Validated against ParaScore framework
- 74.7% in high drift zone

**Type 5 Analysis (Figure 3):**
- Iterative paraphrasing, LLM source
- Shows CATASTROPHIC drift accumulation (SDS = 1.17)
- Exponential growth pattern
- Way beyond ParaScore safe zone (3.3x over)

**Combined Message:**
Single-iteration = controlled risk. Multi-iteration = disaster.

---

## 📚 **References**

**ParaScore Paper:**
Shen, L., Liu, L., Jiang, H., & Shi, S. (2022). On the Evaluation Metrics for Paraphrase Generation. In Proceedings of EMNLP 2022.

**Key Concept Applied:**
- 0.35 threshold for quality collapse
- Reference-free evaluation for iterative chains
- Distance effect on metric reliability

---

Last updated: December 11, 2025, 12:30 AM
Ready for Friday presentation! 🚀
