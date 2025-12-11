"""
Extended Iteration Generator using Local Ollama
Generates 5th, 7th, 9th iterations for Type 5 paraphrase chain
Author: Zaid Shaikh - PADBen Project
"""

import json
import requests
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION - Change these as needed
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"  # or "gemma-3-pro-review"
SAMPLE_SIZE = 100  # Number of samples to process (increase for full run)
OUTPUT_DIR = Path("Type5_Extended_Iterations")

# ============================================================
# Ollama Paraphrase Function
# ============================================================
def paraphrase_with_ollama(text, model=MODEL_NAME, max_retries=3):
    """Generate paraphrase using local Ollama model"""
    
    prompt = f"""Paraphrase the following sentence completely. 
Change the words and sentence structure while preserving the original meaning.
Output ONLY the paraphrased sentence, nothing else. No explanations, no quotes.

Sentence: {text}

Paraphrased:"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": 256
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                paraphrased = result.get("response", "").strip()
                
                # Clean up the response
                paraphrased = paraphrased.replace('"', '').replace("'", "")
                if paraphrased.startswith("Paraphrased:"):
                    paraphrased = paraphrased[12:].strip()
                
                # Basic validation
                if len(paraphrased) > 10 and len(paraphrased) < len(text) * 3:
                    return paraphrased
                    
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    
    return None

# ============================================================
# SDS Calculation Functions
# ============================================================
def calculate_jaccard(text1, text2):
    """Calculate Jaccard similarity"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def calculate_ngram_overlap(text1, text2, n=2):
    """Calculate n-gram overlap (ROUGE-like)"""
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0
    
    intersection = len(ngrams1 & ngrams2)
    return intersection / max(len(ngrams1), len(ngrams2))

def calculate_sds_simple(text1, text2):
    """Calculate simplified SDS (without SBERT - use for quick analysis)"""
    jaccard = calculate_jaccard(text1, text2)
    bigram = calculate_ngram_overlap(text1, text2, 2)
    trigram = calculate_ngram_overlap(text1, text2, 3)
    
    # Combine into divergence score
    jaccard_div = 1 - jaccard
    bigram_div = 1 - bigram
    trigram_div = 1 - trigram
    
    # Weighted composite
    sds = 0.4 * jaccard_div + 0.35 * bigram_div + 0.25 * trigram_div
    return sds

# ============================================================
# Main Processing
# ============================================================
def main():
    print("="*70)
    print("🚀 PADBen Extended Iteration Generator (Ollama Local)")
    print("="*70)
    
    # Check Ollama connection
    print(f"\n📡 Checking Ollama connection...")
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json().get("models", [])
            print(f"✓ Ollama connected! Available models:")
            for m in models:
                print(f"   - {m['name']}")
        else:
            print("❌ Ollama not responding properly")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: `ollama serve`")
        return
    
    # Load data
    print(f"\n📂 Loading data.json...")
    data_path = Path(__file__).parent.parent / "data.json"
    
    if not data_path.exists():
        data_path = Path("C:/EDA/data.json")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} samples")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process samples
    print(f"\n🔄 Generating iterations 5, 7, 9 for {SAMPLE_SIZE} samples...")
    print(f"   Using model: {MODEL_NAME}")
    print("-"*70)
    
    results = []
    
    for i in tqdm(range(min(SAMPLE_SIZE, len(data))), desc="Processing"):
        item = data[i]
        
        # Get existing data
        type2 = item.get('llm_generated_text(type2)', '')
        iter_1 = item.get('llm_paraphrased_generated_text(type5)-1st', '')
        iter_3 = item.get('llm_paraphrased_generated_text(type5)-3rd', '')
        
        if not iter_3:
            continue
        
        # Generate new iterations
        iter_5 = paraphrase_with_ollama(iter_3)
        if not iter_5:
            continue
            
        iter_7 = paraphrase_with_ollama(iter_5)
        if not iter_7:
            continue
            
        iter_9 = paraphrase_with_ollama(iter_7)
        if not iter_9:
            continue
        
        # Calculate SDS for each hop
        sds_1_3 = calculate_sds_simple(iter_1, iter_3)
        sds_3_5 = calculate_sds_simple(iter_3, iter_5)
        sds_5_7 = calculate_sds_simple(iter_5, iter_7)
        sds_7_9 = calculate_sds_simple(iter_7, iter_9)
        
        # Calculate E2E from Type 2
        sds_e2e_3 = calculate_sds_simple(type2, iter_3)
        sds_e2e_5 = calculate_sds_simple(type2, iter_5)
        sds_e2e_7 = calculate_sds_simple(type2, iter_7)
        sds_e2e_9 = calculate_sds_simple(type2, iter_9)
        
        results.append({
            'idx': i,
            'dataset_source': item.get('dataset_source', 'unknown'),
            'type2': type2,
            'iter_1': iter_1,
            'iter_3': iter_3,
            'iter_5': iter_5,
            'iter_7': iter_7,
            'iter_9': iter_9,
            'sds_1_3': sds_1_3,
            'sds_3_5': sds_3_5,
            'sds_5_7': sds_5_7,
            'sds_7_9': sds_7_9,
            'sds_e2e_3': sds_e2e_3,
            'sds_e2e_5': sds_e2e_5,
            'sds_e2e_7': sds_e2e_7,
            'sds_e2e_9': sds_e2e_9
        })
    
    print(f"\n✓ Successfully processed {len(results)} samples")
    
    # Save results
    output_file = OUTPUT_DIR / "extended_iterations_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Results saved to {output_file}")
    
    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "="*70)
    print("📊 EXTENDED ITERATION ANALYSIS")
    print("="*70)
    
    if len(results) > 0:
        # Hop-by-hop statistics
        sds_1_3 = [r['sds_1_3'] for r in results]
        sds_3_5 = [r['sds_3_5'] for r in results]
        sds_5_7 = [r['sds_5_7'] for r in results]
        sds_7_9 = [r['sds_7_9'] for r in results]
        
        print(f"\n📈 HOP-BY-HOP SDS (n={len(results)}):")
        print(f"   Iter 1→3:  {np.mean(sds_1_3):.4f} ± {np.std(sds_1_3):.4f}")
        print(f"   Iter 3→5:  {np.mean(sds_3_5):.4f} ± {np.std(sds_3_5):.4f}")
        print(f"   Iter 5→7:  {np.mean(sds_5_7):.4f} ± {np.std(sds_5_7):.4f}")
        print(f"   Iter 7→9:  {np.mean(sds_7_9):.4f} ± {np.std(sds_7_9):.4f}")
        
        # E2E statistics
        e2e_3 = [r['sds_e2e_3'] for r in results]
        e2e_5 = [r['sds_e2e_5'] for r in results]
        e2e_7 = [r['sds_e2e_7'] for r in results]
        e2e_9 = [r['sds_e2e_9'] for r in results]
        
        print(f"\n📈 CUMULATIVE E2E SDS (vs Type 2):")
        print(f"   After Iter 3: {np.mean(e2e_3):.4f}")
        print(f"   After Iter 5: {np.mean(e2e_5):.4f}")
        print(f"   After Iter 7: {np.mean(e2e_7):.4f}")
        print(f"   After Iter 9: {np.mean(e2e_9):.4f}")
        
        # Pattern analysis
        hop_means = [np.mean(sds_1_3), np.mean(sds_3_5), np.mean(sds_5_7), np.mean(sds_7_9)]
        
        print(f"\n📉 DRIFT PATTERN:")
        for i in range(1, len(hop_means)):
            change = ((hop_means[i] / hop_means[i-1]) - 1) * 100
            print(f"   Step {i} → {i+1}: {change:+.1f}%")
        
        if all(hop_means[i] >= hop_means[i+1] for i in range(len(hop_means)-1)):
            print(f"\n   🔵 PATTERN: ASYMPTOTIC (Decelerating)")
        elif all(hop_means[i] <= hop_means[i+1] for i in range(len(hop_means)-1)):
            print(f"\n   🔴 PATTERN: ACCELERATING (Catastrophic)")
        else:
            print(f"\n   🟡 PATTERN: NON-MONOTONIC")
        
        # Statistical test
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(sds_1_3, sds_7_9)
        cohens_d = (np.mean(sds_1_3) - np.mean(sds_7_9)) / np.sqrt((np.var(sds_1_3) + np.var(sds_7_9))/2)
        
        print(f"\n📊 STATISTICAL SIGNIFICANCE:")
        print(f"   t-statistic: {t_stat:.4f}")
        print(f"   p-value: {p_val:.2e}")
        print(f"   Cohen's d: {cohens_d:.4f}")
        print(f"   Significant: {'✅ YES' if p_val < 0.05 else '❌ NO'}")
        
        # ParaScore comparison
        parascore_threshold = 0.35
        print(f"\n🎯 vs PARASCORE THRESHOLD (γ=0.35):")
        print(f"   After Iter 9: {np.mean(e2e_9)/parascore_threshold:.1f}x threshold")
    
    print("\n" + "="*70)
    print("✅ DONE! Check the output folder for detailed results.")
    print("="*70)

if __name__ == "__main__":
    main()
