"""
Quick Test Script - Verify Ollama Connection & Model
Run this first to make sure everything is working!
"""

import requests
import json

print("="*60)
print("🧪 Ollama Connection Test")
print("="*60)

# Test 1: Check if Ollama is running
print("\n1️⃣ Checking Ollama server...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    if response.status_code == 200:
        models = response.json().get("models", [])
        print("   ✅ Ollama is running!")
        print(f"   📦 Available models ({len(models)}):")
        for m in models:
            size_gb = m.get('size', 0) / (1024**3)
            print(f"      - {m['name']} ({size_gb:.1f} GB)")
    else:
        print("   ❌ Ollama responded but with error")
        exit(1)
except requests.exceptions.ConnectionError:
    print("   ❌ Cannot connect to Ollama!")
    print("   💡 Run this command first: ollama serve")
    exit(1)

# Test 2: Try paraphrasing with each model
print("\n2️⃣ Testing paraphrase generation...")

test_sentence = "The quick brown fox jumps over the lazy dog."

for model_name in ["gpt-oss:20b", "gemma-3-pro-review"]:
    print(f"\n   Testing model: {model_name}")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": f"Paraphrase this sentence. Output ONLY the paraphrased sentence:\n\n{test_sentence}",
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 100}
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            print(f"   ✅ Success!")
            print(f"   📝 Original:    {test_sentence}")
            print(f"   📝 Paraphrased: {result[:80]}...")
        else:
            print(f"   ❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Failed: {e}")

print("\n" + "="*60)
print("✅ Test complete! If models work, run generate_extended_iterations.py")
print("="*60)
