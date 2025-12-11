import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import time

# --- CONFIGURATION ---
API_KEY = "Your_API_Key_Here"  # Replace with your Gemini API key (starts with AIza...)
INPUT_FILE = "data.json"
OUTPUT_FILE = "cumulative_dataset.json"
BATCH_SIZE = 10  # Reduced batch size to stay within rate limits
NUM_ITERATIONS = 4  # Generate type4-1, type4-2, type4-3, type4-4
DELAY_BETWEEN_BATCHES = 4  # Seconds to wait between batches (free tier: 15 RPM)

# 1. SETUP
genai.configure(api_key=API_KEY)

# Disable safety filters for research data integrity
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",  # Using Flash for higher quota limits
    generation_config={"response_mime_type": "application/json"},
    safety_settings=safety_settings,
)

# 2. SYSTEM INSTRUCTION
# We use a generic instruction because the specific prompt is applied in the loop
system_instruction = """
You are a precise paraphrasing engine. You will receive a JSON list of texts.
For EACH text, apply the following instruction exactly:
"Please paraphrase the following text while maintaining its original meaning"

Output a JSON list of objects with this schema:
[
  {"original_text": "string", "paraphrased_text": "string"}
]
"""


# 3. BATCH PROCESSOR
def process_batch(texts: list[str]) -> list[dict[str, str]] | None:
    """Process a batch of texts. Returns list of results or None on error."""
    # texts is a list of strings
    prompt = f"{system_instruction}\n\nINPUT LIST:\n{json.dumps(texts)}"
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"    ⚠️ Batch failed: {e}")
        return None


def process_single(text: str) -> str | None:
    """Process a single text (fallback for batch mismatches)."""
    prompt = f"{system_instruction}\n\nINPUT LIST:\n{json.dumps([text])}"
    try:
        response = model.generate_content(prompt)
        result = json.loads(response.text)
        if result and len(result) > 0:
            return result[0].get("paraphrased_text")
        return None
    except Exception as e:
        print(f"    ⚠️ Single-item failed: {e}")
        return None


# 4. MAIN EXECUTION
def main():
    print(f"📂 Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    total_records = len(data)
    print(f"🚀 Loaded {total_records} records. Starting cumulative generation...")

    # We will run 4 distinct passes over the entire dataset
    for iteration in range(1, NUM_ITERATIONS + 1):
        # Define source and target keys
        # Iteration 1: Source = 'llm_paraphrased_original_text(type4)-prompt-based', Target = 'type4-1'
        # Iteration 2: Source = 'type4-1', Target = 'type4-2'
        if iteration == 1:
            source_key = "llm_paraphrased_original_text(type4)-prompt-based"
        else:
            source_key = f"type4-{iteration - 1}"

        target_key = f"type4-{iteration}"

        print(f"\n🔄 --- ITERATION {iteration}/{NUM_ITERATIONS} ---")
        print(f"   Taking input from: '{source_key}'")
        print(f"   Generating: '{target_key}'")

        # Collect inputs for this pass
        # We keep track of indices to map results back to the correct record
        inputs_to_process: list[str] = []
        indices_map: list[int] = []

        for idx, record in enumerate(data):
            text = record.get(source_key)
            if text and isinstance(text, str):
                inputs_to_process.append(text)
                indices_map.append(idx)
            else:
                # If source is missing, we can't generate the next step
                record[target_key] = None

        # Process in batches with mismatch handling and fallback to single-item processing
        total_inputs = len(inputs_to_process)

        for i in range(0, total_inputs, BATCH_SIZE):
            batch_texts = inputs_to_process[i : i + BATCH_SIZE]
            batch_indices = indices_map[i : i + BATCH_SIZE]

            # Rate limiting: wait between batches to avoid quota errors
            if i > 0:
                time.sleep(DELAY_BETWEEN_BATCHES)

            # Retry logic for batch
            success = False
            for attempt in range(3):
                output = process_batch(batch_texts)

                if output and len(output) == len(batch_texts):
                    # Success: Map results back to the main data list
                    for k, result in enumerate(output):
                        record_idx = batch_indices[k]
                        data[record_idx][target_key] = result.get("paraphrased_text")

                    print(
                        f"   ✅ Processed {min(i + BATCH_SIZE, total_inputs)}/{total_inputs} (batch success)"
                    )
                    success = True
                    break
                else:
                    # Output length mismatch or rate limit: wait and retry
                    expected = len(batch_texts)
                    actual = len(output) if output else 0
                    print(
                        f"   ⚠️ Output mismatch: expected {expected}, got {actual}. Retry {attempt + 1}/3..."
                    )
                    # Exponential backoff for rate limit errors
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)

            if not success:
                # Fallback: process remaining items one-by-one to preserve order and identify missing results
                print("   📍 Batch failed; falling back to single-item processing...")
                for k, text in enumerate(batch_texts):
                    record_idx = batch_indices[k]
                    paraphrased = process_single(text)
                    data[record_idx][target_key] = paraphrased
                    if paraphrased:
                        print(
                            f"      ✅ Single-item {k + 1}/{len(batch_texts)}: Success"
                        )
                    else:
                        print(
                            f"      ❌ Single-item {k + 1}/{len(batch_texts)}: Failed (result set to None)"
                        )

        # Save intermediate progress (so you don't lose everything if it crashes on step 4)
        print(f"💾 Saving progress after Iteration {iteration}...")
        with open(OUTPUT_FILE, "w") as f:
            json.dump(data, f, indent=2)

    print(f"\n🎉 Done! Final dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
