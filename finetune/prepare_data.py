import os
import pandas as pd
import json

# === Paths ===
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../datasets_raw"))
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset.jsonl")

# === Define sources ===
csv_sources = [
    {
        "name": "TripAdvisor Hotel Reviews",
        "file": "hotel.reviews.csv",
        "column": "text",
        "prompt": "Summarize this hotel review:"
    },
    {
        "name": "Tripadvisor Reviews 2023",
        "file": "tripadvisor_reviews_2023.csv",
        "column": "Review",
        "prompt": "Summarize this travel review:"
    },
    {
        "name": "European Tour Destinations Dataset",
        "file": "tourist.destinations.csv",
        "column": "Description",
        "prompt": "Based on this description, suggest a travel itinerary:"
    }
]

examples = []

# === Process each source ===
for src in csv_sources:
    path = os.path.join(DATA_DIR, src["name"], src["file"])
    print(f"📂 Reading {path}...")
    try:
        df = pd.read_csv(path, encoding_errors="ignore")
        for _, row in df.iterrows():
            content = row.get(src["column"], "")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if len(content) < 30 or len(content) > 1000:
                continue

            # Define input and output explicitly
            instruction = src["prompt"]
            input_text = content[:300]  # optional truncation
            output_text = content       # use full response as label for now

            examples.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text
            })
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")

# === Write to JSONL ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for ex in examples:
        json.dump(ex, f)
        f.write("\n")

print(f"✅ Saved {len(examples)} examples to {OUTPUT_PATH}")
