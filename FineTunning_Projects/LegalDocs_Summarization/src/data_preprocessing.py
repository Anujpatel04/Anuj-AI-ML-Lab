import os
import json
from tqdm import tqdm

input_dataset_dir = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/LegalDocs/dataset"
output_dataset_dir = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/FineTunning_Projects/dataset"

judgement_dir = os.path.join(input_dataset_dir, "IN-Ext/judgement/")
full_summary_dir = os.path.join(input_dataset_dir, "IN-Ext/summary/full/")
segment_summary_dir = os.path.join(input_dataset_dir, "IN-Ext/summary/segment-wise/")
output_dir = os.path.join(output_dataset_dir, "processed-IN-Ext/")
os.makedirs(output_dir, exist_ok=True)  

def load_full_summaries(judgement_dir, full_summary_dir, author):
    """
    Load full summaries written by a specific author.
    """
    data = []
    for filename in tqdm(os.listdir(judgement_dir)):
        if filename.endswith(".txt"):
            judgement_path = os.path.join(judgement_dir, filename)
            summary_path = os.path.join(full_summary_dir, author, filename)

            if os.path.exists(summary_path):
                with open(judgement_path, "r", encoding="utf-8") as f:
                    judgement = f.read()
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = f.read()
                data.append({"filename": filename, "judgement": judgement, "summary": summary, "author": author})
    return data

def load_segment_summaries(segment_summary_dir, author):
    """
    Load segment-wise summaries written by a specific author, handling potential encoding issues.
    """
    data = []
    segments = ["analysis", "argument", "facts", "judgement", "statute"]
    for filename in tqdm(os.listdir(os.path.join(segment_summary_dir, author, "analysis"))):
        if filename.endswith(".txt"):
            segment_text = {}
            for segment in segments:
                segment_path = os.path.join(segment_summary_dir, author, segment, filename)
                if os.path.exists(segment_path):
                    
                    try:
                        with open(segment_path, "r", encoding="utf-8") as f:
                            segment_text[segment] = f.read()
                    except UnicodeDecodeError:
                        with open(segment_path, "r", encoding="latin-1") as f:
                            segment_text[segment] = f.read()
            data.append({"filename": filename, "segments": segment_text, "author": author})
    return data

print("Loading full summaries...")
full_summaries_A1 = load_full_summaries(judgement_dir, full_summary_dir, "A1")
full_summaries_A2 = load_full_summaries(judgement_dir, full_summary_dir, "A2")

print("Loading segment-wise summaries...")
segment_summaries_A1 = load_segment_summaries(segment_summary_dir, "A1")
segment_summaries_A2 = load_segment_summaries(segment_summary_dir, "A2")

print("Saving datasets...")

with open(os.path.join(output_dir, "full_summaries_A1.jsonl"), "w", encoding="utf-8") as f:
    for item in full_summaries_A1:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "full_summaries_A2.jsonl"), "w", encoding="utf-8") as f:
    for item in full_summaries_A2:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "segment_summaries_A1.jsonl"), "w", encoding="utf-8") as f:
    for item in segment_summaries_A1:
        f.write(json.dumps(item) + "\n")

with open(os.path.join(output_dir, "segment_summaries_A2.jsonl"), "w", encoding="utf-8") as f:
    for item in segment_summaries_A2:
        f.write(json.dumps(item) + "\n")

print("Datasets saved in:", output_dir)
