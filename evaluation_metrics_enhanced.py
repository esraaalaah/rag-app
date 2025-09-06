"""
Multi-Model QA Evaluation on SciQ Dataset
-----------------------------------------

This script compares multiple models on the SciQ dataset.
For each model:
    - Generates answers given a question
    - Evaluates predictions using ROUGE, BLEU, and BERTScore
    - Saves results and predictions
    - Produces a visualization comparing models

Metrics:
    - ROUGE-1: unigram recall overlap
    - ROUGE-2: bigram recall overlap
    - ROUGE-L: longest common subsequence
    - BLEU: n-gram precision (common in MT)
    - BERTScore-F1: semantic similarity (cosine sim in embedding space)

Author: You 😊
"""

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import torch

# ============================
# 1. Setup
# ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# Models to compare (you can add/remove)
model_names = {
    "BART-base": "facebook/bart-base",
    "T5-small": "t5-small",
    "FLAN-T5-base": "google/flan-t5-base"
}

# ============================
# 2. Load Dataset
# ============================
dataset = load_dataset("bigbio/sciq", "sciq_source")["test"]

# Ensure gold answers are labeled "reference"
if "reference" not in dataset.column_names:
    dataset = dataset.rename_column("correct_answer", "reference")
references = dataset["reference"]

# ============================
# 3. Metrics
# ============================
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

results = {}
all_predictions = pd.DataFrame({"reference": references, "question": dataset["question"]})

# ============================
# 4. Model Evaluation Loop
# ============================
for label, model_name in model_names.items():
    print(f"\n⏳ Generating predictions with {label}...")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # Generate predictions
    def generate_prediction(example):
        inputs = tokenizer(example["question"], return_tensors="pt", truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=32, num_beams=4)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    predictions = dataset.map(lambda ex: {f"{label}_prediction": generate_prediction(ex)}, batched=False)
    preds = predictions[f"{label}_prediction"]

    # Save predictions to DataFrame
    all_predictions[label] = preds

    # Compute metrics
    rouge_res = rouge.compute(predictions=preds, references=references)
    bleu_res = bleu.compute(predictions=preds, references=references)
    bert_res = bertscore.compute(predictions=preds, references=references, lang="en")
    avg_bert_f1 = sum(bert_res["f1"]) / len(bert_res["f1"])

    results[label] = {
        "ROUGE-1": rouge_res["rouge1"],
        "ROUGE-2": rouge_res["rouge2"],
        "ROUGE-L": rouge_res["rougeL"],
        "BLEU": bleu_res["bleu"],
        "BERTScore-F1": avg_bert_f1,
    }

# ============================
# 5. Save Results
# ============================
# Scores
results_df = pd.DataFrame(results).T
results_df.to_csv("evaluation_scores.csv")
print("\n✅ Scores saved to evaluation_scores.csv")

# Predictions
all_predictions.to_csv("all_predictions.csv", index=False)
print("✅ Predictions saved to all_predictions.csv")

# ============================
# 6. Visualization
# ============================
# Plot all metrics as grouped bar chart
results_df.plot(kind="bar", figsize=(12, 6))
plt.title("Model Evaluation on SciQ")
plt.ylabel("Score")
plt.xlabel("Models")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()
plt.savefig("evaluation_scores.png")
plt.show()

print("\n📊 Visualization saved as evaluation_scores.png")
