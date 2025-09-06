import pandas as pd
import evaluate

# 1. تحميل الداتا من ملف CSV
# تأكدي إن الملف في نفس فولدر المشروع أو حطي المسار الكامل
data = pd.read_csv("my_data.csv")

# 2. تجهيز المرجع والتنبؤات لكل نظام
references = data["reference"].tolist()

systems = {
    "RAG": data["rag_prediction"].tolist(),
    "DR-RAG": data["dr_rag_prediction"].tolist(),
    "DR-RAG+CAG": data["dr_rag_cag_prediction"].tolist(),
}

# 3. تحميل المقاييس
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

# 4. حساب المقاييس لكل نظام
results = {}

for system_name, predictions in systems.items():
    print(f"\n🔹 Evaluating {system_name}...")
    
    # ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=references)
    
    # BLEU
    bleu_result = bleu.compute(predictions=predictions, references=references)
    
    # BERTScore
    bert_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    avg_bert_f1 = sum(bert_result["f1"]) / len(bert_result["f1"])
    
    results[system_name] = {
        "ROUGE-1": rouge_result["rouge1"],
        "ROUGE-2": rouge_result["rouge2"],
        "ROUGE-L": rouge_result["rougeL"],
        "BLEU": bleu_result["bleu"],
        "BERTScore-F1": avg_bert_f1,
    }

# 5. طباعة النتائج
print("\n===== 📊 Final Evaluation Results =====")
for system, metrics in results.items():
    print(f"\nSystem: {system}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")