# data/
Use `download_dataset.py` to fetch real datasets (SciQ, ARC) from Hugging Face and convert them into a unified JSONL format used by the pipeline.

Examples:
- SciQ (science MCQ, has correct answer + distractors)
- ARC (AI2 Reasoning Challenge; multiple-choice with answer keys)

**Command examples:**
```bash
python data/download_dataset.py --dataset sciq --split train --out data/sciq_train.jsonl --subject science
python data/download_dataset.py --dataset sciq --split validation --out data/sciq_val.jsonl --subject science --limit 200
python data/download_dataset.py --dataset arc --config ARC-Challenge --split train --out data/arc_challenge_train.jsonl --subject science
```
