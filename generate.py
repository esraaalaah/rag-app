import argparse, os, json, pandas as pd
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from utils.openai_wrap import chat_json
from utils.io_jsonl import write_jsonl
from utils.cache import cache_load, cache_save, cache_key_from_params, history_load, history_append

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

def dynamic_retrieve(collection, query: str, subject: str | None, max_k: int = 12, min_k: int = 4, distance_delta: float = 0.25):
    """Retrieve adaptively: start with top results, keep those close to the best distance.
    For cosine distance (smaller better), we keep items whose distance <= best + delta.
    Ensure at least min_k items as a fallback.
    """
    where = {}
    if subject:
        where["subject"] = subject
    res = collection.query(query_texts=[query], n_results=max_k, where=where or None, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0]*len(docs)
    if not docs:
        return []
    best = dists[0] if dists else 0.0
    selected = []
    for d,m,dist in zip(docs, metas, dists):
        if len(selected) < min_k or (dist is not None and dist <= best + distance_delta):
            selected.append((d,m,dist))
    if not selected:
        selected = list(zip(docs[:min_k], metas[:min_k], dists[:min_k]))
    # Pretty block
    lines = []
    for d,m,dist in selected:
        src = m.get("source","");
        lines.append(f"- ({src}) {d}")
    return "\n".join(lines)

def build_history_block(history_items, max_lines=6):
    lines = []
    for item in history_items[-max_lines:]:
        stem = item.get("stem","");
        subject = item.get("subject","");
        topic = item.get("topic","");
        lines.append(f"* [{subject}/{topic}] {stem}")
    return "\n".join(lines)

def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Subject tag used in ingestion (e.g., science)")
    ap.add_argument("--topic", required=True, help="Topic to generate questions about")
    ap.add_argument("--qtype", choices=["mcq","tf"], default="mcq")
    ap.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    ap.add_argument("--bloom_level", choices=["remember","understand","apply","analyze","evaluate","create"], default="understand")
    ap.add_argument("--n", type=int, default=5, help="Number of questions to generate")
    ap.add_argument("--collection", default="exam_bank", help="Chroma collection name")
    ap.add_argument("--max_k", type=int, default=12, help="Max retrieved examples for style guidance")
    ap.add_argument("--prompt_path", default="prompts/qg_prompt.txt")
    ap.add_argument("--out", default=None, help="Output JSONL path; also creates CSV with same stem")
    ap.add_argument("--use_cache", action="store_true", help="Use cache to reuse prior generations")    
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(args.collection)

    # ----- Cache check
    params = {
        "subject": args.subject, "topic": args.topic, "qtype": args.qtype,
        "difficulty": args.difficulty, "bloom_level": args.bloom_level, "n": args.n
    }
    cache_key = cache_key_from_params(params)
    if args.use_cache:
        cache = cache_load()
        if cache_key in cache:
            print("Loaded from cache.")
            records = cache[cache_key]
            # Save also to outputs (JSONL/CSV) for convenience
            out = args.out or f"outputs/{args.subject}_{args.topic}_{args.qtype}_{args.difficulty}_{args.bloom_level}_{args.n}.jsonl"
            write_jsonl(records, out)
            import pandas as pd
            df = pd.DataFrame(records)
            csv_path = out.replace(".jsonl",".csv")
            os.makedirs("outputs", exist_ok=True)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return

    # ----- Dynamic RAG
    retrieved_block = dynamic_retrieve(collection, query=args.topic, subject=args.subject, max_k=args.max_k)

    # ----- History Context
    history_items = history_load(limit=20)
    history_block = build_history_block(history_items, max_lines=6)

    # ----- Prompt (Context-Aware)
    prompt_template = load_template(args.prompt_path)
    prompt = (
        prompt_template
        .replace("{{subject}}", args.subject)
        .replace("{{topic}}", args.topic)
        .replace("{{qtype}}", args.qtype)
        .replace("{{difficulty}}", args.difficulty)
        .replace("{{bloom_level}}", args.bloom_level)
        .replace("{{n}}", str(args.n))
        .replace("{{retrieved_block}}", retrieved_block)
        .replace("{{history_block}}", history_block)
    )

    messages = [
        {"role":"system","content":"You are a strict exam question generator that outputs pure JSON."},
        {"role":"user","content": prompt}
    ]

    result = chat_json(messages, max_tokens=2200, temperature=0.4)
    questions = result.get("questions", [])
    # Normalize TF options if necessary
    norm = []
    for q in questions:
        if args.qtype == "tf":
            q["options"] = ["True", "False"]
            if str(q.get("answer_idx","0")) not in ["0","1",0,1]:
                q["answer_idx"] = 0
        # enforce bloom & difficulty in output
        q["bloom_level"] = args.bloom_level
        q["difficulty"] = args.difficulty
        norm.append(q)

    # Add metadata and save
    records = []
    for i, q in enumerate(norm):
        rec = {
            "id": f"gen-{args.subject}-{args.topic}-{args.bloom_level}-{i}",
            "subject": args.subject,
            "topic": args.topic,
            "type": args.qtype,
            "stem": q["stem"],
            "options": q["options"],
            "answer_idx": q["answer_idx"],
            "explanation": q.get("explanation",""),
            "bloom_level": q.get("bloom_level", args.bloom_level),
            "difficulty": q.get("difficulty", args.difficulty)
        }
        records.append(rec)

    out = args.out or f"outputs/{args.subject}_{args.topic}_{args.qtype}_{args.difficulty}_{args.bloom_level}_{args.n}.jsonl"
    write_jsonl(records, out)

    # Also export CSV for convenience
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(records)
    csv_path = out.replace(".jsonl",".csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ----- Save to cache and history
    cache = cache_load()
    cache[cache_key] = records
    cache_save(cache)
    history_append(records)

    print(f"Saved {len(records)} questions to: {out}")
    print(f"CSV also saved to: {csv_path}")

if __name__ == "__main__":
    main()
