import argparse, json, os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from utils.io_jsonl import write_jsonl
from utils.openai_wrap import chat_json

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH","./chroma")

def retrieve_examples(collection, query: str, subject: str | None, top_k: int = 6):
    where = {}
    if subject:
        where["subject"] = subject
    res = collection.query(query_texts=[query], n_results=top_k, where=where or None)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    lines = []
    for d,m in zip(docs, metas):
        lines.append(f"- ({m.get('source','')}) {d}")
    return "\n".join(lines)

def load_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def generate_set(with_retrieval: bool, subject, topic, qtype, difficulty, n):
    # Note: This uses the same prompt but optionally with empty retrieval block.
    tpl = load_template("prompts/qg_prompt.txt")
    retrieved_block = "" if not with_retrieval else "<retrieval included in judge prompt below>"
    prompt = (
        tpl.replace("{{subject}}", subject)
           .replace("{{topic}}", topic)
           .replace("{{qtype}}", qtype)
           .replace("{{difficulty}}", difficulty)
           .replace("{{n}}", str(n))
           .replace("{{retrieved_block}}", retrieved_block)
    )
    messages = [
        {"role":"system","content":"You are a strict exam question generator that outputs pure JSON."},
        {"role":"user","content": prompt}
    ]
    out = chat_json(messages, max_tokens=1800, temperature=0.6)
    return out.get("questions", [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--qtype", choices=["mcq","tf"], default="mcq")
    ap.add_argument("--difficulty", choices=["easy","medium","hard"], default="medium")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--collection", default="exam_bank")
    ap.add_argument("--top_k", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(args.collection)
    retrieved_block = retrieve_examples(collection, args.topic, args.subject, args.top_k)

    rag_set = generate_set(True, args.subject, args.topic, args.qtype, args.difficulty, args.n)
    norag_set = generate_set(False, args.subject, args.topic, args.qtype, args.difficulty, args.n)

    judge_tpl = load_template("prompts/judge_rubric.txt")
    judge_prompt = (
        judge_tpl.replace("{{subject}}", args.subject)
                 .replace("{{topic}}", args.topic)
                 .replace("{{qtype}}", args.qtype)
                 .replace("{{difficulty}}", args.difficulty)
                 .replace("{{retrieved_block}}", retrieved_block)
                 .replace("{{rag_block}}", json.dumps(rag_set, ensure_ascii=False, indent=2))
                 .replace("{{norag_block}}", json.dumps(norag_set, ensure_ascii=False, indent=2))
    )
    messages = [
        {"role":"system","content":"You are an impartial exam-quality judge that outputs JSON only."},
        {"role":"user","content": judge_prompt}
    ]
    verdict = chat_json(messages, max_tokens=800, temperature=0.0)

    result = {
        "subject": args.subject,
        "topic": args.topic,
        "qtype": args.qtype,
        "difficulty": args.difficulty,
        "retrieved_block": retrieved_block,
        "rag_set": rag_set,
        "norag_set": norag_set,
        "judge": verdict
    }

    out = args.out or f"outputs/compare_{args.subject}_{args.topic}_{args.qtype}_{args.difficulty}.jsonl"
    write_jsonl([result], out)
    print(f"Saved comparison to {out}")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
