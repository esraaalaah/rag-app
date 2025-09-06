import os, json, pandas as pd, streamlit as st
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from utils.openai_wrap import chat_json
from utils.io_jsonl import write_jsonl
from utils.cache import cache_load, cache_save, cache_key_from_params, history_load, history_append

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH","./chroma")

st.set_page_config(page_title="RAG Question Generator", page_icon="📝", layout="centered")
st.title("📝 RAG Question Generator (MCQ / TF) — Dynamic RAG + Bloom + Cache + Context-Aware")


with st.sidebar:
    st.header("Settings")
    subject = st.text_input("Subject", value="science")
    topic = st.text_input("Topic", value="photosynthesis")
    qtype = st.selectbox("Question Type", ["mcq","tf"], index=0)
    difficulty = st.selectbox("Difficulty", ["easy","medium","hard"], index=1)
    bloom_level = st.selectbox("Bloom Level", ["remember","understand","apply","analyze","evaluate","create"], index=1)
    n = st.number_input("Number of questions", min_value=1, max_value=20, value=5, step=1)
    max_k = st.slider("Max retrieved examples (dynamic)", min_value=4, max_value=20, value=12, step=1)
    use_cache = st.checkbox("Use cache when available", value=True)

client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection("exam_bank")


def dynamic_retrieve(query: str, subject: str | None, max_k: int = 12, min_k: int = 4, distance_delta: float = 0.25):
    where = {}
    if subject:
        where["subject"] = subject
    res = collection.query(query_texts=[query], n_results=max_k, where=where or None, include=["documents","metadatas","distances"])    
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0]*len(docs)
    if not docs:
        return ""
    best = dists[0] if dists else 0.0
    selected = []
    for d,m,dist in zip(docs, metas, dists):
        if len(selected) < min_k or (dist is not None and dist <= best + 0.25):
            selected.append((d,m,dist))
    if not selected:
        selected = list(zip(docs[:min_k], metas[:min_k], dists[:min_k]))
    lines = []
    for d,m,dist in selected:
        lines.append(f"- ({m.get('source','')}) {d}")
    return "\n".join(lines)

def build_history_block(history_items, max_lines=6):
    lines = []
    for item in history_items[-max_lines:]:
        stem = item.get("stem","");
        subject = item.get("subject","");
        topic = item.get("topic","");
        lines.append(f"* [{subject}/{topic}] {stem}")
    return "\n".join(lines)

with st.form("gen"):
    st.subheader("Generate")
    submitted = st.form_submit_button("Generate Now")
    if submitted:
        # Cache check
        params = {
            "subject": subject, "topic": topic, "qtype": qtype,
            "difficulty": difficulty, "bloom_level": bloom_level, "n": int(n)
        }
        cache_key = cache_key_from_params(params)
        if use_cache:
            cache = cache_load()
            if cache_key in cache:
                st.info("Loaded from cache.")
                records = cache[cache_key]
                df = pd.DataFrame(records)
                st.dataframe(df)
            else:
                # Build prompt with context-aware blocks
                retrieved_block = dynamic_retrieve(topic, subject, max_k=max_k)
                history_items = history_load(limit=20)
                history_block = build_history_block(history_items, max_lines=6)

                with open("prompts/qg_prompt.txt","r",encoding="utf-8") as f:
                    prompt = (
                        f.read()
                        .replace("{{subject}}", subject)
                        .replace("{{topic}}", topic)
                        .replace("{{qtype}}", qtype)
                        .replace("{{difficulty}}", difficulty)
                        .replace("{{bloom_level}}", bloom_level)
                        .replace("{{n}}", str(int(n)))
                        .replace("{{retrieved_block}}", retrieved_block)
                        .replace("{{history_block}}", history_block)
                    )
                messages = [
                    {"role":"system","content":"You are a strict exam question generator that outputs pure JSON."},
                    {"role":"user","content": prompt}
                ]
                result = chat_json(messages, max_tokens=2200, temperature=0.4)
                questions = result.get("questions", [])
                if qtype == "tf":
                    for q in questions:
                        q["options"] = ["True","False"]
                        if str(q.get("answer_idx","0")) not in ["0","1",0,1]:
                            q["answer_idx"] = 0
                # enrich with metadata
                records = [{
                    "id": f"gen-{subject}-{topic}-{bloom_level}-{i}",
                    "subject": subject, "topic": topic, "type": qtype,
                    "stem": q["stem"], "options": q["options"],
                    "answer_idx": q["answer_idx"],
                    "explanation": q.get("explanation",""),
                    "bloom_level": bloom_level, "difficulty": difficulty
                } for i,q in enumerate(questions)]
                # save to cache + history
                cache[cache_key] = records
                cache_save(cache)
                history_append(records)
                df = pd.DataFrame(records)
                st.dataframe(df)
        else:
            retrieved_block = dynamic_retrieve(topic, subject, max_k=max_k)
            history_items = history_load(limit=20)
            history_block = build_history_block(history_items, max_lines=6)

            with open("prompts/qg_prompt.txt","r",encoding="utf-8") as f:
                prompt = (
                    f.read()
                    .replace("{{subject}}", subject)
                    .replace("{{topic}}", topic)
                    .replace("{{qtype}}", qtype)
                    .replace("{{difficulty}}", difficulty)
                    .replace("{{bloom_level}}", bloom_level)
                    .replace("{{n}}", str(int(n)))
                    .replace("{{retrieved_block}}", retrieved_block)
                    .replace("{{history_block}}", history_block)
                )
            messages = [
                {"role":"system","content":"You are a strict exam question generator that outputs pure JSON."},
                {"role":"user","content": prompt}
            ]
            result = chat_json(messages, max_tokens=2200, temperature=0.4)
            questions = result.get("questions", [])
            if qtype == "tf":
                for q in questions:
                    q["options"] = ["True","False"]
                    if str(q.get("answer_idx","0")) not in ["0","1",0,1]:
                        q["answer_idx"] = 0
            records = [{
                "id": f"gen-{subject}-{topic}-{bloom_level}-{i}",
                "subject": subject, "topic": topic, "type": qtype,
                "stem": q["stem"], "options": q["options"],
                "answer_idx": q["answer_idx"],
                "explanation": q.get("explanation",""),
                "bloom_level": bloom_level, "difficulty": difficulty
            } for i,q in enumerate(questions)]
            history_append(records)
            df = pd.DataFrame(records)
            st.dataframe(df)

        # Save to disk
        os.makedirs("outputs", exist_ok=True)
        out = f"outputs/{subject}_{topic}_{qtype}_{difficulty}_{bloom_level}_{int(n)}.jsonl"
        write_jsonl(records, out)
        csv_path = out.replace(".jsonl",".csv")
        pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
        st.success(f"Saved JSONL to {out} and CSV to {csv_path}.")
