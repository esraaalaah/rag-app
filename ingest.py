import argparse, os, json
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

from utils.io_jsonl import read_jsonl
from utils.embedder import STEmbeddingFunction

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma")

def build_text(row):
    # A compact representation to embed and retrieve by semantic similarity
    stem = row.get("stem","").strip()
    subject = row.get("subject","")
    topic = row.get("topic","")
    source = row.get("source","")
    return f"[subject:{subject}] [topic:{topic}] {stem} (source:{source})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL file (unified schema)")
    ap.add_argument("--collection", default="exam_bank", help="Chroma collection name")
    ap.add_argument("--subject", default=None, help="Optional subject tag to store as metadata filter")
    args = ap.parse_args()

    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    emb_fn = STEmbeddingFunction()
    col = client.get_or_create_collection(name=args.collection, embedding_function=emb_fn, metadata={"hnsw:space":"cosine"})

    ids = []
    docs = []
    metas = []

    for row in read_jsonl(args.input):
        qid = row["id"]
        ids.append(qid)
        docs.append(build_text(row))
        meta = {
            "subject": row.get("subject",""),
            "topic": row.get("topic",""),
            "type": row.get("type","mcq"),
            "source": row.get("source","")
        }
        metas.append(meta)

    # Chroma upsert
    step = 1000
    for i in range(0, len(ids), step):
        col.upsert(
            ids=ids[i:i+step],
            documents=docs[i:i+step],
            metadatas=metas[i:i+step]
        )
    print(f"Ingested {len(ids)} items into collection '{args.collection}' at {CHROMA_PATH}")

if __name__ == "__main__":
    main()
