import os, json, hashlib

CACHE_FILE = "outputs/cache.json"
HISTORY_FILE = "outputs/history.jsonl"

def _ensure_dirs():
    os.makedirs("outputs", exist_ok=True)

def _load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return default
    return default

def _save_json(path, obj):
    _ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def cache_load():
    return _load_json(CACHE_FILE, {})

def cache_save(cache):
    _save_json(CACHE_FILE, cache)

def history_load(limit=None):
    # Read JSONL lines (safer for append)
    items = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    if limit:
        return items[-limit:]
    return items

def history_append(records):
    _ensure_dirs()
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def cache_key_from_params(params: dict) -> str:
    # Stable md5 over sorted params
    s = json.dumps(params, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode("utf-8")).hexdigest()
