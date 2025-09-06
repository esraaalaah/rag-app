# RAG-QG: Simplified RAG Question Generation (MCQ / True-False)

**لغة الشرح:** عربي + English hints  
**هدف المشروع:** نظام بسيط لتوليد أسئلة امتحانية موضوعية باستخدام RAG + LLM، مع **داتا حقيقية** تُحمَّل تلقائيًا من `Hugging Face Datasets` (مثال: SciQ).

---

## ✅ What you get
- تحميل داتا **حقيقية** (SciQ) وتحويلها إلى **بنك أسئلة MCQ**.
- بناء **متجهات** و **فهرس** (Chroma + SentenceTransformers).
- **توليد أسئلة جديدة** بأسلوب الامتحانات الواقعية عبر OpenAI LLM.
- **تجربة تفاعلية** عبر Streamlit، مع حفظ النتائج إلى JSONL/CSV.

> **ملاحظة**: المشروع مصمّم ليعمل بخطوات قليلة جدًا. يمكنك التوسعة لاحقًا بسهولة.

---

## 📦 المتطلبات (Requirements)
- Python 3.9+
- حساب OpenAI (ضع المفتاح في `.env`).
- اتصال إنترنت لأول مرة فقط لتنزيل الداتا والموديل.
- OS: Windows / macOS / Linux.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env  # ثم حط مفتاح OpenAI داخل .env
```

افتح `.env` وعدّل:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
CHROMA_PATH=./chroma
EMBEDDING_NAME=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🗂️ تحميل داتا **حقيقية** (SciQ) إلى JSONL
نستخدم سكريبت بسيط لتحميل وتحويل SciQ إلى بنك أسئلة MCQ:

```bash
python data/download_dataset.py --dataset sciq --split train --out data/sciq_train.jsonl --subject science
```

أوامر إضافية (اختيارية):
```bash
# جزء من الداتا (لتجربة سريعة)
python data/download_dataset.py --dataset sciq --split validation --out data/sciq_val.jsonl --subject science --limit 200

# تحميل ARC-Challenge (MCQ صعبة)
python data/download_dataset.py --dataset arc --config ARC-Challenge --split train --out data/arc_challenge_train.jsonl --subject science
```

> الناتج يكون في صيغة JSONL موحدة (سطر لكل سؤال):
```json
{"id":"...", "subject":"science", "topic":"", "type":"mcq", "stem":"...", "options":["A","B","C","D"], "answer_idx":0, "source":"sciq"}
```

---

## 🔎 بناء الفهرس (Ingestion)
حوّل ملف JSONL إلى متجهات وخزّنه في Chroma:

```bash
python ingest.py --input data/sciq_train.jsonl --subject science --collection exam_bank
```

- سيُنشئ/يُحدّث مسار `CHROMA_PATH` المحدد في `.env`.
- يمكنك تكرار الأمر مع ملفات أخرى لدمجها في نفس الـ collection.

---

## ✨ توليد أسئلة جديدة (RAG Generation)
مثال: توليد 5 أسئلة MCQ حول "photosynthesis" بصعوبة متوسطة:

```bash
python generate.py --subject science --topic "photosynthesis" --qtype mcq --difficulty medium --n 5 --collection exam_bank --out outputs/questions_photosynthesis.jsonl
```

سيتم:
1) استرجاع أمثلة قريبة من الفهرس.  
2) تمريرها للـ LLM عبر prompt مُحكَم.  
3) حفظ أسئلة جديدة بصيغة JSONL + CSV.

---

## 🖥️ واجهة تفاعلية (Streamlit)
```bash
streamlit run app.py
```
- حدّد **Subject** و **Topic** و **نوع السؤال** و **الصعوبة** وعدد الأسئلة.  
- اعرض الأسئلة، نزّلها CSV أو JSONL.

---

## 🧪 مقارنة سريعة (Optional)
لمقارنة **with-RAG** مقابل **no-RAG** على نفس الإعدادات، وشوف الفارق في الطابع الامتحاني:
```bash
python eval_pairwise.py --subject science --topic "photosynthesis" --qtype mcq --difficulty medium --n 5 --collection exam_bank --out outputs/compare_photosynthesis.jsonl
```
- السكريبت يطلب من LLM تقييم أي مخرجات أقرب لأسلوب الامتحان (Rubric مبسّط).

---

## 📁 هيكل المشروع
```
rag-qg/
  app.py
  ingest.py
  generate.py
  eval_pairwise.py
  requirements.txt
  .env.example
  prompts/
    qg_prompt.txt
    judge_rubric.txt
  data/
    download_dataset.py
    README.md
  utils/
    io_jsonl.py
    openai_wrap.py
    embedder.py
  outputs/  (سيُنشأ تلقائياً)
```

---

## 🔐 الملاحظات
- ضَع مفتاح OpenAI في `.env` فقط (لا تضعه داخل الكود).  
- SciQ و ARC datasets يتم تنزيلهما قانونيًا من Hugging Face في جهازك.  

بالتوفيق! ✨


## 🆕 What’s new
- **Bloom Levels**: remember / understand / apply / analyze / evaluate / create.
- **Dynamic RAG**: adaptive retrieval size based on similarity.
- **Cache-Augmented Generation (CAG)**: reuse previous generations for identical params.
- **Context-Aware Generation**: inject meta + history context into prompts to avoid duplicates and stay on-topic.



