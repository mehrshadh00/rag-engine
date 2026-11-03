# موتور RAG

یک سامانه ماژولار **Retrieval-Augmented Generation (تولید تقویت‌شده با بازیابی)** برای پاسخ‌گویی مستند بر اساس فایل‌های PDF، Markdown و TXT.

## قابلیت‌ها

- بارگذاری اسناد و استخراج متن
- قطعه‌بندی متن با هم‌پوشانی (Chunking)
- ساخت بردارهای معنایی با Sentence Transformers
- ذخیره‌سازی پایدار در ChromaDB
- بازیابی معنایی و ساخت Context
- تولید پاسخ با ارجاع به منبع و دستور خودداری از حدس‌زدن
- API مبتنی بر FastAPI
- معیارهای ارزیابی بازیابی شامل Precision@K، Recall@K، Reciprocal Rank و Hit Rate@K، به‌همراه تست‌های خودکار

## راه‌اندازی

Python نسخه 3.11 یا 3.12 پیشنهاد می‌شود.

```bash
git clone https://github.com/mehrshadh00/rag-engine.git
cd rag-engine
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

برای تولید پاسخ باید `OPENAI_API_KEY` را در فایل `.env` قرار دهید. بارگذاری، embedding و بازیابی از مدل محلی Sentence Transformers و پایگاه برداری محلی استفاده می‌کنند.

فایل‌ها را داخل پوشه `data/` قرار دهید و اجرا کنید:

```bash
python -m rag_engine.cli ingest data
python -m rag_engine.cli ask "اسناد درباره هوش مصنوعی چه می‌گویند؟"
```

اجرای API:

```bash
uvicorn rag_engine.api:app --reload
```

- `POST /ingest` با ورودی `{"directory": "data"}`
- `POST /ask` با ورودی `{"question": "سؤال شما", "top_k": 5}`
- `GET /health`

## معماری

```text
Documents -> Loaders -> Chunking -> Embeddings -> ChromaDB
                                                   |
Question -> Query Embedding -> Retrieval -> Context
                                                   |
                                               LLM
                                                   |
                                         Answer + Citations
```

## تنظیمات

تنظیمات در `.env.example` مستند شده‌اند؛ از جمله مدل embedding، مسیر پایگاه داده، اندازه chunk، میزان overlap، تعداد نتایج بازیابی و مدل زبانی.

## تست

```bash
pytest
```

ماژول ارزیابی معیارهای Precision@K، Recall@K، Reciprocal Rank و Hit Rate@K را محاسبه می‌کند. تابع `evaluate_retrieval()` مجموعه پرسش‌های برچسب‌گذاری‌شده را روی هر Retriever سازگار اجرا می‌کند و نتایج تک‌تک پرسش‌ها و میانگین معیارها را برمی‌گرداند. تست‌های واحد به کلید API یا دانلود مدل embedding نیاز ندارند.

### بنچمارک قابل تکرار بازیابی

یک بنچمارک کوچک و دستی شامل سه سند فنی و ۲۰ پرسش طبیعی به پروژه اضافه شده است. برچسب‌گذاری در سطح سند انجام می‌شود؛ یعنی سندی مرتبط است که شواهد لازم برای پاسخ را در خود داشته باشد. اسکریپت، chunkهای رتبه‌بندی‌شده را به اسناد یکتا تبدیل می‌کند و رتبه هر سند را بر اساس بهترین chunk آن در نظر می‌گیرد.

```bash
python scripts/run_benchmark.py
# تغییر تعداد اسناد یکتای ارزیابی‌شده
python scripts/run_benchmark.py --k 2
```

نتیجه به‌صورت JSON شامل معیارهای کلی و جزئیات هر پرسش چاپ می‌شود. بنچمارک از مدل embedding محلی تنظیم‌شده و یک Collection جداگانه در مسیر `.benchmark_chroma/` استفاده می‌کند و نیازی به فراخوانی API مدل زبانی ندارد؛ البته ممکن است اجرای اول مدل embedding را دانلود کند. مقدار پیش‌فرض `k=2` انتخاب شده تا رتبه‌بندی در مجموعه سه‌سندی معنادار بماند؛ برای بررسی برش‌های دیگر می‌توان از `--k 1` یا `--k 3` استفاده کرد. این بنچمارک یک خط مبنای شفاف برای مقایسه تغییرات است، نه رتبه‌بندی عمومی مدل‌ها. پرسش‌ها و برچسب‌ها در فایل `benchmarks/datasets/rag_fundamentals_v1.json` قرار دارند.

## محدودیت‌های فعلی

- PDF اسکن‌شده به OCR نیاز دارد و فعلاً پشتیبانی نمی‌شود.
- بازیابی فعلی dense است؛ Hybrid Search و Cross-Encoder Reranking در برنامه توسعه قرار دارند.
- لایه تولید پاسخ در حال حاضر از OpenAI Chat Completions استفاده می‌کند.
- این پروژه در حال توسعه است و ادعای گواهی یا آمادگی Production ندارد.


## ساختار بنچمارک

- `benchmarks/corpus/`: سه سند مرجع فنی
- `benchmarks/datasets/`: مجموعه پرسش‌ها و برچسب‌های نسخه‌بندی‌شده
- `scripts/run_benchmark.py`: اجرای بنچمارک و تولید گزارش JSON
