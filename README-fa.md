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

ماژول ارزیابی می‌تواند مجموعه‌ای از پرسش‌های برچسب‌گذاری‌شده را روی هر Retriever سازگار اجرا کند و هم نتایج هر پرسش و هم میانگین معیارها را برگرداند. تست‌ها برای منطق اصلی از fakeها استفاده می‌کنند و به کلید API یا دانلود مدل embedding نیاز ندارند.

## محدودیت‌های فعلی

- PDF اسکن‌شده به OCR نیاز دارد و فعلاً پشتیبانی نمی‌شود.
- بازیابی فعلی dense است؛ Hybrid Search و Cross-Encoder Reranking در برنامه توسعه قرار دارند.
- لایه تولید پاسخ در حال حاضر از OpenAI Chat Completions استفاده می‌کند.
- این پروژه در حال توسعه است و ادعای گواهی یا آمادگی Production ندارد.
