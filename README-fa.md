# موتور RAG

یک سامانه ماژولار **Retrieval-Augmented Generation (تولید تقویت‌شده با بازیابی)** برای پاسخ‌گویی مستند بر اساس فایل‌های PDF، Markdown و TXT.

## قابلیت‌ها

- بارگذاری اسناد و استخراج متن
- قطعه‌بندی متن با هم‌پوشانی (Chunking)
- ساخت بردارهای معنایی با Sentence Transformers
- ذخیره‌سازی پایدار در ChromaDB
- بازیابی ترکیبی Dense و Sparse با BM25
- ادغام رتبه‌ها با Reciprocal Rank Fusion (RRF)
- تولید پاسخ مستند و API مبتنی بر FastAPI
- ارزیابی با Precision@K، Recall@K، Reciprocal Rank و Hit Rate@K
- بنچمارک مقایسه‌ای Dense و Hybrid و تست‌های خودکار

## راه‌اندازی

Python نسخه 3.11 یا 3.12 پیشنهاد می‌شود.

```bash
git clone https://github.com/mehrshadh00/rag-engine.git
cd rag-engine
python -m venv .venv
# Windows CMD: .venv\Scripts\activate.bat
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

برای تولید پاسخ باید `OPENAI_API_KEY` را در فایل `.env` قرار دهید. Ingestion و Retrieval از مدل embedding محلی استفاده می‌کنند و به کلید API مدل زبانی نیاز ندارند.

## معماری بازیابی ترکیبی

```text
Documents -> Loaders -> Chunking -> Embeddings -> ChromaDB
                              |                     |
                              +------> BM25 Index <-+
                                          |
Question -> Dense Search + BM25 Search
                    |
          Reciprocal Rank Fusion
                    |
             Ranked Chunks -> Context -> LLM
                                      |
                              Answer + Citations
```

- **Dense Retrieval:** جست‌وجوی معنایی با embedding و ChromaDB.
- **Sparse Retrieval:** الگوریتم Okapi BM25 روی متن chunkها؛ بدون وابستگی خارجی.
- **RRF:** ترکیب رتبه‌ها، به‌جای مقایسه مستقیم امتیازهای ناهم‌مقیاس Dense و BM25.

فرمول امتیاز RRF:

`score(d) = Σ weight / (rrf_k + rank(d))`

وزن پیش‌فرض هر دو مسیر ۱ و مقدار `RRF_K=60` است. ایندکس BM25 هنگام شروع Pipeline از داده‌های پایدار ChromaDB ساخته و پس از ingestion به‌روزرسانی می‌شود.

## تنظیمات

| متغیر | کاربرد |
|---|---|
| `RETRIEVAL_MODE` | حالت `hybrid` (پیش‌فرض) یا `dense` |
| `HYBRID_CANDIDATE_K` | تعداد کاندیداهای هر مسیر پیش از ادغام |
| `RRF_K` | پارامتر هموارسازی RRF |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | وزن مسیرهای بازیابی |
| `EMBEDDING_MODEL` | مدل embedding |
| `CHROMA_PATH` | مسیر پایگاه برداری |
| `CHROMA_COLLECTION` | نام Collection |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | اندازه chunk و هم‌پوشانی |
| `TOP_K` | تعداد نتایج نهایی |
| `OPENAI_API_KEY` | کلید API برای تولید پاسخ |

برای استفاده از خط مبنای قبلی، مقدار `RETRIEVAL_MODE=dense` را تنظیم کنید.

## تست و بنچمارک مقایسه‌ای

```bash
pytest

# مقایسه Dense و Hybrid
python scripts/run_benchmark.py

# تغییر K
python scripts/run_benchmark.py --k 1
python scripts/run_benchmark.py --k 3

# اجرای جداگانه هر روش
python scripts/run_benchmark.py --mode dense
python scripts/run_benchmark.py --mode hybrid
```

بنچمارک شامل سه سند فنی و ۲۰ پرسش برچسب‌گذاری‌شده است. هر دو روش روی corpus، برچسب‌ها و K یکسان ارزیابی می‌شوند و خروجی JSON شامل معیارهای کلی و جزئیات هر سؤال است. این بنچمارک کوچک ابزار تشخیصی است، نه رتبه‌بندی عمومی مدل‌ها.

## محدودیت‌های فعلی

- Tokenizer فعلی BM25 حروف و اعداد انگلیسی را lowercase می‌کند؛ stemming و تحلیل‌گرهای زبان‌محور هنوز اضافه نشده‌اند.
- PDF اسکن‌شده به OCR نیاز دارد.
- ادغام Hybrid در سطح chunk انجام می‌شود؛ ارزیابی سطح سند chunkها را بر اساس source یکتا می‌کند.
- Cross-Encoder Reranking در برنامه توسعه قرار دارد.
- امتیاز بالای یک بنچمارک کوچک تضمین‌کننده عملکرد عمومی یا آمادگی Production نیست.
