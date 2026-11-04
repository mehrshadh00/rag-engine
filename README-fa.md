# موتور RAG

یک سامانه ماژولار **Retrieval-Augmented Generation (تولید تقویت‌شده با بازیابی)** برای پاسخ‌گویی مستند بر اساس فایل‌های PDF، Markdown و TXT.

## قابلیت‌ها

- بارگذاری اسناد و استخراج متن
- قطعه‌بندی متن با هم‌پوشانی (Chunking)
- ساخت بردارهای معنایی با Sentence Transformers
- ذخیره‌سازی پایدار بردارها در ChromaDB
- بازیابی متراکم (Dense) و تنک (Sparse) با الگوریتم BM25
- ادغام رتبه‌ها با Reciprocal Rank Fusion (RRF)
- رتبه‌بندی مجدد اختیاری با Cross-Encoder
- تشخیص پاسخ‌پذیری بر اساس امتیاز Cross-Encoder و خودداری از پاسخ در صورت شواهد ضعیف
- ساخت زمینه همراه با منبع و تولید پاسخ با مدل سازگار با OpenAI
- ارائه API برای ورود اسناد و پرسش‌وپاسخ با FastAPI
- بنچمارک نسخه‌بندی‌شده و تست‌های خودکار

## معماری

```text
Files -> Loaders -> Chunker -> Embeddings -> ChromaDB
                              |                 |
                              +-> BM25 Index <--+
                                      |
Question -> Dense Search + BM25 Search
                    |
             Reciprocal Rank Fusion
                    |
              Candidate Chunks
                    |
        Optional Cross-Encoder Reranking
                    |
        Ranked Chunks -> Context -> LLM
                                      |
                              Answer + Citations
```

## راه‌اندازی

Python نسخه 3.11 یا 3.12 پیشنهاد می‌شود.

```bash
git clone https://github.com/mehrshadh00/rag-engine.git
cd rag-engine
python -m venv .venv
# Windows CMD: .venv\\Scripts\\activate.bat
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

برای تولید پاسخ باید `OPENAI_API_KEY` را در فایل `.env` قرار دهید. Ingestion و Retrieval از مدل‌های محلی استفاده می‌کنند و برای بازیابی به کلید API مدل زبانی نیاز ندارند.

برای افزودن اسناد، فایل‌ها را در پوشه `data/` قرار دهید و سپس اجرا کنید:

```bash
python -m rag_engine.cli ingest data
python -m rag_engine.cli ask "What do the documents say about artificial intelligence?"
uvicorn rag_engine.api:app --reload
```

- `POST /ingest` با بدنه JSON مانند `{"directory": "data"}`
- `POST /ask` با بدنه JSON مانند `{"question": "Your question", "top_k": 5}`
- `GET /health`

در اجرای اول، مدل embedding از Hugging Face دریافت می‌شود. در صورت فعال‌سازی Reranker، مدل Cross-Encoder نیز در اولین استفاده دانلود خواهد شد.

## بازیابی ترکیبی و رتبه‌بندی مجدد

- **Dense Retrieval:** جست‌وجوی معنایی با embedding و ChromaDB.
- **Sparse Retrieval:** الگوریتم Okapi BM25 روی متن chunkها؛ بدون وابستگی خارجی.
- **RRF:** ترکیب رتبه‌ها، به‌جای مقایسه مستقیم امتیازهای ناهم‌مقیاس Dense و BM25.
- **Cross-Encoder Reranking:** امتیازدهی مشترک به سؤال و هر chunk و مرتب‌سازی مجدد کاندیداهای بازیابی‌شده، پیش از ساخت Context.

فرمول امتیاز RRF:

`score(d) = Σ weight / (rrf_k + rank(d))`

وزن پیش‌فرض هر دو مسیر برابر با ۱ و مقدار `RRF_K=60` است. ایندکس BM25 هنگام شروع Pipeline از داده‌های پایدار ChromaDB ساخته و پس از ingestion به‌روزرسانی می‌شود.

Reranking به‌صورت پیش‌فرض غیرفعال است. برای فعال‌سازی، مقدار `RERANK_ENABLED=true` را قرار دهید. مدل پیش‌فرض:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

Reranker فقط ترتیب کاندیداهای بازیابی‌شده را تغییر می‌دهد؛ اگر یک chunk در مرحله اول بازیابی نشده باشد، Reranker نمی‌تواند آن را بازیابی کند. همچنین اجرای آن زمان پردازش بیشتری نیاز دارد.

### تشخیص پاسخ‌پذیری و خودداری از پاسخ (Abstention)

وقتی `RERANK_ENABLED=true` و `ANSWERABILITY_ENABLED=true` باشند، Pipeline بالاترین امتیاز Cross-Encoder را پس از رتبه‌بندی بررسی می‌کند. اگر امتیاز از `ANSWERABILITY_THRESHOLD` کمتر باشد یا هیچ شواهدی بازیابی نشود، سیستم از پاسخ‌گویی خودداری می‌کند و به‌جای فراخوانی مدل تولید پاسخ، فیلدهای `abstained`، دلیل خودداری و اطلاعات امتیاز را برمی‌گرداند. با `ANSWERABILITY_ENABLED=false` می‌توان این لایه را غیرفعال کرد. فیلدهای قبلی `answer` و `sources` در پاسخ `/ask` حفظ شده‌اند و اطلاعات `answerable`، `abstained`، `abstention_reason` و `answerability` نیز اضافه می‌شوند.

آستانه پیش‌فرض (`0.67`) بر اساس ۶۰ پرسش مجموعه توسعه انتخاب شده است: ۴۱ True Positive، صفر False Positive، ۱۲ True Negative و ۷ False Negative (دقت کلی ۸۸.۳۳٪، Precision برابر ۱۰۰٪، Recall برابر ۸۵.۴۲٪ و F1 برابر ۹۲.۱۳٪). این مقدار یک نقطه عملیاتی تجربی روی Dev است و **احتمال کالیبره‌شده محسوب نمی‌شود**؛ امتیازهای Cross-Encoder به مدل و دامنه داده وابسته‌اند و فاصله امتیازها در اطراف آستانه کم است، بنابراین تعمیم آن تضمین‌شده نیست. با تغییر مدل یا Corpus، آستانه را روی یک مجموعه توسعه نماینده دوباره تنظیم کنید و سپس پیش از ارزیابی Test آن را ثابت نگه دارید. این روش یک heuristic سبک است و تضمین‌کننده صحت factual پاسخ نیست.

## تنظیمات

جزئیات کامل در فایل `.env.example` قرار دارد.

| متغیر | کاربرد |
|---|---|
| `RETRIEVAL_MODE` | حالت `hybrid` (پیش‌فرض)، `dense` یا `bm25` |
| `HYBRID_CANDIDATE_K` | تعداد کاندیداهای هر مسیر پیش از ادغام |
| `RRF_K` | پارامتر هموارسازی RRF |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | وزن مسیرهای بازیابی |
| `RERANK_ENABLED` | فعال‌سازی Cross-Encoder؛ پیش‌فرض `false` |
| `RERANKER_MODEL` | شناسه مدل Cross-Encoder |
| `RERANK_CANDIDATE_K` | تعداد chunkهای ورودی به Reranker |
| `ANSWERABILITY_ENABLED` | فعال‌سازی Abstention مبتنی بر امتیاز؛ پیش‌فرض `true` و فقط همراه Reranker |
| `ANSWERABILITY_THRESHOLD` | حداقل امتیاز Cross-Encoder برای تولید پاسخ؛ پیش‌فرض `0.67` (تنظیم‌شده با Dev) |
| `EMBEDDING_MODEL` | مدل embedding |
| `CHROMA_PATH` / `CHROMA_COLLECTION` | مسیر و نام پایگاه برداری |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | اندازه chunk و هم‌پوشانی |
| `TOP_K` | تعداد نتایج نهایی |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | کلید و مدل تولید پاسخ |

## ارزیابی و تست

اجرای تست‌ها:

```bash
pytest
```

تست‌ها از Fakeهای سبک استفاده می‌کنند و برای اجرا به کلید API یا دانلود مدل نیاز ندارند.

### دیتاست بنچمارک نسخه ۲

بنچمارک شامل ۱۵ سند فنی و ۱۰۰ پرسش برچسب‌گذاری‌شده در سطح سند است. پرسش‌ها در پنج دسته تقسیم شده‌اند:

- Direct: پرسش مستقیم و factual
- Paraphrase: بازنویسی معنایی
- Exact Term: اصطلاحات دقیق
- Multi-hop: پرسش نیازمند شواهد از چند سند
- Unanswerable: پرسش بدون پاسخ در مجموعه اسناد

برای هر دسته ۱۲ پرسش Dev و ۸ پرسش Test در نظر گرفته شده است؛ در مجموع ۶۰ پرسش توسعه و ۴۰ پرسش آزمون نهایی.

چهار روش مقایسه می‌شوند:

1. بازیابی Dense
2. بازیابی BM25
3. بازیابی Hybrid با Weighted RRF
4. بازیابی Hybrid همراه با Cross-Encoder Reranking

در مسیر Reranked، هر دو بازیاب Dense و BM25 حداکثر `HYBRID_CANDIDATE_K` کاندیدا تولید می‌کنند. سپس RRF نتایج را ادغام می‌کند و حداکثر `RERANK_CANDIDATE_K` chunk به Cross-Encoder می‌رسد. این تنظیم با مرحله تولید کاندیدا در Pipeline اصلی هم‌راستا است. رتبه‌بندی مجدد پیش از حذف اسناد تکراری انجام می‌شود؛ بنابراین K نهایی نشان‌دهنده تعداد اسناد منبع یکتا است.

### نتایج بنچمارک

تمام نتایج زیر با **K=3 سند منبع یکتا** محاسبه شده‌اند. معیارهای Precision@K، Recall@K، MRR و Hit Rate@K فقط روی پرسش‌های پاسخ‌پذیر میانگین‌گیری شده‌اند (۴۸ پرسش در Dev و ۳۲ پرسش در Test). پرسش‌های غیرقابل‌پاسخ جداگانه و از طریق نرخ بازیابی کاذب گزارش می‌شوند. نرخ شواهد کامل Multi-hop نیز فقط روی زیرمجموعه پرسش‌های چندمرحله‌ای محاسبه شده است (۱۲ پرسش Dev و ۸ پرسش Test). در زمان ارزیابی، Corpus شامل ۴۳ chunk بوده است. مقدار تنظیم‌شده برای سقف کاندیداهای هر بازیاب ۵۰ است؛ بنابراین بازیاب‌ها حداکثر همان ۴۳ chunk موجود را برگردانده‌اند. در مسیر Reranked، حداکثر ۲۰ کاندیدای ادغام‌شده به Cross-Encoder داده شده است.

**مجموعه توسعه (Dev) — ۶۰ پرسش**

| معیار | Dense | BM25 | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Precision@3 | 35.42% | 36.11% | 37.50% | **39.58%** |
| Recall@3 | 88.54% | 89.58% | 91.67% | **96.88%** |
| MRR | 88.54% | 84.38% | 87.15% | **90.63%** |
| Hit Rate@3 | 95.83% | 95.83% | 95.83% | **100.00%** |
| شواهد کامل Multi-hop | 41.67% | 50.00% | 66.67% | **75.00%** |

**مجموعه آزمون مستقل (Test) — ۴۰ پرسش**

| معیار | Dense | BM25 | Hybrid | Hybrid + Reranker |
|---|---:|---:|---:|---:|
| Precision@3 | 35.42% | 35.42% | 37.50% | **39.58%** |
| Recall@3 | 87.50% | 87.50% | 92.19% | **96.88%** |
| MRR | 83.85% | 76.56% | 84.90% | **89.58%** |
| Hit Rate@3 | 93.75% | 93.75% | 96.88% | **100.00%** |
| شواهد کامل Multi-hop | 50.00% | 50.00% | 62.50% | **75.00%** |

در این دیتاست، روش Hybrid + Reranker در هر پنج معیار گزارش‌شده و در هر دو مجموعه Dev و Test بهترین نتیجه را ثبت کرده است. در مجموعه Test، نسبت به Hybrid، مقدار Precision@3 به‌اندازه ۲.۰۸ واحد درصد، Recall@3 به‌اندازه ۴.۶۹ واحد درصد، MRR به‌اندازه ۴.۶۹ واحد درصد، Hit Rate@3 به‌اندازه ۳.۱۲ واحد درصد و شواهد کامل Multi-hop به‌اندازه ۱۲.۵۰ واحد درصد بهبود داشته است.

### تفسیر نتایج و محدودیت‌ها

- این اعداد حاصل ارزیابی روی یک Corpus ثابت و یک دیتاست برچسب‌گذاری‌شده هستند. نتایج برای مقایسه روش‌ها در شرایط یکسان مفیدند، اما عملکرد عمومی RAG در دامنه‌های دیگر یا بهبود آماری قطعی را اثبات نمی‌کنند.
- مجموعه Test فقط ۴۰ پرسش (۸ پرسش در هر دسته) دارد؛ بنابراین تغییر چند نمونه می‌تواند روی معیارها اثر محسوسی بگذارد.
- مقدار `unanswerable_false_positive_rate_at_k` برای هر چهار روش بازیابی و در هر دو Split برابر ۱.۰۰ بوده است. این معیار فقط بازیابی حداقل یک سند را اندازه می‌گیرد، نه تصمیم نهایی Pipeline برای خودداری از پاسخ. اکنون لایه اختیاری Abstention مبتنی بر آستانه Cross-Encoder اضافه شده و در صورت انتخاب حالت Reranked، معیارهای مستقل Answerability نیز در خروجی بنچمارک گزارش می‌شوند.
- Reranker فقط کاندیداهای بازیابی‌شده را مرتب می‌کند و نمی‌تواند اسناد ازدست‌رفته در مرحله اول را بازیابی کند؛ همچنین زمان استنتاج را افزایش می‌دهد.
- این بنچمارک بازیابی سند را ارزیابی می‌کند، نه صحت پاسخ نهایی، صحت ارجاع‌ها، تأخیر، توان عملیاتی یا کیفیت End-to-End تولید پاسخ را.

اجرای بنچمارک:

```bash
# مجموعه توسعه (پیش‌فرض)
python scripts/run_benchmark.py --split dev --k 3

# مجموعه آزمون نهایی؛ پس از تثبیت تصمیم‌های توسعه
python scripts/run_benchmark.py --split test --k 3

# فقط روش Hybrid همراه با Reranker
python scripts/run_benchmark.py --mode hybrid-reranked --split dev

# روش Hybrid بدون بارگذاری Cross-Encoder
python scripts/run_benchmark.py --mode hybrid --split dev
```

هر اجرا Collection اختصاصی دیتاست را پیش از ورود مجدد اسناد پاک‌سازی می‌کند. بنچمارک به API مدل زبانی متصل نمی‌شود. هنگام تنظیم پارامترها، مجموعه Test را خارج از چرخه توسعه نگه دارید.

## ساختار پروژه

```text
src/rag_engine/
  api.py
  answerability.py
  benchmark.py
  bm25.py
  chunking.py
  config.py
  embeddings.py
  evaluation.py
  generation.py
  hybrid.py
  ingestion.py
  pipeline.py
  reranking.py
  retrieval.py
  vector_store.py
benchmarks/
  corpus_v2/
  datasets/
scripts/
  run_benchmark.py
tests/
.github/workflows/tests.yml
```

## محدودیت‌های فعلی پروژه

- Tokenizer فعلی BM25 حروف و اعداد انگلیسی را lowercase می‌کند؛ stemming و تحلیل‌گرهای زبان‌محور هنوز اضافه نشده‌اند.
- استخراج PDF مبتنی بر متن است؛ PDF اسکن‌شده به OCR نیاز دارد.
- Reranking وابسته به کیفیت و پوشش کاندیداهای مرحله اول است.
- نتایج بنچمارک کوچک تضمین‌کننده عملکرد عمومی یا آمادگی Production نیستند.

## مجوز

MIT
