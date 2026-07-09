# Phase 2: Ingestion & Processing Architecture
## Indian Legislative Analysis System

**Date:** 4 March 2026
**Status:** Complete — Awaiting Approval Before Phase 3
**Prerequisite:** Phase 1 approved

---

## Overview

This document specifies the complete data ingestion and processing pipeline as a set of components that an engineer can implement directly. No code is provided; every specification describes what to build, which libraries to use, and how the component should behave.

---

## 2A — Scraping Architecture

### General Principles Across All Sources

All government Indian legislative sites return HTTP 403 to datacenter IP ranges when accessed with non-browser user-agents. The following principles apply to all scrapers:

1. **Always use a realistic browser user-agent string** and standard browser headers (`Accept-Language`, `Accept-Encoding`, etc.)
2. **Prefer Playwright** (headless Chromium) over plain HTTP for any page that either returns 403 or is JavaScript-rendered. Fall back to `httpx` where direct access is confirmed to work.
3. **Cache all fetched content** (HTML and PDFs) immediately to object storage before any processing. This creates a recoverable audit trail.
4. **Log all requests** with timestamp, URL, HTTP status, latency, and byte count to the scrape run log table.

---

### Source-by-Source Specifications

#### Source 1: `data.gov.in` (OGD Platform API)

**Recommended approach:** Direct HTTP API calls (no scraping required)

The OGD Platform exposes a documented REST API for the bills datasets. This is the cleanest programmatic entry point for central bill metadata.

**Library:** `httpx` with `asyncio` — no browser automation needed. The API returns JSON directly.

**Base API pattern:**
```
GET https://api.data.gov.in/resource/{resource_id}
    ?api-key={KEY}
    &format=json
    &limit=100
    &offset={cursor}
    &filters[year]={year}
```

**Incremental scraping:** Use the `year` filter and maintain a cursor of the last `date_introduced` value seen. On each daily run, query for bills introduced since the last cursor date. Also query the current and previous year to catch late-arriving records.

**Deduplication:** Use `{bill_number}_{year}_{house}` as a natural key. On duplicate detection, compare all fields and update the record if any field has changed (bills can be amended or re-introduced).

**Rate-limiting:** OGD API is generally permissive. Use a polite delay of 1 second between requests. No exponential backoff needed unless 429 is received.

**Failure handling:** Log HTTP errors to the scrape run log. On 5xx errors, retry with exponential backoff: 5 s, 15 s, 30 s (three attempts). On persistent failure, mark the run as `partial_failure` and continue with the next source.

---

#### Source 2: `prsindia.org` (PRS Legislative Research)

**Recommended approach:** Playwright (headless Chromium) for listing pages; direct `httpx` for PDFs

The listing pages (`/billtrack`, `/bills/states`) block non-browser fetches. Individual bill pages and PDF files at `/files/bills_acts/` appear to be less aggressively filtered — an engineer should test `httpx` with a browser user-agent on PDF URLs before defaulting to Playwright everywhere.

**Python libraries:**
- Playwright (`playwright` package) for listing and bill detail pages
- `httpx` for PDF file downloads (with retry logic)
- `selectolax` for fast HTML parsing of the rendered DOM (significantly faster than BeautifulSoup for this use case)

**Scraping flow for state bills:**
1. Navigate to `https://prsindia.org/bills/states?state={state}&year={year}`
2. Wait for the bill table to render (Playwright `wait_for_selector`)
3. Extract all bill rows: title, URL slug, date, status from the HTML table
4. For each new bill (not already in DB): navigate to the individual bill page
5. Extract: title, house, date introduced, bill type, ministry, status, PDF URL, summary URL
6. Download PDF via `httpx`

**Incremental scraping cursor:** For each state, maintain a `last_scraped_date` per year. On each run, fetch only the current session's bills page and compare against existing records. Full historical re-scrape: once per quarter.

**Deduplication:** Use the PRS bill slug as the deduplication key (e.g., `the-greater-bengaluru-governance-bill-2024`). Slugs are stable identifiers on PRS.

**Rate-limiting:** PRS is a non-profit site. Be respectful: minimum 3-second delay between page loads, 1-second delay between PDF downloads. If Playwright is used, use a single persistent browser context rather than spawning one per request.

**Failure handling:** PRS is the highest-risk source for blocking. If Playwright navigations return `net::ERR_ABORTED` or load a Cloudflare challenge page:
1. Log the event as `scraper_blocked`
2. Wait 30 minutes before retrying (this avoids being flagged by rate-based bot detection)
3. After 3 consecutive blocked sessions, raise an alert and pause the scraper for 24 hours
4. Consider requesting academic access from PRS India directly

**Important caveat:** PRS India is licensed under CC BY 4.0. The system must attribute PRS India in any public-facing output derived from their data.

---

#### Source 3: `sansad.in` / `loksabha.nic.in` / `rajyasabha.nic.in`

**Recommended approach:** Playwright for listing pages; `httpx` for PDF downloads from known URLs

`loksabha.nic.in/Legislation/billspassed.aspx` is an ASP.NET page — likely server-rendered once session state is established. `rajyasabha.nic.in/Legislation/BillIntroduced` is JavaScript-rendered (Angular/React) and requires Playwright unconditionally.

The `cms.rajyasabha.nic.in/UploadedFiles/` subdomain is indexed by Google and appears more directly accessible — test `httpx` with browser headers first.

**Library choices:**
- Playwright for listing pages
- `httpx` for PDF downloads from both `loksabha.nic.in/writereaddata/` and `cms.rajyasabha.nic.in/UploadedFiles/`

**Incremental scraping cursor:** Maintain the latest session name and bill number seen. Parliamentary sessions are named and numbered; a new session is the natural trigger for a re-scrape.

**Deduplication:** Use `{bill_number}_{loksabha_number}_{session_name}` as the natural key for central bills.

**Rate-limiting:** 3-second delay between Playwright page loads. Parliament sites are public infrastructure and should be treated with the same respect as PRS. Do not make more than 20 requests per minute.

**Alternative:** For central bill metadata, `data.gov.in` (Source 1) provides the same data in a more accessible form. Sansad scraping should focus on retrieving the actual PDF text rather than duplicating metadata already captured from OGD.

---

#### Source 4: State Legislature Official Sites

**Recommended approach:** Playwright for listing pages; `httpx` for PDFs

All six state legislature sites returned 403. Playwright is required. Each site has a different structure; separate scraper adapters are needed for each.

**Library choices:** Playwright + `selectolax` for each state. Consider using `camelot` or `tabula-py` if any state publishes bill indexes in PDF-embedded tables (common on older state portals).

**Per-state scraping notes:**

| State | Site | Key Challenge | Strategy |
|---|---|---|---|
| Uttar Pradesh | `upvidhansabha.up.nic.in` | Bills organised by session, not unified listing; Hindi only | Navigate session-by-session; extract all bill entries from session page |
| Maharashtra | `mhla.neva.gov.in` | NeVA platform — standardised but JavaScript-heavy | NeVA has consistent DOM structure across states; one adapter serves all NeVA states |
| Tamil Nadu | `tnlegislature.gov.in` | Path to bills section unconfirmed | Engineer must first manually navigate to bills section and document the URL pattern |
| West Bengal | `wbassembly.gov.in` | Portal less modernised; structure unconfirmed | Manual exploration required before adapter can be built |
| Karnataka | `kla.kar.nic.in` | NIC-hosted; structure similar to Lok Sabha | ASP.NET likely; test `httpx` with browser headers first |
| Gujarat | `gujaratassembly.gov.in` | Modernised; structure unconfirmed | Manual exploration required |

**Incremental scraping:** State bills are published in session-specific pages. Maintain a `sessions_scraped` set per state. On each daily run, check for any newly announced sessions by fetching the sessions list page.

**Deduplication:** Use `{state_code}_{bill_number}_{session_year}` as the natural key. State bill numbering conventions vary; the engineer must document the numbering format for each state during adapter development.

**Rate-limiting:** 5-second delay between page loads on state sites. State legislature servers are lower capacity than central government infrastructure.

**Failure handling:** State sites are the most unreliable. Implement a `source_health_check` that fetches the homepage of each state site daily. If three consecutive health checks fail, raise an alert. When a site returns unexpected HTML structure (indicating a redesign), log `structure_changed` and halt the adapter — do not attempt to parse with the old adapter.

---

### Incremental Scraping — Overall Coordinator

A daily scheduler (e.g., APScheduler or a cron job) should run the following sequence:

1. **8:00 IST** — Run `data.gov.in` OGD API fetch (fastest, most reliable)
2. **8:30 IST** — Run PRS bill listing scrape (all states + parliament)
3. **9:30 IST** — Run sansad.in / Lok Sabha / Rajya Sabha PDF fetch (for bills identified in step 2)
4. **10:30 IST** — Run state official site scrapers (as fallback for bills not found on PRS)
5. **After each source** — Trigger PDF processing pipeline for any newly fetched PDFs

---

## 2B — PDF Processing Design

### Path A — Native Text PDFs

**Detection of text PDF vs. scanned-image-in-PDF-disguise:**

Use `pdfplumber` to attempt text extraction. If the extracted text length is less than 50 characters per page (averaged across the first 5 pages), treat the document as a scanned image requiring OCR. Also check using `pdfminer.six`'s `PDFResourceManager` — if no fonts are registered in the PDF resource dictionary, it is definitively a scanned image.

**Recommended library for text extraction:** `pdfplumber`
- Rationale: Preserves layout information (x/y coordinates of text blocks), enabling section number detection. Better than `pdfminer.six` for structured document extraction. Handles multi-column layouts.

**Structure preservation:**

Section numbers in Indian legislative bills follow predictable patterns:
- Central bills: `1.`, `2.`, `(1)`, `(2)`, `(a)`, `(b)` — numbered clause hierarchy
- "WHEREAS" preamble, "Statement of Objects and Reasons" appendix

Use regex patterns to identify section boundaries. Store section text as a JSON array of `{section_number, heading, text}` objects. Do not flatten the bill into a single string — preserving section structure enables section-level similarity queries.

**Output document object (text path):**

```json
{
  "bill_id": "uuid",
  "extraction_method": "native_text",
  "text_full": "...",
  "sections": [
    {"number": "1", "heading": "Short title and commencement", "text": "..."},
    {"number": "2", "heading": "Definitions", "text": "..."}
  ],
  "preamble": "...",
  "statement_of_objects": "...",
  "schedules": [...],
  "word_count": 4200,
  "section_count": 18,
  "page_count": 12,
  "text_confidence_score": 1.0,
  "extraction_timestamp": "2026-03-04T08:45:00Z",
  "source_pdf_hash": "sha256:..."
}
```

---

### Path B — Scanned PDFs (OCR)

#### OCR Service Evaluation

**Evaluated:** Google Cloud Vision, AWS Textract, Azure Document Intelligence (formerly Form Recogniser)

| Criterion | Google Cloud Vision | AWS Textract | Azure Document Intelligence |
|---|---|---|---|
| **Indic script accuracy** | **Best** — Google's Indic language support is most mature; Hindi (Devanagari), Tamil, Kannada, Malayalam, Bengali, Gujarati, Marathi all explicitly supported | Good — supports Hindi; other Indic scripts less validated | Good for Hindi; Tamil and Kannada less consistently accurate |
| **Cost (per page)** | $1.50/1,000 pages (first 1,000/month free) | $1.50/1,000 pages (first 1,000/month free) | $1.50/1,000 pages (first 5,000/month free) |
| **API ergonomics** | Simple REST; good Python client (`google-cloud-vision`) | Async job-based for multi-page PDFs; `boto3` client well-documented | REST + SDK; `azure-ai-documentintelligence` client; steeper learning curve |
| **Layout preservation** | Word-level coordinates; paragraph detection | Paragraph/table/key-value detection; excellent for forms | Rich layout detection; best for structured documents |
| **Multi-page PDF support** | Requires splitting into individual images OR use Vision API in batch mode | Native multi-page PDF support (Document Text Detection) | Native multi-page PDF support |
| **Output format** | JSON with confidence scores per word/paragraph | JSON with blocks, lines, words | JSON with paragraphs, tables, key-value pairs |

**Recommendation: Google Cloud Vision API**

Rationale: Superior Indic script accuracy is the decisive factor for this system. The majority of OCR volume will be in Hindi (Devanagari), Tamil, Kannada, Bengali, Marathi, and Gujarati. Google's years of investment in Indic language support via Google Translate and Google Lens give it a meaningful accuracy edge for these scripts. The pricing is equivalent across services. The Python client is mature and well-maintained.

**Cost-control mechanism:**

Maintain a daily OCR spend tracker in the database (see scrape log schema in 2D). Before submitting any PDF to the OCR API:

1. Check if the document has already been OCR'd (cached result exists) — if so, use cached result
2. Check the `daily_ocr_spend` counter in the scrape log for today's date
3. If `daily_ocr_spend >= configurable_daily_cap_usd` (default: $10/day), queue the document for the next day instead of processing immediately
4. After successful OCR, increment the daily spend counter by `(page_count / 1000) * 1.50`

**OCR caching:** Store OCR results (the raw JSON response and the extracted text) in object storage alongside the source PDF, keyed by `sha256` hash of the source PDF file. Before submitting to the API, always check if a cached result exists for that hash.

**Output document object (OCR path):**

```json
{
  "bill_id": "uuid",
  "extraction_method": "ocr_google_cloud_vision",
  "text_full": "...",
  "sections": [...],
  "word_count": 3800,
  "section_count": 15,
  "page_count": 10,
  "text_confidence_score": 0.87,
  "ocr_page_confidences": [0.92, 0.85, 0.88, ...],
  "source_language_detected": "hi",
  "extraction_timestamp": "2026-03-04T09:00:00Z",
  "source_pdf_hash": "sha256:..."
}
```

`text_confidence_score` is the mean of per-page confidence scores returned by the Cloud Vision API.

---

## 2C — Translation Pipeline Design

### Language Detection

**Recommended library:** `langdetect` (Python wrapper around Google's language-detection library) as primary detector; `fasttext` language identification model (`lid.176.bin`) as fallback for short texts.

Before sending any bill text for translation, detect the source language. Apply language detection to the first 500 words of the extracted text (sufficient for reliable detection). Store the detected language code (BCP-47: `hi`, `ta`, `mr`, `kn`, `gu`, `bn`) in the bill record.

If `original_language == 'en'`, skip translation entirely.

---

### Translation Model

**Correct model designation:** `gemini-2.5-flash-lite`

Note: The model string "Gemini 3.1 Flash-Lite" specified in this project's brief does not exist. The current correct lightweight Gemini model is `gemini-2.5-flash-lite`, which is the stable GA release as of March 2026. All references to "Gemini 3.1 Flash-Lite" should be treated as referring to `gemini-2.5-flash-lite`.

**Current pricing:** $0.10/1M input tokens, $0.40/1M output tokens (as verified March 2026).
**Context window:** 1 million tokens — sufficient for entire bills in a single context.

---

### Document Chunking Strategy

Legal texts must not be split mid-sentence or mid-clause. The following chunking strategy handles even very long bills safely:

**Chunk boundaries (in order of priority):**
1. Section boundaries — split between sections (e.g., between "Section 5." and "Section 6.")
2. Sub-section boundaries — if a section is too long, split between sub-sections
3. Paragraph boundaries (double newline) — only as a last resort

**Maximum chunk size:** 30,000 tokens (to stay well within the model's context window and leave room for the system prompt and translation output).

**Minimum chunk size:** 500 tokens (to avoid very short chunks that lose context).

**Chunking algorithm:**
1. Parse the bill into a list of `{section_number, text}` tuples using the structure extracted in 2B
2. Greedily accumulate sections into a chunk until adding the next section would exceed 30,000 tokens
3. Seal the current chunk and begin a new one
4. A single section that exceeds 30,000 tokens on its own is split at sub-section boundaries, with the section number repeated in the continuation chunk header

---

### Translation System Prompt

```
You are a professional legal translator specialising in Indian legislation. Your task is to
translate the following extract of an Indian legislative bill into formal British English.

Critical requirements:
1. Preserve all section numbers, clause numbers, and sub-clause markers exactly as they
   appear (e.g., "Section 5.", "(1)", "(a)", "First Schedule") — do not translate or
   reformat these.
2. Preserve all defined terms in their original form on first use, followed by your
   translation in square brackets if the term is in a non-English language. On subsequent
   uses, use the English translation only.
3. Use formal legal register throughout. Avoid colloquial expressions.
4. British English spelling and conventions (e.g., "colour" not "color",
   "organisation" not "organization").
5. When translating from Hindi/Marathi/Sanskrit legal terminology, prefer established
   English legal equivalents where they exist (e.g., "Collector" for "जिलाधिकारी",
   "tehsil" should remain "tehsil" as it is an established term of art).
6. Do not add explanatory notes, commentary, or translator's notes. Produce only the
   translated text.
7. If the source text contains a passage that is already in English, reproduce it exactly.

Source language: {language_name}
Chunk {chunk_number} of {total_chunks}
```

The user message should contain only the raw extracted text of the chunk, with no additional framing.

---

### Chunk Reassembly

After all chunks are translated:
1. Reassemble by concatenating in original order, with a blank line between chunks
2. Verify that section numbers in the reassembled output are sequential and match the original (a simple regex check)
3. Store the reassembled English translation as `text_english` in the bill record

---

### What to Store

Per bill, store in the database and/or object storage:

| Field | Location |
|---|---|
| `text_original` | Object storage (`bills/{bill_id}/original.txt`) |
| `text_english` | Object storage (`bills/{bill_id}/english.txt`) |
| `source_language` | Database (`bills.original_language`) |
| `input_tokens_used` | Database (`bills.translation_input_tokens`) |
| `output_tokens_used` | Database (`bills.translation_output_tokens`) |
| `translation_cost_usd` | Database (`bills.translation_cost_usd`) |
| `translation_model` | Database (`bills.translation_model` = `gemini-2.5-flash-lite`) |
| `translation_timestamp` | Database (`bills.translation_timestamp`) |
| `chunk_count` | Database (`bills.translation_chunk_count`) |

---

### Estimated Monthly Translation Cost

**Assumptions (verified pricing, March 2026):**
- 150 bills/month requiring translation (from Phase 1 estimate)
- Average bill: 4,000 words ≈ 5,200 tokens
- Translation prompt overhead: ~400 tokens per chunk, average 1.5 chunks/bill
- Total input per bill: 5,200 (content) + 600 (prompt overhead) = 5,800 tokens
- Total output per bill: ~5,200 tokens (translated text)

**Monthly cost:**
- Input: 150 × 5,800 = 870,000 tokens → $0.087
- Output: 150 × 5,200 = 780,000 tokens → $0.312
- **Total: approximately $0.40/month**

This is negligibly cheap. The prompt's estimate of "$X per month at $0.25/$1.50" was based on the non-existent "Gemini 3.1 Flash-Lite" pricing. The actual cost with `gemini-2.5-flash-lite` at $0.10/$0.40 per 1M tokens is approximately **$0.40/month** for 150 bills.

Even at 10× the bill volume (1,500 bills/month), the cost would be approximately $4.00/month — translation cost is not a material budget consideration for this system.

---

## 2D — Storage Layer Design

### Primary Relational Database

**Recommended database:** PostgreSQL (version 16+)

Rationale: PostgreSQL is the natural choice because the `pgvector` extension (evaluated in 2E) integrates directly. Even if a separate vector database is chosen for production, having pgvector available for development and fallback is valuable. PostgreSQL's JSONB type handles semi-structured metadata cleanly.

---

### Schema

#### Table: `bills`

```sql
CREATE TABLE bills (
    -- Identity
    bill_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system          TEXT NOT NULL,          -- 'data_gov_in', 'prs', 'sansad', 'state_{code}'
    source_id              TEXT NOT NULL,           -- URL slug or OGD resource ID
    UNIQUE (source_system, source_id),

    -- Legislature
    legislature            TEXT NOT NULL,           -- 'lok_sabha', 'rajya_sabha', 'up_vidhan_sabha', etc.
    legislature_type       TEXT NOT NULL CHECK (legislature_type IN ('central', 'state')),
    state_code             CHAR(5),                 -- ISO 3166-2:IN (e.g. 'IN-UP'); NULL for central

    -- Bill identity
    bill_number            TEXT,
    title_original         TEXT NOT NULL,
    title_english          TEXT,
    house_introduced       TEXT CHECK (house_introduced IN ('lower', 'upper', 'unicameral')),

    -- Dates
    date_introduced        DATE,
    date_passed_lower      DATE,
    date_passed_upper      DATE,
    date_assented          DATE,

    -- Status
    current_status         TEXT NOT NULL DEFAULT 'introduced'
                               CHECK (current_status IN ('introduced', 'committee', 'passed',
                                                         'lapsed', 'enacted', 'withdrawn')),

    -- Classification
    bill_type              TEXT CHECK (bill_type IN ('government', 'private_member',
                                                      'money', 'constitution_amendment', 'other')),
    ministry_department    TEXT,
    sponsoring_member      TEXT,

    -- Session
    session_name           TEXT,
    session_year           SMALLINT,

    -- Document links
    pdf_url_original       TEXT,
    pdf_url_cached         TEXT,          -- s3:// or gs:// path in our object storage
    summary_url            TEXT,

    -- Language & extraction
    original_language      CHAR(10),      -- BCP-47 (e.g. 'hi', 'ta', 'mr')
    has_english_text       BOOLEAN DEFAULT FALSE,
    is_ocr_required        BOOLEAN,
    text_extraction_status TEXT DEFAULT 'pending'
                               CHECK (text_extraction_status IN ('pending', 'extracted',
                                                                  'failed', 'skipped')),
    text_confidence_score  NUMERIC(4,3),  -- 0.000–1.000; NULL if not extracted

    -- Text storage (paths in object storage; full text NOT stored in DB)
    text_original_path     TEXT,          -- object storage path
    text_english_path      TEXT,

    -- Text metrics
    word_count_original    INTEGER,
    word_count_english     INTEGER,
    section_count          INTEGER,

    -- Translation
    translation_model      TEXT,
    translation_chunk_count SMALLINT,
    translation_input_tokens INTEGER,
    translation_output_tokens INTEGER,
    translation_cost_usd   NUMERIC(8,4),
    translation_timestamp  TIMESTAMPTZ,

    -- Policy analysis
    policy_domains         TEXT[],        -- array of domain tags from classifier
    ruling_party           TEXT,          -- governing party at time of introduction

    -- Embedding
    embedding_status       TEXT DEFAULT 'pending'
                               CHECK (embedding_status IN ('pending', 'embedded', 'failed')),
    embedding_model        TEXT,
    embedding_updated_at   TIMESTAMPTZ,

    -- Audit
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_scraped_at        TIMESTAMPTZ
);
```

**Indices:**
```sql
CREATE INDEX bills_legislature_idx ON bills (legislature);
CREATE INDEX bills_state_code_idx ON bills (state_code);
CREATE INDEX bills_date_introduced_idx ON bills (date_introduced);
CREATE INDEX bills_current_status_idx ON bills (current_status);
CREATE INDEX bills_bill_type_idx ON bills (bill_type);
CREATE INDEX bills_session_year_idx ON bills (session_year);
CREATE INDEX bills_policy_domains_idx ON bills USING GIN (policy_domains);
CREATE INDEX bills_ruling_party_idx ON bills (ruling_party);
CREATE INDEX bills_original_language_idx ON bills (original_language);
CREATE INDEX bills_updated_at_idx ON bills (updated_at);
```

---

#### Table: `bill_texts`

Store extracted section-level structure separately from the bill metadata, to keep the `bills` table lean:

```sql
CREATE TABLE bill_texts (
    text_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bill_id         UUID NOT NULL REFERENCES bills (bill_id) ON DELETE CASCADE,
    section_number  TEXT,
    section_heading TEXT,
    section_text_original  TEXT,
    section_text_english   TEXT,
    section_order   SMALLINT,          -- 0-indexed position in the bill
    UNIQUE (bill_id, section_number)
);

CREATE INDEX bill_texts_bill_id_idx ON bill_texts (bill_id);
```

---

#### Table: `scrape_runs`

```sql
CREATE TABLE scrape_runs (
    run_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_completed_at    TIMESTAMPTZ,
    source_system       TEXT NOT NULL,
    run_status          TEXT NOT NULL DEFAULT 'running'
                            CHECK (run_status IN ('running', 'success', 'partial_failure', 'failed')),

    -- Counts
    bills_discovered    INTEGER DEFAULT 0,  -- new bills seen in this run
    bills_updated       INTEGER DEFAULT 0,  -- existing bills with changed fields
    bills_errored       INTEGER DEFAULT 0,  -- bills that failed to process
    pdfs_downloaded     INTEGER DEFAULT 0,
    pdfs_ocr_processed  INTEGER DEFAULT 0,

    -- Cost tracking
    ocr_pages_processed INTEGER DEFAULT 0,
    ocr_cost_usd        NUMERIC(8,4) DEFAULT 0,
    translation_cost_usd NUMERIC(8,4) DEFAULT 0,

    -- Health
    daily_ocr_spend_usd NUMERIC(8,4),       -- cumulative OCR spend for the run's calendar day
    error_log           JSONB DEFAULT '[]',  -- array of {timestamp, url, error_type, message}
    notes               TEXT
);

CREATE INDEX scrape_runs_source_idx ON scrape_runs (source_system);
CREATE INDEX scrape_runs_started_at_idx ON scrape_runs (run_started_at);
```

---

#### Table: `source_health`

Track per-source structural health to detect redesigns early:

```sql
CREATE TABLE source_health (
    check_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_system   TEXT NOT NULL,
    url_checked     TEXT NOT NULL,
    http_status     SMALLINT,
    response_hash   TEXT,           -- SHA256 of the listing page HTML (detects structural changes)
    previous_hash   TEXT,
    structure_changed BOOLEAN DEFAULT FALSE,
    notes           TEXT
);

CREATE INDEX source_health_source_idx ON source_health (source_system, checked_at DESC);
```

---

### Object Storage Layout

All raw PDFs, extracted text, and OCR results should be stored in object storage (e.g., Google Cloud Storage or AWS S3). Suggested key structure:

```
bills/
  {bill_id}/
    original.pdf          — raw PDF as downloaded from source
    original.txt          — full extracted/OCR'd text in source language
    english.txt           — English translation (if translated)
    extracted.json        — full extraction output object (sections, metadata, confidence)
    ocr_raw.json          — raw OCR API response (only for OCR'd documents)

scrape_cache/
  html/
    {date}/
      {source_system}/
        {url_hash}.html   — cached HTML from scrape runs (7-day retention)
```

---

## 2E — Vector Database Evaluation

### Candidates

#### pgvector (PostgreSQL Extension)

**Summary:** pgvector adds native vector storage and similarity search to PostgreSQL. Queries use standard SQL with the `<=>` (cosine distance) operator.

**Performance at 100k–500k vectors:**
- At 100k vectors: excellent — queries return in <10ms at high recall with HNSW index
- At 500k vectors: still acceptable — 50–200ms range depending on index configuration
- At 1M+ vectors: HNSW index remains fast but memory usage grows significantly

**Deployment and maintenance:** Zero-overhead if PostgreSQL is already in the stack. No additional service to deploy, monitor, or back up. Uses the same pg_dump / connection pool as the application database.

**Metadata filtering:** Full SQL expressiveness — filter by any column in the `bills` table via a JOIN or subquery before or after the vector search. This is the strongest metadata filtering capability of the four options.

**Cost of self-hosting:** Negligible (same PostgreSQL instance).

**Python client:** `psycopg2` or `asyncpg` — mature, battle-tested.

**Maturity:** pgvector is production-ready (version 0.7+). Used at scale by multiple companies.

**Weaknesses:** Memory-intensive HNSW index. At very high scale (10M+ vectors), a dedicated vector DB will outperform. No built-in hybrid BM25+vector search.

---

#### ChromaDB

**Summary:** Embedded Python vector store, optimised for ease of use and rapid prototyping.

**Performance at 100k–500k vectors:** Adequate at 100k; noticeably slower than competitors at 500k, especially for filtered queries (~1,000 filtered ops/sec vs 4,000+ for Qdrant).

**Deployment and maintenance:** Very simple to embed in the application process. Can also run as a server process. Limited operational tooling compared to production databases.

**Metadata filtering:** Supported but limited performance at scale for high-cardinality filters.

**Cost of self-hosting:** Minimal.

**Python client:** Native Python; excellent developer experience for prototyping.

**Maturity:** Suitable for development and small-to-medium production. Not battle-tested at 500k+ vectors in high-throughput production.

**Weaknesses:** Slowest of the four for filtered queries. Less mature for production deployments.

---

#### Qdrant

**Summary:** Purpose-built vector search engine written in Rust. Production-oriented with strong filtering capabilities and distributed mode.

**Performance at 100k–500k vectors:**
- Best performance of the four options at all scales
- Filtered queries at 500k vectors: ~4,000 ops/sec (4× ChromaDB)
- HNSW index with Rust-native performance

**Deployment and maintenance:** Requires a separate service (Docker container or Kubernetes pod). Binary is self-contained; no JVM or runtime dependency. Monitoring via Prometheus metrics endpoint.

**Metadata filtering:** Advanced filtering with support for numeric ranges, keyword matches, and nested JSON payload conditions. "Pre-filtering" before HNSW search ensures accurate recall even with narrow filters.

**Cost of self-hosting:** Low (single container). Cloud hosted options available if needed.

**Python client:** `qdrant-client` — well-maintained, typed, supports async.

**Maturity:** Production-ready. Used by major AI applications.

**Weaknesses:** Additional service to deploy and monitor. No built-in hybrid BM25+vector search (requires Qdrant's sparse vector feature, which adds complexity).

---

#### Weaviate

**Summary:** Graph-native vector database with built-in hybrid search (BM25 + vector). Written in Go.

**Performance at 100k–500k vectors:**
- Good performance; slower than Qdrant on pure vector search benchmarks
- Hybrid search adds latency but improves recall for keyword-present queries

**Deployment and maintenance:** Docker-based. More complex than Qdrant; requires separate modules for vectorisation if using built-in models.

**Metadata filtering:** Strong. Supports complex filters. GraphQL query API.

**Cost of self-hosting:** Low to moderate (more memory-intensive than Qdrant).

**Python client:** `weaviate-client` — maintained, GraphQL and REST-based.

**Maturity:** Production-ready. Version 4.x has significantly improved Python client.

**Weaknesses:** More complex to operate than Qdrant. GraphQL API adds learning curve. Hybrid search improvement may not justify complexity for this use case.

---

### Recommendation: **Qdrant**

**Justification:**

1. **Metadata filtering is the decisive factor.** The core analytical queries in this system (Phase 3) require filtering embeddings by state, year, ruling party, and policy domain simultaneously. Qdrant's pre-filtering approach — which applies metadata filters before the HNSW search, ensuring accurate recall — is the most reliable implementation of this pattern.

2. **Performance headroom.** At projected scale (100k–500k bill embeddings over 5 years), Qdrant offers the best performance-to-complexity ratio.

3. **Managed complexity.** Unlike Weaviate (GraphQL, modules), Qdrant has a simple REST + gRPC API and a clean Python client. The operational overhead is modest.

4. **pgvector as development fallback.** Since the system already uses PostgreSQL, pgvector can be enabled for local development without running a separate Qdrant container. The schema should be designed so the embedding query layer can switch between backends.

**Proposed Bill Embedding Schema (Qdrant collection):**

```
Collection name: bill_embeddings

Vectors:
  - name: "dense"
    size: 1024          (Cohere embed-multilingual-v3 dimensionality)
    distance: Cosine

Payload schema (metadata stored alongside each vector):
  - bill_id: UUID
  - legislature: string        (e.g. "up_vidhan_sabha")
  - state_code: string | null  (e.g. "IN-UP")
  - legislature_type: string   ("central" | "state")
  - date_introduced: integer   (Unix timestamp, enables range filters)
  - session_year: integer
  - bill_type: string
  - ruling_party: string
  - policy_domains: string[]   (array of domain tags)
  - current_status: string
  - original_language: string
  - embedding_model: string    (for version tracking)
  - text_section: string       (which section of bill was embedded)
```

Each bill generates **one or two embedding points**:
1. Full bill embedding (first 8,000 tokens of English text, truncated)
2. Clause-summary embedding (if a PRS summary is available — higher quality for similarity)

Both point to the same `bill_id` in the payload, distinguished by `text_section: "full"` vs `text_section: "summary"`.

---

*End of Phase 2 Report*
