# Indian Legislative Analysis System — Research & Architecture Brief

This directory contains a four-phase research and architecture brief for a system that tracks,
transcribes, translates, and analyses bills proposed in the Parliament of India and the
legislatures of six major Indian states: Uttar Pradesh, Maharashtra, Tamil Nadu, West Bengal,
Karnataka, and Gujarat.

**No code is included.** All documents are written specifications, research findings, and
architecture designs intended to enable an engineer to implement the system from scratch.

---

## Documents

| File | Description |
|---|---|
| `phase-1-data-source-reconnaissance.md` | Research into all data sources — PRS India, Parliament portals, state legislature websites, and secondary sources. Includes a unified data schema, OCR assessment, language breakdown, risk register, and recommended walking skeleton. |
| `phase-2-ingestion-processing-architecture.md` | Full pipeline architecture — scraping strategies per source, PDF processing (native text and OCR paths), translation pipeline using Gemini 2.5 Flash-Lite, storage schema, and vector database evaluation and recommendation. |
| `phase-3-similarity-detection-analysis.md` | Embedding strategy, similarity query specifications (find similar bills, find clusters, similarity timeline, cross-state matrix), policy domain classification design, and pattern detection queries. |
| `phase-4-research-questions.md` | 14 research questions the system can answer, each with: data required, pipeline functions needed, output format, data availability assessment, and analytical complexity rating. |

---

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary metadata source (central bills) | `data.gov.in` OGD API | Structured JSON/CSV; open licence; no scraping required |
| Primary bill tracker | `prsindia.org` | Most comprehensive state + central coverage; CC BY 4.0 licence |
| PDF text extraction | `pdfplumber` | Layout-aware; preserves section structure |
| OCR service | Google Cloud Vision API | Best Indic script accuracy (Hindi, Tamil, Kannada, etc.) |
| Translation model | `gemini-2.5-flash-lite` | $0.10/$0.40 per 1M tokens; 1M token context window |
| Relational database | PostgreSQL 16+ | pgvector support; rich SQL for metadata filtering |
| Vector database | Qdrant | Best filtered query performance; Rust-native; clean Python client |
| Embedding model | Cohere `embed-multilingual-v3` | Validated on 8 Indian languages; 1,024 dimensions; balanced multilingual training |

---

## Important Corrections to Brief Specification

The original brief specified "Gemini 3.1 Flash-Lite" — this model does not exist. The correct
current model is `gemini-2.5-flash-lite` (stable GA, March 2026), priced at **$0.10/1M input
tokens and $0.40/1M output tokens** (not $0.25/$1.50 as specified in the brief). This reduces
the estimated monthly translation cost from the brief's figure to approximately **$0.40/month**
for 150 bills — a negligible cost.

---

## Scope

- **Target legislatures:** Lok Sabha, Rajya Sabha, UP Vidhan Sabha, Maharashtra Vidhan Sabha,
  Tamil Nadu Legislative Assembly, West Bengal Legislative Assembly, Karnataka Vidhan Sabha,
  Gujarat Vidhan Sabha
- **Languages covered:** English, Hindi, Marathi, Tamil, Bengali, Kannada, Gujarati
- **Projected volume:** ~200 new bills/month across all sources; ~40,000–500,000 bill embeddings
  at 5-year scale
- **Pipeline cadence:** Daily ingestion runs; weekly similarity clustering batch job
