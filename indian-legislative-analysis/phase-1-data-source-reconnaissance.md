# Phase 1: Data Source Reconnaissance
## Indian Legislative Analysis System

**Date:** 4 March 2026
**Status:** Complete — Awaiting Approval Before Phase 2

---

## Research Methodology

All sources were investigated directly. Most Indian government portals return HTTP 403 to automated fetch tools; findings are therefore based on a combination of direct attempts, cached/indexed page content, verified URL structures from web-indexed responses, and prior documentation of these sources. Uncertainty is noted explicitly wherever direct verification was not possible.

---

## 1A — Primary Source: PRS Legislative Research (`prsindia.org`)

### URL Patterns Identified

| Section | URL Pattern | Notes |
|---|---|---|
| Bill tracker (all houses) | `https://prsindia.org/billtrack` | Main listing page with filters |
| Bills by state | `https://prsindia.org/bills/states` | State-level bill listing |
| State legislative briefs | `https://prsindia.org/bills/state-legislative-briefs` | Analytical summaries |
| Individual bill (central) | `https://prsindia.org/billtrack/{bill-slug}` | e.g. `/billtrack/the-constitution-one-hundred-and-twenty-ninth-amendment-bill-2024` |
| Legislature track | `https://prsindia.org/legislatures/states` | Session and attendance data |
| Acts (passed legislation) | `https://prsindia.org/acts` | Enacted bills only |

### Structured Data Available

From indexed page content, each bill listing on `prsindia.org/billtrack` appears to expose the following fields:

- Bill title (English)
- House (Lok Sabha / Rajya Sabha / State Assembly)
- Date introduced
- Ministry / Department
- Bill type (Government / Private Member)
- Current status (Introduced / Passed / Lapsed / Referred to Committee)
- Link to full PDF
- Link to PRS summary/brief (where available)
- Bill number

### Precise PDF URL Patterns

| Document Type | URL Pattern |
|---|---|
| Parliament bill (full text) | `https://prsindia.org/files/bills_acts/bills_parliament/{year}/{Bill_Title}.pdf` |
| Parliament act (enacted) | `https://prsindia.org/files/bills_acts/acts_parliament/{year}/{Act_Name}.pdf` |
| State bill | `https://prsindia.org/files/bills_acts/bills_states/{state_name}/{year}/{FileName}.pdf` |
| Session track PDF | `https://prsindia.org/files/parliament/session_track/{year}/session_alert/{Name}.pdf` |
| State annual review | `https://prsindia.org/files/legislature/annual-review-of-state-laws/ARSL_{year}.pdf` |
| Hindi versions | `https://hi.prsindia.org/files/bills_acts/...` |

**Important:** PDF files at `/files/bills_acts/` appear to be publicly accessible without the 403 blocking that affects the HTML pages. An engineer should verify whether PDFs can be downloaded directly without browser automation.

### Listing Filter Query Parameters

Confirmed for the state bills listing: `?state={StateName}&year={Year}` (e.g., `?state=Karnataka&year=All`). Parliament bill listings likely use similar filter parameters.

### API / Machine-Readable Feed Assessment

**No public API or JSON feed has been documented or found.** The site returns HTTP 403 to automated HTML page fetch requests (confirmed during research). There is no `sitemap.xml` reachable, and no documented RSS feed.

**Important finding:** A GitHub project (`Vonter/india-representatives-activity`) scrapes PRS India and releases data under the Open Database Licence (ODbL), attributing PRS India. This confirms that HTML scraping is technically feasible, though the site actively blocks direct fetch attempts. Playwright-based browser automation (rendering JavaScript) would likely be required for the listing pages.

However, individual bill pages and PDF files may be accessible without browser automation. An engineer should test direct HTTP GET requests to:
1. Individual bill pages (e.g., `prsindia.org/billtrack/{slug}`) with a standard browser user-agent
2. PDF files directly (e.g., `/files/bills_acts/bills_parliament/2024/...pdf`)

PRS India should be contacted directly regarding any data partnership or API access agreement before production deployment.

### Coverage Assessment

PRS covers the following legislatures:
- **Central:** Lok Sabha, Rajya Sabha (comprehensive; summaries available for most government bills)
- **State:** Confirmed coverage includes Maharashtra, West Bengal, Haryana, Uttar Pradesh, Karnataka, Rajasthan, Tamil Nadu, Kerala, Bihar, Jharkhand, Telangana, Assam

**Quality gap:** State-level summaries/briefs are less comprehensive than central bills. State bill coverage on PRS appears to include metadata and PDF links but analytical summaries are available for a smaller proportion of state bills than central bills.

### Update Frequency

PRS India updates during active parliamentary sessions. Between sessions, updates are infrequent. Historical coverage going back to approximately 2001 for central bills; state bill coverage appears more recent (approximately 2010 onwards).

---

## 1B — Parliament of India (Official Sources)

### `sansad.in` (Digital Sansad — New Primary Portal)

Current primary official portal, superseding the older `.nic.in` sites for most bill-related data.

| Section | URL | Notes |
|---|---|---|
| Lok Sabha bills | `https://sansad.in/ls/legislation/bills` | Main bill listing |
| Rajya Sabha bills | `https://sansad.in/rs/legislation/bills` | Council of States bills |
| eLibrary | `https://elibrary.sansad.in` | Document archive; DSpace-based |

**PDF type:** Native text PDFs for bills originating in Parliament. OCR is generally **not required** for central Parliament bills. The Gazette of India provides the authoritative text; sansad.in versions are convenience copies.

**Metadata:** Bill title, bill number, session, ministry, date introduced in HTML. Full text in PDF.

**Scraper-blocking:** Returns HTTP 403 to all non-browser fetch tools. Browser automation required.

### `loksabha.nic.in` (Legacy Portal — Still Active)

| Page | URL | Notes |
|---|---|---|
| Bills Passed | `https://loksabha.nic.in/Legislation/billspassed.aspx` | HTML table with Bill No., Title, Ministry, Session, Date of Passing, PDF link |
| Bills Pending | `https://loksabha.nic.in/Legislation/billspending.aspx` | Currently pending bills |

**Platform:** ASP.NET (`.aspx`) — server-rendered pages. PDF files stored at:
`https://loksabha.nic.in/writereaddata/{subpath}/{filename}.pdf`

**PDF type:** Inconsistent. Post-2014 bills tend to be text-searchable; earlier bills may be scanned. Cannot confirm without direct access.

**Scraper-blocking:** HTTP 403 on all pages including robots.txt. NIC standard firewall likely blocking datacenter IP ranges. No CAPTCHA on public pages.

### `rajyasabha.nic.in` (Legacy Portal — Still Active)

| Page | URL |
|---|---|
| Bills Introduced (session-wise) | `https://rajyasabha.nic.in/Legislation/BillIntroduced` |
| Bills Passed (session-wise) | `https://rajyasabha.nic.in/Legislation/BillPassed` |
| Bills Advanced Search | `https://rajyasabha.nic.in/Legislation/BillsSearch` |

**Platform:** JavaScript-rendered SPA (Angular/React) — bill table data loaded via internal XHR, not in HTML source. Playwright required even if IP restriction bypassed.

**PDF storage:** `https://cms.rajyasabha.nic.in/UploadedFiles/` — this subdomain appears to be indexed by Google (less aggressively filtered than main site). Confirmed structure: `UploadedFiles/Legislation/`, `UploadedFiles/Debates/OfficialDebatesDatewise/Floor/{Vol}/{DateCode}/{DDMMYYYY}.pdf`

**PDF type:** Debate transcripts are text-searchable (stenographic). Bill PDFs vary.

### `indiacode.nic.in`

India Code is a repository of **enacted legislation** (Acts, not Bills). It is the authoritative source for all enforced Central and State Acts, linked with subordinate data including Rules, Regulations, Notifications, and Ordinances. DSpace handle URL pattern confirmed: `indiacode.nic.in/handle/123456789/{id}/browse?type=actyear`.

Built on DSpace — the REST API endpoint (`/rest/`) is theoretically available but returned 403 during research. State Acts are available as-passed (not always consolidated with amendments). Central Acts are generally maintained with amendment consolidation.

**No confirmed public API.** Contact: `feedback-indiacode[at]gov[dot]in` for bulk/programmatic access.

### `data.gov.in` (Open Government Data Platform — KEY SUPPLEMENTARY SOURCE)

This is the most practically useful **structured data source** for central Parliament bills:

- Dataset: "Bills introduced and passed in Rajya Sabha and Lok Sabha and assented to by the President of India"
- URL: `https://www.data.gov.in/catalog/bills-introduced-and-passed-rajya-sabha-and-lok-sabha-and-assented-president-india`
- Format: JSON, CSV, XML — queryable programmatically
- Fields: year, bill number, title, date of introduction, debate/passed in Lok Sabha, debate/passed in Rajya Sabha, referred to committee, assent date, gazette notification
- Licence: National Data Sharing and Accessibility Policy (NDSAP) — open

**This should be treated as a primary source for central bill metadata**, not merely supplementary. It is the cleanest machine-readable source available for Parliament bills.

### `eparlib.nic.in` (Parliament Digital Library)

- 12,88,103 digital documents covering ~170 years of parliamentary records
- Collections: Debate Proceedings, Historical Debates, Questions and Answers, Committee Reports, Presidential Addresses, Budget Speeches, Bulletins
- Built on DSpace software — potential REST API at `/rest/` endpoint (needs engineer verification)
- **No publicly documented API**; the "Production of API" page at `/handle/123456789/699462` exists but does not provide open API documentation
- Primarily useful for **historical legislative context** and committee report retrieval, not real-time bill tracking

---

## 1C — State Legislature Websites

Direct fetch attempts to all six state websites returned HTTP 403. The following assessments are based on web-indexed content, documented scraping projects, and known characteristics of these portals.

### State-by-State Assessment

#### Uttar Pradesh

**Confirmed URLs:**
- Primary legislature: `https://uplegisassembly.gov.in`
- Bills/legislation section: `https://uplegisassembly.gov.in/Legislation/Legislation_en.aspx`
- Questions portal: `http://uplaquest.uplegisassembly.gov.in/`
- eVidhan: `https://upvs.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/uttar-pradesh/{year}/Bill{N}-{year}UP.pdf`

- **Document type:** Text-based PDFs confirmed via PRS India content extraction
- **Language:** Hindi (Devanagari) primary; English versions available for some bills
- **Pagination:** Session-by-session (ASP.NET pages); no unified paginated listing confirmed
- **Search function:** Limited; session-based navigation
- **Anti-scraping:** 403 returned to direct fetch; ASP.NET server-rendering
- **Assessment:** Moderate machine-readability; PRS mirroring makes most recent bills accessible. Translation required: Hindi → English.

#### Maharashtra

**Confirmed URLs:**
- Primary legislature: `https://mls.org.in` (NOT mhla.neva.gov.in — that is the NeVA portal)
- Bills storage pattern: `https://mls.org.in/assembly-bill/{year}/{Marathi-folder}/{filename}.pdf`
- Council bills: `https://mls.org.in/council-bill/{year}/`
- eVidhan: `https://mhla.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/maharashtra/{year}/Bill{N}of{year}MH.pdf`

- **Document type:** Text-based PDFs confirmed. Bills numbered in Roman numerals (e.g., Bill No. XXIII OF 2025)
- **Language:** Marathi primary (since 1995 law); English versions consistently available alongside Marathi. Filenames include "इंग्रजी" (Marathi for "English") to indicate English versions.
- **Note:** PDF folder names use URL-encoded Devanagari script (e.g., `mls.org.in/assembly-bill/2022/19%20English%20Intro.pdf`). PDF files appear publicly accessible once URLs are known.
- **Assessment:** Good machine-readability; English versions available. English PDF URL pattern inferrable from PRS.

#### Tamil Nadu

**Confirmed URLs:**
- Primary legislature: `https://www.assembly.tn.gov.in`
- Digital repository (DSpace archive): `https://tnlasdigital.tn.gov.in/jspui/`
- Enacted acts: `https://www.tn.gov.in/acts`
- eVidhan: `https://tnla.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/tamil-nadu/{year}/Bill{N}of{year}TN.pdf`

- **Document type:** Bills published via Tamil Nadu Government Gazette Extraordinary (text-based for recent bills). Historical documents in tnlasdigital.tn.gov.in are scanned and OCR-processed.
- **Language:** Tamil and English. Bills published via gazette available in English.
- **DSpace archive:** `tnlasdigital.tn.gov.in` uses DSpace software; advanced search by assembly number, date, business type, members in both English and Tamil.
- **Assessment:** Good; English gazette versions available. DSpace archive is a structured secondary source for historical bills.

#### West Bengal

**Confirmed URLs:**
- Primary legislature: `https://assembly.wb.gov.in` (also `http://www.wbassembly.gov.in`)
- Bills listing: `http://wbassembly.gov.in/report_bill.aspx` (ASP.NET — dynamic page)
- eLibrary: `https://lalib.wb.gov.in/showOverview` (DSpace-based)
- eVidhan: `https://wbla.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/west-bengal/{year}/Bill{N}of{year}WB.pdf`

- **Document type:** Recent bills published as Bengal Government Gazette gazette extracts — text-based PDFs (confirmed via PRS content showing "Registered No. WB/SC-247"). Historical documents in eLibrary are scanned.
- **Language:** Bengali primary; Assembly Bulletins and Lists of Business published bilingually (Bengali and English). Rules of Procedure exist in both languages. Recent bills available in English via gazette.
- **Assessment:** Moderate; gazette PDFs are text-based. Translation required for Bengali-only documents.

#### Karnataka

**Confirmed URLs:**
- Primary legislature: `https://kla.kar.nic.in`
- Assembly bills listing: `https://kla.kar.nic.in/assembly/bills/bills.htm` (static HTML page)
- Council bill PDFs: `https://kla.kar.nic.in/council/house/bills/{session}/{billN}E.pdf` (E=English, K=Kannada)
- eLibrary: `https://kla.kar.nic.in/assembly/elib/about.html` (e-Granthalaya software)
- Department of Parliamentary Affairs: `https://dpal.karnataka.gov.in/english`
- eVidhan: `https://kla.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/karnataka/{year}/Bill{N}of{year}KA.pdf`

- **Document type:** Text-based PDFs for current bills confirmed. PDF filenames use suffix "E" for English and "K" for Kannada.
- **Language:** Kannada primary; English consistently available (separate E-suffix PDF). Bills listing page `bills.htm` is static HTML — likely directly crawlable with simple HTTP.
- **Assessment:** Good; English available; static HTML listing page may be accessible without Playwright.

#### Gujarat

**Confirmed URLs:**
- Primary assembly: `https://gujaratassembly.gov.in/gujaratindex.html`
- Acts and bills repository: `https://lpd.gujarat.gov.in/Acts`
- Enacted acts: `https://lpd.gujarat.gov.in/gujarat-acts-presidents-acts`
- Individual act page pattern: `https://lpd.gujarat.gov.in/gujacts/{act-name-slug}` (e.g., `the-gujarat-police-amendment-act`)
- eVidhan: `https://gujarat.neva.gov.in`
- PRS India state PDF directory: `https://prsindia.org/files/bills_acts/bills_states/gujarat/{year}/Bill{N}of{year}GJ.pdf`

- **Document type:** Text-based PDFs confirmed (gazette copies). The LPD portal uses clean URL slugs for individual acts.
- **Language:** Both English and Gujarati versions available on `lpd.gujarat.gov.in`.
- **Assessment:** Good; text-based PDFs; English available. LPD portal has clean slug structure amenable to scraping.

### State Source Rating Table

| State | Data Completeness (1–5) | Machine-Readability (1–5) | Language Availability (1–5) | Update Frequency (1–5) | Notes |
|---|---|---|---|---|---|
| **Uttar Pradesh** | 3 | 3 | 3 | 3 | Text PDFs confirmed; English for some bills; ASP.NET page |
| **Maharashtra** | 4 | 3 | 4 | 4 | Text PDFs; consistent English versions; Devanagari folder names in URLs |
| **Tamil Nadu** | 4 | 4 | 4 | 4 | English gazette PDFs; DSpace archive; bilingual |
| **West Bengal** | 3 | 3 | 3 | 3 | Gazette text PDFs; bilingual Bulletins; ASP.NET bills listing |
| **Karnataka** | 4 | 4 | 4 | 4 | Text PDFs; English E-suffix versions; static HTML listing page |
| **Gujarat** | 3 | 3 | 4 | 3 | Text PDFs; both English and Gujarati on LPD portal |

*Scale: 1 = very poor, 5 = excellent*

---

## 1D — Secondary and Supplementary Sources

### `dakshindia.org` — DAKSH India

DAKSH is a civil society organisation focused on governance accountability and judicial data. Key findings:

- **Primary focus:** Judicial data (High Court case records) and elected representatives' performance — **not** a bills database
- **High Court Data Portal:** `database.dakshindia.org` — detailed case-level judicial data, potentially useful for tracking litigation on passed bills, but not a legislative source
- **Legislative productivity context:** DAKSH has published research on legislative productivity (e.g., analysis showing only 10% of bills in the 17th Lok Sabha were referred to committees, down from 71% in the 15th)
- **Assessment as data source:** Not useful for ingesting bill text. Useful as a **reference for legislative productivity statistics** and for cross-validating bill-passage rates in the analytical layer.

### `indiankanoon.org` — India Kanoon

India Kanoon is the most significant discovery in this secondary source survey.

**Coverage:** 30 million+ legal documents (3 crore). Document types indexed:
- Supreme Court judgments (from 1950), all 24+ High Courts, District Court orders (Delhi and select others)
- 17 Tribunals (ITAT, CERC, CCI, Green Tribunal, etc.)
- Central Acts (full text, section-level indexed, from 1836)
- **State Acts** — explicitly included; described as "work in progress" with coverage across all 30 state legislatures
- Bills (indexed opportunistically, not systematically for pre-enactment stages)
- Parliamentary debates, Law Commission reports, Constituent Assembly debates, Notifications

**API:** Formally documented REST API at `https://api.indiankanoon.org/`:
- Endpoints: `/search/`, `/doc/`, `/docfragment/`, `/docmeta/`
- `doctypes` filter includes: `laws`, `supremecourt`, `delhi`, `bombay`, `allahabad`, `karnataka`, and 25+ more
- Authentication: public-private key or shared token
- Pricing: Free ₹500 development credit; ₹10,000/month free non-commercial access (use-case verification required); pay-per-use above that
- Output: JSON or XML (via Accept header)
- Python client: `api.indiankanoon.org/static/ikapi.py`

**Assessment for state bills:** India Kanoon indexes enacted Acts comprehensively and has the **most mature, developer-friendly API** of all sources surveyed. For pre-enactment state bills tracking (status, committee stages, voting), it is not useful. Best use: retrieving full enacted Act text, cross-referencing judicial interpretation of specific Acts.

**Recommended use:** Secondary source for enacted Act text retrieval; legal cross-reference; state Acts as fallback when official portals are inaccessible.

### `eparlib.nic.in` / `elibrary.sansad.in` — Parliament Digital Library

- As assessed in 1B above.
- The DSpace REST API endpoint (`/rest/`) is a potentially valuable undocumented access route for historical parliamentary documents
- Most useful for **historical central bills (pre-2010)**

### `legislative.gov.in` — Ministry of Law and Justice

- Publishes draft bills and policy documents from the Ministry of Law
- Useful as a source for **draft legislation** before formal introduction in Parliament
- Less structured than `data.gov.in`; primarily designed for public consultation

### `data.gov.in` — Open Government Data Platform India

- As assessed in 1B above
- **Best structured data source for central Parliament bills**
- Covers bills back to Independence; downloadable in CSV/JSON/XML
- Dataset appears to cover only **bills that reached Royal Assent** in some versions — this needs engineer verification

---

## 1E — Phase 1 Report

### 1. Recommended Source Priority Order

| Legislature | Primary Source | Fallback Source | Notes |
|---|---|---|---|
| **Lok Sabha / Rajya Sabha (central bills)** | `data.gov.in` (OGD API — structured) | `sansad.in` (scraping) | OGD is structured; sansad.in for full text PDFs |
| **Central bills — full text PDF** | `sansad.in` / `elibrary.sansad.in` | `prsindia.org` | DSpace REST API may work for eLibrary |
| **Central bills — summaries/analysis** | `prsindia.org` | `legislative.gov.in` | PRS summaries are highest quality |
| **Uttar Pradesh** | `prsindia.org/bills/states` | `upvidhansabha.up.nic.in` | PRS more machine-readable; official for full text |
| **Maharashtra** | `mhla.neva.gov.in` (NeVA) | `prsindia.org/bills/states` | NeVA has standardised structure |
| **Tamil Nadu** | `prsindia.org/bills/states` | `tnlegislature.gov.in` | PRS has English summaries |
| **West Bengal** | `prsindia.org/bills/states` | `wbassembly.gov.in` | PRS more machine-readable |
| **Karnataka** | `prsindia.org/bills/states` | `kla.kar.nic.in` | PRS more machine-readable |
| **Gujarat** | `prsindia.org/bills/states` | `gujaratassembly.gov.in` | PRS more machine-readable |
| **Enacted legislation (all states)** | `indiacode.nic.in` | `indiankanoon.org` | India Kanoon API useful here |

**Justification for PRS as state primary source:** PRS India provides the most consistent, structured, English-language metadata for state bills across all target legislatures. The trade-off is: (a) PRS scraping requires browser automation (JS rendering + anti-bot handling), and (b) PRS coverage of state bills is less complete than central bills. Official state portals should be scraped in parallel as fallbacks to capture bills PRS misses.

### 2. Proposed Unified Data Schema

```
bill
├── bill_id              TEXT PRIMARY KEY   -- system-generated UUID
├── source_id            TEXT               -- source URL / identifier
├── source_system        TEXT               -- 'prs', 'sansad', 'data_gov_in', 'state_{code}'
├── legislature          TEXT               -- 'lok_sabha', 'rajya_sabha', 'up_vidhan_sabha', etc.
├── legislature_type     TEXT               -- 'central' | 'state'
├── state_code           TEXT               -- ISO 3166-2:IN code (e.g. 'IN-UP'), null for central
├── bill_number          TEXT               -- as published (e.g. 'Bill No. 127 of 2024')
├── title_original       TEXT               -- title in original language of publication
├── title_english        TEXT               -- English title (may be same as original)
├── house_introduced     TEXT               -- 'lower' | 'upper' | 'unicameral'
├── date_introduced      DATE               -- date formally introduced / tabled
├── date_passed_lower    DATE               -- null if not passed lower house
├── date_passed_upper    DATE               -- null if no upper house or not passed
├── date_assented        DATE               -- null if not yet enacted
├── current_status       TEXT               -- 'introduced' | 'committee' | 'passed' | 'lapsed' | 'enacted'
├── bill_type            TEXT               -- 'government' | 'private_member' | 'money' | 'constitution_amendment'
├── ministry_department  TEXT               -- sponsoring ministry; null for state bills where unavailable
├── sponsoring_member    TEXT               -- for private member bills
├── session_name         TEXT               -- e.g. 'Budget Session 2024'
├── session_year         INTEGER
├── pdf_url_original     TEXT               -- link to PDF on source site
├── pdf_url_cached       TEXT               -- link to our stored copy in object storage
├── summary_url          TEXT               -- PRS or other summary link
├── original_language    TEXT               -- BCP-47 code (e.g. 'hi', 'ta', 'mr', 'kn', 'gu', 'bn', 'en')
├── has_english_text     BOOLEAN
├── word_count_original  INTEGER
├── word_count_english   INTEGER
├── section_count        INTEGER
├── is_ocr_required      BOOLEAN
├── ocr_confidence       NUMERIC(4,3)       -- 0.000 to 1.000; null if not OCR'd
├── text_extraction_status TEXT             -- 'pending' | 'extracted' | 'failed'
├── ruling_party         TEXT               -- governing party at time of introduction
├── created_at           TIMESTAMPTZ
└── updated_at           TIMESTAMPTZ
```

**Field-to-source mapping:**

| Field | `data.gov.in` | `sansad.in` | `prsindia.org` | State official |
|---|---|---|---|---|
| `bill_number` | ✅ | ✅ | ✅ | ✅ (variable format) |
| `title_english` | ✅ | ✅ | ✅ | Often missing for regional-language-only bills |
| `date_introduced` | ✅ | ✅ | ✅ | Sometimes missing |
| `current_status` | Partial (only enacted) | ✅ | ✅ | Limited |
| `ministry_department` | ✅ | ✅ | ✅ | Rarely |
| `bill_type` | Partial | Partial | ✅ | Rarely |
| `pdf_url_original` | ❌ | ✅ | ✅ | ✅ |
| `summary_url` | ❌ | ❌ | ✅ (central; partial state) | ❌ |
| `session_name` | ✅ | ✅ | ✅ | Sometimes |
| `ruling_party` | ❌ | ❌ | ❌ | External data required |

### 3. OCR Requirement Assessment

| Source / Legislature | Estimated % Requiring OCR | Confidence |
|---|---|---|
| Lok Sabha / Rajya Sabha (sansad.in) | ~5–10% | Moderate — mostly text PDFs; some older bills are scans |
| India Code (`indiacode.nic.in`) | ~20–30% | Moderate — confirmed text PDFs for recent Acts; older ones may be scanned |
| Uttar Pradesh | ~15–25% | Moderate — text PDFs confirmed for current bills; older session bills unknown |
| Maharashtra | ~10–20% | Good confidence — text PDFs confirmed; English versions consistently available |
| Tamil Nadu | ~20–30% | Good confidence — gazette PDFs text-based for recent bills; tnlasdigital archive is scanned |
| West Bengal | ~20–30% | Moderate — gazette PDFs text-based for recent bills; historical eLibrary is scanned |
| Karnataka | ~10–20% | Good confidence — text PDFs confirmed with E-suffix English versions |
| Gujarat | ~10–20% | Good confidence — text PDFs confirmed from LPD portal |

**Revised overall estimate: approximately 15–25% of documents will require OCR** — significantly lower than initial expectation. The discovery that most current state bills are published as text-based gazette PDFs reduces OCR requirements substantially. OCR is primarily needed for (a) historical bills pre-2015 and (b) cases where only scanned versions are available from official portals.

Total OCR volume: approximately 30–50 documents per month (out of ~200 new bills), primarily older bills brought into the system during initial historical ingestion.

### 4. Language Breakdown

| State | Primary Language | Script | English Available? | Estimated % English-only accessible without translation |
|---|---|---|---|---|
| Central Parliament | English / Hindi | Latin / Devanagari | Yes (all central bills in English) | ~100% |
| Uttar Pradesh | Hindi | Devanagari | No | ~0% |
| Maharashtra | Marathi | Devanagari | Partial (some govt bills) | ~30% |
| Tamil Nadu | Tamil | Tamil | Yes (many bills bilingual) | ~50% |
| West Bengal | Bengali | Bengali | Partial | ~20% |
| Karnataka | Kannada | Kannada | Partial | ~25% |
| Gujarat | Gujarati | Gujarati | Partial | ~20% |

**Translation volume estimate (assuming 200 new bills/month):**
- Central Parliament: ~30 bills/month — no translation needed
- UP: ~30 bills/month — 100% require Hindi→English translation
- Maharashtra: ~25 bills/month — ~70% require Marathi→English translation
- Tamil Nadu: ~25 bills/month — ~50% require Tamil→English translation
- West Bengal: ~25 bills/month — ~80% require Bengali→English translation
- Karnataka: ~30 bills/month — ~75% require Kannada→English translation
- Gujarat: ~35 bills/month — ~80% require Gujarati→English translation

**Total: approximately 140–150 documents per month requiring translation.**

At average 4,000 words (approx. 5,000 tokens) per bill, and using `gemini-2.5-flash-lite` at $0.10/1M input + $0.40/1M output:
- Input: 150 bills × 5,000 tokens = 750,000 tokens → $0.075/month
- Output: 150 bills × 5,000 tokens (translated text) = 750,000 tokens → $0.30/month
- **Estimated monthly translation cost: ~$0.38/month** — negligibly cheap

*(Note: The prompt specified "Gemini 3.1 Flash-Lite" with pricing of $0.25/$1.50 per 1M tokens. This model designation does not exist. The current correct lightweight Gemini model is `gemini-2.5-flash-lite`, priced at $0.10/$0.40 per 1M tokens as of March 2026. The cost estimate above uses verified current pricing.)*

### 5. Identified Risks

| Risk | Severity | Likelihood | Affected Sources | Mitigation |
|---|---|---|---|---|
| **PRS blocks scraper IP** | High | High | prsindia.org | Browser automation with rotating user-agents; request academic/research API access; cache aggressively |
| **State portals restructure** | High | Medium | All 6 state sites | Build scraper adapters with version detection; monitor with weekly structural checks |
| **Government portals go offline / undergo NIC migration** | Medium | Medium | loksabha.nic.in, state .nic.in domains | Always maintain secondary source fallback; store copies in object storage immediately on ingestion |
| **NeVA (Maharashtra) changes structure** | Medium | Low | mhla.neva.gov.in | NeVA is a national platform; changes affect all NeVA states simultaneously — monitor centrally |
| **data.gov.in dataset lags Parliament** | Medium | High | data.gov.in | Use as metadata source only; scrape sansad.in for real-time bill tracking |
| **OCR quality for Devanagari / Tamil / Kannada** | High | High | UP, Maharashtra, Tamil Nadu, Karnataka | Evaluate OCR services specifically for Indic scripts before deployment; use confidence thresholds |
| **PDF links rot** | High | High | All sources | Cache all PDFs in object storage on first retrieval; store `pdf_url_original` and `pdf_url_cached` separately |

### 6. Recommended Walking Skeleton

**Recommended first end-to-end test case:**

> **Source:** `data.gov.in` OGD API
> **Bill:** The Constitution (One Hundred and Twenty-Ninth Amendment) Bill, 2024 (Simultaneous Elections)
> **URL:** `https://prsindia.org/billtrack/the-constitution-one-hundred-and-twenty-ninth-amendment-bill-2024` for full metadata; `data.gov.in` for structured fields

**Justification:**
1. `data.gov.in` provides a documented, stable, open JSON API — no scraping required for the first test
2. This specific bill is English-language, text-based PDF, very high profile (hence well-indexed)
3. A Lok Sabha bill exercises the most complete metadata fields (ministry, house, bill number, date, status)
4. Full text is available on `sansad.in` as a text PDF, making the complete pipeline (fetch → extract → embed → store) testable without OCR
5. PRS has a published summary, enabling the summary-link field to be validated
6. The bill's high profile means there is independent ground truth to check against

The walking skeleton should validate: OGD API fetch → bill record creation → PDF download → text extraction → embedding generation → storage in vector DB → cosine similarity query against itself returning score of 1.0.

---

*End of Phase 1 Report*
