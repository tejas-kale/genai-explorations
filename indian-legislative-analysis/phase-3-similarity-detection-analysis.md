# Phase 3: Similarity Detection & Analysis Design
## Indian Legislative Analysis System

**Date:** 4 March 2026
**Status:** Complete — Awaiting Approval Before Phase 4
**Prerequisite:** Phase 2 approved

---

## 3A — Embedding Strategy

### Model Selection

Three candidates were evaluated:

#### OpenAI `text-embedding-3-large`
- **Dimensionality:** 3,072 (can be reduced via Matryoshka truncation)
- **Languages:** 100+ supported; however, training corpus is English-dominant
- **Indian language performance:** Supported but not specifically optimised. No dedicated Indian language validation published by OpenAI.
- **Cost:** $0.13/1M tokens
- **Storage:** 3,072 × 4 bytes = 12.3KB per vector (4× Cohere's storage requirement)
- **Context window:** 8,192 tokens

#### Cohere `embed-multilingual-v3`
- **Dimensionality:** 1,024
- **Languages:** 100+ supported; trained on **balanced multilingual data** (not English-dominant)
- **Indian language performance:** Explicitly validated across Bengali, Kannada, Malayalam, Tamil, Telugu, Hindi, Marathi, and English in an AWS Bedrock RAG demonstration. This is the most thoroughly validated model for Indian language embedding among available options.
- **Cost:** Cohere API pricing (approximately $0.10/1M tokens for embed; exact current pricing should be confirmed at deployment time)
- **Storage:** 1,024 × 4 bytes = 4.1KB per vector
- **Context window:** 512 tokens (maximum embedding input length)
- **Compression:** Supports binary quantisation for 50–75% storage reduction

#### Other Candidates Considered

**`text-embedding-3-small` (OpenAI):** Lower cost but inferior multilingual performance. Not recommended given Indian language requirements.

**`multilingual-e5-large` (Microsoft, open source):** Strong multilingual benchmark performance, but requires self-hosting an inference server (adds operational complexity). Worth revisiting if embedding API costs become material at scale.

**`jina-embeddings-v3` (Jina AI):** Frontier multilingual embedding model with task-specific dimensions. Strong MTEB benchmark performance. API-accessible. Worth evaluating as a future alternative.

### Recommendation: **Cohere `embed-multilingual-v3`**

**Justification:**
1. **Validated Indian language support** is the decisive factor. The system ingests bills in Hindi, Tamil, Marathi, Kannada, Gujarati, and Bengali — all validated on Cohere's model.
2. **Balanced multilingual training** means inter-language similarity scores are meaningful (a Tamil bill semantically similar to a Hindi bill will score higher than an unrelated English bill). This is essential for cross-state cluster detection.
3. **Efficient dimensionality:** 1,024 dimensions is sufficient for legal document semantic similarity. The 3× storage saving vs. `text-embedding-3-large` is significant at 500k+ vectors.
4. **Context window caveat:** The 512-token limit is a real constraint. See pre-processing strategy below.

---

### Pre-processing Before Embedding

#### What to Prepend

Each text chunk sent for embedding should begin with a metadata preamble:

```
Legislature: {legislature_full_name}
Bill Type: {bill_type}
Policy Domain: {primary_domain}
Year: {session_year}
---
{bill_text_chunk}
```

This preamble provides the embedding model with legislative context that anchors the semantic space. Without it, two bills on the same topic from different states may embed slightly differently due to stylistic variation in their preambles.

#### How Much Text to Include

Given the 512-token limit of `embed-multilingual-v3`:

**Strategy: Section-level embeddings with bill-level aggregation**

Do not try to embed the entire bill in one vector. Instead:

1. **Embed each section** of the bill separately (using the section's heading + text)
2. **Create a bill-level summary embedding** by averaging the section embeddings (weighted by section word count)
3. **Additionally embed the bill's "statement of objects and reasons"** separately — this is typically the most information-dense section and produces the highest-signal embedding

Store three embedding points per bill in Qdrant:
- `text_section: "full_average"` — mean of all section embeddings
- `text_section: "objects_reasons"` — embedding of the statement of objects and reasons
- `text_section: "prs_summary"` — embedding of the PRS summary (where available)

The `full_average` and `objects_reasons` embeddings are the most important for similarity detection. Use `objects_reasons` as the primary comparison point, falling back to `full_average` when the statement of objects is not available.

#### Handling Very Long Bills

Bills with more than 100 sections (uncommon but possible for major legislation like codifications):

1. Embed the statement of objects and reasons
2. Embed the definitions section (Section 2 in most Indian bills)
3. Embed sections in groups of 5, averaging within each group
4. Average all group embeddings for the bill-level vector

---

### Batching Strategy

Cohere's embed API accepts batches of up to 96 texts. To minimise API calls:

1. Accumulate newly extracted sections across all bills processed in a given run
2. Submit in batches of 90 (leaving headroom for API limits)
3. Write all embeddings to Qdrant in a single upsert call per batch

For initial historical ingestion (all historical bills at once), use an offline batch job that processes 10,000 sections per hour (staying within typical API rate limits).

---

### Embedding Update Policy

If a bill's text is later corrected or supplemented (e.g., PRS adds a summary that wasn't there initially):

1. Re-extract and re-embed only the specific `text_section` type that changed (e.g., re-embed `prs_summary` without touching `full_average`)
2. Upsert the new vector into Qdrant using the same point ID (overwrite in-place)
3. Update `embedding_updated_at` in the `bills` table
4. Log the update in `scrape_runs` as `bills_updated`

Do not delete and re-insert — Qdrant's upsert semantics handle this cleanly.

---

## 3B — Similarity Query Specifications

### Query 1: Find Similar Bills

**Purpose:** Given a bill, find the N most similar bills across all legislatures.

**Inputs:**
- `bill_id` (UUID) — the reference bill
- `n` (integer, default 10) — number of results to return
- `min_similarity` (float, default 0.75) — minimum cosine similarity threshold
- `exclude_same_legislature` (boolean, default False) — if True, exclude bills from the same legislature as the reference bill
- `filters` (optional dict) — additional Qdrant metadata filters (e.g., `{"session_year": {"gte": 2020}}`)

**Logic:**
1. Look up the reference bill in Qdrant by `bill_id`; retrieve its `objects_reasons` embedding (fall back to `full_average` if unavailable)
2. Submit a Qdrant `search` request with:
   - The reference embedding as the query vector
   - `limit = n * 3` (over-fetch to account for same-bill matches and filtered results)
   - `with_payload = True`
   - Apply `filters` if provided
3. From results, remove:
   - The reference bill itself
   - Bills with cosine similarity < `min_similarity`
   - If `exclude_same_legislature`, bills from the same `legislature`
4. Truncate to `n` results

**Output:**
```json
{
  "reference_bill_id": "uuid",
  "results": [
    {
      "bill_id": "uuid",
      "title_english": "...",
      "legislature": "...",
      "date_introduced": "2023-07-15",
      "cosine_similarity": 0.92,
      "bill_type": "government",
      "current_status": "enacted"
    }
  ]
}
```

**Threshold guidance:** A cosine similarity of ≥0.90 indicates near-identical legislation (likely model legislation). 0.75–0.89 indicates substantively similar legislation on the same policy issue. <0.75 is topic-related but distinct.

---

### Query 2: Find Bill Clusters

**Purpose:** Group all bills into clusters of near-identical legislation — the core model-legislation detector.

**Inputs:**
- `similarity_threshold` (float, default 0.88) — minimum cosine similarity for two bills to be in the same cluster
- `min_cluster_size` (integer, default 2) — minimum bills per cluster to be reported
- `embedding_type` (string, default "objects_reasons") — which embedding to use for clustering

**Logic:**

This is a graph-based clustering problem. The algorithm:

1. **Build a similarity graph:** For each bill (node), find all bills with cosine similarity ≥ `similarity_threshold` using Qdrant batch search. Add edges between each pair of similar bills.

2. **Find connected components:** Apply a standard connected-components algorithm (BFS/DFS) on the similarity graph. Each connected component is a cluster.

3. **For each cluster with ≥ `min_cluster_size` members:**
   a. **Identify the template bill:** The bill with the earliest `date_introduced` in the cluster
   b. **List adopters:** All other bills in the cluster, sorted by `date_introduced`
   c. **Compute adoption lag:** For each non-template bill, compute `date_introduced - template.date_introduced` in days

**Output per cluster:**
```json
{
  "cluster_id": "generated-uuid",
  "cluster_size": 5,
  "template_bill": {
    "bill_id": "uuid",
    "title_english": "The XYZ Bill, 2019",
    "legislature": "lok_sabha",
    "date_introduced": "2019-02-01",
    "ruling_party": "BJP"
  },
  "adopters": [
    {
      "bill_id": "uuid",
      "title_english": "The XYZ (State Amendment) Bill, 2020",
      "legislature": "up_vidhan_sabha",
      "date_introduced": "2020-03-15",
      "adoption_lag_days": 408,
      "ruling_party": "BJP",
      "current_status": "enacted"
    }
  ],
  "mean_cosine_similarity": 0.93,
  "policy_domains": ["land_revenue", "agriculture"],
  "notes": "Central bill adopted by 4 BJP-governed states within 18 months"
}
```

**Computational note:** At 100k bills, the all-pairs similarity computation is prohibitively expensive if done naively. Use Qdrant's batch search: for each bill, query the top-100 most similar bills. This generates a sparse similarity graph that can be clustered in O(n) time. Run as a scheduled weekly batch job (not real-time).

---

### Query 3: Similarity Timeline

**Purpose:** For a given cluster, show how legislation spread across legislatures chronologically.

**Input:**
- `cluster_id` (UUID) — from the cluster table

**Logic:**
1. Retrieve all bills in the cluster from the `bill_clusters` table (see schema in 3B supplementary)
2. Sort by `date_introduced` ascending
3. Compute cumulative adoption count at each date
4. For each bill, retrieve: legislature, state, ruling_party, current_status, cosine_similarity to template

**Output:**
```json
{
  "cluster_id": "uuid",
  "template_bill": {...},
  "timeline": [
    {
      "sequence": 1,
      "bill_id": "uuid",
      "legislature": "lok_sabha",
      "state_code": null,
      "date_introduced": "2019-02-01",
      "days_from_template": 0,
      "ruling_party": "BJP",
      "current_status": "enacted",
      "cosine_similarity_to_template": 1.0
    },
    {
      "sequence": 2,
      "bill_id": "uuid",
      "legislature": "up_vidhan_sabha",
      "state_code": "IN-UP",
      "date_introduced": "2020-03-15",
      "days_from_template": 408,
      "ruling_party": "BJP",
      "current_status": "enacted",
      "cosine_similarity_to_template": 0.94
    }
  ]
}
```

This output directly produces a timeline visualisation (horizontal timeline, nodes coloured by ruling_party).

---

### Query 4: Cross-State Similarity Matrix

**Purpose:** Compute a matrix where each cell `M[i][j]` is the average pairwise cosine similarity between all bills from legislature `i` and all bills from legislature `j`.

**Inputs:**
- `legislatures` (list of strings) — which legislatures to include (default: all 8 target legislatures)
- `session_year_range` (optional tuple) — filter to bills introduced in a given year range
- `policy_domain` (optional string) — filter to a specific policy domain

**Logic:**

For each pair of legislatures `(i, j)`:
1. Retrieve all bill embeddings for legislature `i` and legislature `j` from Qdrant (filtered by `session_year_range` and `policy_domain` if provided)
2. Compute the mean pairwise cosine similarity across all bill pairs `(bill_from_i, bill_from_j)`
3. For efficiency at large scale: sample up to 500 bills per legislature before computing pairwise similarities (random sample, seeded for reproducibility)

**Output:**
```json
{
  "matrix": {
    "lok_sabha": {
      "lok_sabha": 1.0,
      "up_vidhan_sabha": 0.71,
      "maharashtra_vidhan_sabha": 0.68,
      ...
    },
    "up_vidhan_sabha": {
      "lok_sabha": 0.71,
      "up_vidhan_sabha": 1.0,
      ...
    }
  },
  "bill_counts": {
    "lok_sabha": 320,
    "up_vidhan_sabha": 487,
    ...
  },
  "filters_applied": {
    "session_year_range": [2019, 2024],
    "policy_domain": null
  }
}
```

The matrix is symmetric. Present as a heatmap for analysis. Cells where one or both legislatures have <20 bills should be flagged as `low_confidence`.

---

### Supplementary Schema: `bill_clusters` and `bill_cluster_members`

These tables store the output of the weekly clustering batch job:

```sql
CREATE TABLE bill_clusters (
    cluster_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    similarity_threshold NUMERIC(4,3) NOT NULL,
    cluster_size        INTEGER NOT NULL,
    template_bill_id    UUID NOT NULL REFERENCES bills(bill_id),
    mean_similarity     NUMERIC(4,3),
    policy_domains      TEXT[],
    notes               TEXT
);

CREATE TABLE bill_cluster_members (
    cluster_id          UUID NOT NULL REFERENCES bill_clusters(cluster_id),
    bill_id             UUID NOT NULL REFERENCES bills(bill_id),
    cosine_similarity_to_template NUMERIC(4,3),
    adoption_lag_days   INTEGER,
    is_template         BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (cluster_id, bill_id)
);
```

---

## 3C — Policy Domain Classification Design

### Proposed Starter Taxonomy

The following 20 domains cover the major legislative categories active in Indian state and central legislatures:

| Code | Domain | Notes |
|---|---|---|
| `agriculture` | Agriculture & Food Security | Farming, irrigation, APMC, land acquisition for farming |
| `environment` | Environment & Climate | Pollution control, forests, wildlife, climate adaptation |
| `land_revenue` | Land Revenue & Property | Land records, tenancy, urban land, registration |
| `education` | Education | Schools, universities, technical education, scholarships |
| `health` | Public Health | Hospitals, drugs, medical education, epidemics |
| `labour_employment` | Labour & Employment | Industrial relations, wages, employment exchanges, MGNREGA |
| `urban_development` | Urban Development | Municipalities, town planning, housing, real estate |
| `rural_development` | Rural Development & Panchayati Raj | Panchayats, rural infrastructure, PMGSY |
| `police_law_order` | Police & Law & Order | Police acts, criminal procedure, prisons, forensics |
| `taxation_finance` | Taxation & Finance | State taxes, budgets, fiscal management, GST amendments |
| `commerce_industry` | Commerce & Industry | SEZs, industrial policy, MSME, investment promotion |
| `energy` | Energy & Power | Electricity distribution, renewable energy, petroleum |
| `transport` | Transport & Infrastructure | Roads, public transport, vehicle regulation, airports |
| `water` | Water Resources | Rivers, irrigation, dams, drinking water |
| `social_welfare` | Social Welfare & Protection | SC/ST/OBC welfare, disability, senior citizens, women |
| `food_civil_supplies` | Food & Civil Supplies | PDS, ration shops, essential commodities |
| `governance_admin` | Governance & Administration | Civil service, anti-corruption, RTI, service rules |
| `religious_cultural` | Religious & Cultural | Endowments, waqf, religious places, cultural heritage |
| `constitution_amendment` | Constitutional Amendments | Bills amending the Indian Constitution |
| `other` | Other / Unclassified | Catchall for bills not fitting the above |

This taxonomy is intentionally coarse. An implementation detail: each bill can receive multiple domain tags (an array, not a single value). The `policy_domains` column in the `bills` table and the Qdrant payload both store arrays.

---

### Classification Prompt Structure

**Model:** `gemini-2.5-flash-lite` (optimised for cost at high volume)

Classification occurs at the same time as translation (or immediately after), using the English bill text.

**System prompt:**
```
You are a policy analyst classifying Indian legislative bills by policy domain. You will be
given the text of a bill (or its summary) and must return one or more policy domain tags
from the fixed list below. Return only the domain codes, nothing else.

POLICY DOMAINS (use only these codes):
agriculture, environment, land_revenue, education, health, labour_employment,
urban_development, rural_development, police_law_order, taxation_finance,
commerce_industry, energy, transport, water, social_welfare, food_civil_supplies,
governance_admin, religious_cultural, constitution_amendment, other

Rules:
1. Return between 1 and 3 domain codes, separated by commas.
2. Choose the most specific applicable domain(s).
3. Use "other" only when no other code applies.
4. Return ONLY the comma-separated list of codes. No explanation, no punctuation, no
   markdown. Example output: agriculture,water
```

**User message:**
```
Bill Title: {title_english}
Legislature: {legislature_full_name}
Year: {session_year}
Bill Type: {bill_type}

{first_2000_words_of_english_text}
```

Using only the first 2,000 words (approximately 2,600 tokens) is sufficient for domain classification and minimises token cost. The full bill text is not needed for this task.

**Parsing the output:** Split on commas, strip whitespace, validate each code against the taxonomy list. If the model returns an invalid code, log a `classification_error` and default to `other`.

---

### Making the Taxonomy Configurable Without Re-classification

**Problem:** Adding new domain codes, renaming existing codes, or splitting a domain should not require re-running the classifier on all existing bills.

**Solution: Domain mapping table**

```sql
CREATE TABLE policy_domain_taxonomy (
    code            TEXT PRIMARY KEY,
    label           TEXT NOT NULL,
    description     TEXT,
    parent_code     TEXT REFERENCES policy_domain_taxonomy(code),
    is_active       BOOLEAN DEFAULT TRUE,
    deprecated_in_favour_of TEXT REFERENCES policy_domain_taxonomy(code),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

When a domain is renamed or split:
1. Insert the new domain code(s) into `policy_domain_taxonomy` with `is_active = TRUE`
2. Mark the old code with `deprecated_in_favour_of = new_code` and `is_active = FALSE`
3. Run a SQL update to migrate existing `bills.policy_domains` from old code to new code

This requires a one-time SQL migration for taxonomy changes, not a re-classification of bills.

When a new domain is added that didn't exist in the original taxonomy, existing bills do not need to be re-classified unless the analyst specifically identifies bills likely to belong to the new domain. The classification prompt is updated to include the new code, and all new bills will use the updated taxonomy automatically.

---

## 3D — Pattern Detection Query Specifications

### Query 1: Ruling Party Alignment

**Research question:** Are states governed by the same party passing significantly more similar bills to each other than to states governed by other parties?

**Data required:**
- `bills` table with `ruling_party` field populated (requires external party affiliation data — see Phase 4)
- Bill embeddings in Qdrant
- The cross-state similarity matrix (Query 4 from 3B)

**Logic:**

1. Compute the cross-state similarity matrix for all state legislature pairs
2. For each pair of state legislatures `(i, j)`, look up whether `ruling_party[i] == ruling_party[j]` at the time of each bill's introduction
3. Group all matrix cells into two sets: `same_party_pairs` and `different_party_pairs`
4. Compute mean similarity for each group
5. Run a Mann-Whitney U test (non-parametric, robust to non-normal distributions) on the two groups to assess statistical significance

**Output:**
```json
{
  "same_party_mean_similarity": 0.74,
  "different_party_mean_similarity": 0.61,
  "difference": 0.13,
  "p_value": 0.003,
  "statistically_significant": true,
  "sample_size_same_party": 42,
  "sample_size_different_party": 156,
  "caveat": "Ruling party data populated for 7 of 8 target legislatures. Central government (BJP-led NDA) treated as BJP-aligned."
}
```

**External data requirement:** A `legislature_party` table mapping `(legislature, date_from, date_to, ruling_party)`. This must be maintained manually or sourced from election result databases (see Phase 4).

---

### Query 2: Legislative Velocity

**Purpose:** How many bills in a given policy domain has a state introduced per quarter? Detects policy sprints.

**Inputs:**
- `legislature` (string) — which legislature
- `policy_domain` (string) — which domain
- `start_date`, `end_date` (dates) — time window

**Logic:**
1. Query `bills` table: `WHERE legislature = ? AND ? = ANY(policy_domains) AND date_introduced BETWEEN ? AND ?`
2. Group by calendar quarter (extract year and quarter from `date_introduced`)
3. Count bills per quarter
4. Compute rolling 4-quarter average to smooth noise
5. Flag quarters where bill count exceeds `mean + 2 × std_dev` as "sprint detected"

**Output:**
```json
{
  "legislature": "gujarat_vidhan_sabha",
  "policy_domain": "land_revenue",
  "quarters": [
    {"period": "2022-Q1", "bill_count": 2, "rolling_avg": 1.8},
    {"period": "2022-Q2", "bill_count": 8, "is_sprint": true, "rolling_avg": 3.2}
  ]
}
```

---

### Query 3: Template Legislation Report

**Purpose:** Full report of all detected bill clusters with 3 or more members, ranked by adoption breadth.

**Logic:**
1. Query `bill_clusters` table: `WHERE cluster_size >= 3`
2. For each cluster, retrieve template bill details and all members
3. Join with `bills` to get ruling_party for each adopting state at time of adoption
4. Compute:
   - Number of distinct states adopting (distinct `state_code` values in cluster)
   - Number of states where bill was enacted vs. only introduced
   - "Party spread" — number of distinct ruling parties among adopters
5. Rank by number of adopting states (descending)

**Output:** A structured report with one row per cluster, showing: cluster size, template bill, template legislature, template date, adopting states, state count, party spread, fastest adopter, slowest adopter, mean adoption lag.

---

### Query 4: Early Adopter Identification

**Purpose:** Which states consistently introduce bills in a domain before others? These are the policy laboratories.

**Inputs:**
- `policy_domain` (string)
- `min_cluster_size` (integer, default 3) — only count clusters with at least this many members

**Logic:**
1. For each cluster containing `policy_domain` with `cluster_size >= min_cluster_size`:
   - Record which legislature introduced the bill earliest (the template, or the first adopter if template is central Parliament)
   - Record the state code of the first state-level introducer
2. Count, per state, how many times it was the first state adopter in the given domain
3. Compute an "early adopter score" = `times_first / times_in_cluster`
4. Rank states by early adopter score, filtering to states that appeared in at least 3 clusters

**Output:**
```json
{
  "policy_domain": "labour_employment",
  "early_adopters": [
    {
      "state_code": "IN-MH",
      "state_name": "Maharashtra",
      "times_first_state_adopter": 7,
      "times_in_cluster": 12,
      "early_adopter_score": 0.58,
      "representative_bills": [...]
    }
  ]
}
```

---

### Query 5: Bill Mortality Rate

**Purpose:** What percentage of bills lapse without being passed? Compare across legislatures and over time.

**Inputs:**
- `legislatures` (list, optional) — if omitted, compute for all
- `session_year_range` (tuple, optional)
- `bill_type` (string, optional) — e.g., filter to `private_member` only

**Logic:**

For each legislature, compute:
- `introduced_count` = `COUNT(*) WHERE legislature = ? AND date_introduced BETWEEN ? AND ?`
- `lapsed_count` = `COUNT(*) WHERE ... AND current_status = 'lapsed'`
- `enacted_count` = `COUNT(*) WHERE ... AND current_status = 'enacted'`
- `pending_count` = `COUNT(*) WHERE ... AND current_status IN ('introduced', 'committee', 'passed')`
- `mortality_rate` = `lapsed_count / (lapsed_count + enacted_count)` (exclude pending bills from denominator)

**Caveat:** Bill mortality data depends on accurate `current_status` tracking. For state bills, status data is less complete than for central bills. Only compute mortality rates for states where `pending_count / introduced_count < 0.3` (i.e., most bills have a known outcome).

**Output:**
```json
{
  "bill_mortality": [
    {
      "legislature": "lok_sabha",
      "introduced_count": 450,
      "lapsed_count": 89,
      "enacted_count": 361,
      "pending_count": 0,
      "mortality_rate": 0.198,
      "mortality_rate_private_member": 0.94,
      "data_completeness": 1.0
    }
  ]
}
```

---

*End of Phase 3 Report*
