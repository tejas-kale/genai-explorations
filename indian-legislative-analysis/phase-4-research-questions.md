# Phase 4: Research Questions
## Indian Legislative Analysis System

**Date:** 4 March 2026
**Status:** Complete
**Prerequisite:** Phases 1–3 approved

---

## Overview

This document catalogues every research question the proposed pipeline can answer, organised by theme. For each question, the following are specified:

- **Data required:** Which tables, fields, and external sources are needed
- **Pipeline functions:** Which Phase 3 queries and Phase 2 pipeline outputs are needed
- **Output format:** The most useful presentation for analysts and journalists
- **Data availability:** Whether the current pipeline can fully answer it
- **Analytical complexity:** Straightforward query / Clustering or ML / External data required

---

## On Model Legislation and Spread

---

### Q1. Which bills have been introduced in near-identical form across 3 or more state legislatures? Who introduced each, and in what order?

**Data required:**
- `bills` table: `bill_id`, `title_english`, `legislature`, `state_code`, `date_introduced`, `sponsoring_member`, `ministry_department`, `ruling_party`, `current_status`
- `bill_clusters` and `bill_cluster_members` tables (output of clustering batch job)
- Bill embeddings in Qdrant

**Pipeline functions:**
- Phase 3 Query 2 (Find Bill Clusters) to generate clusters
- Phase 3 Query 3 (Similarity Timeline) to show adoption order per cluster

**Output format:** A table with one row per cluster, showing cluster ID, policy domain, template bill, template legislature, template date, and a nested list of adopters in chronological order (legislature, date, ruling party, status, cosine similarity to template). A separate timeline visualisation (horizontal axis = time, nodes = adoptions) renders the spread graphically.

**Data availability:** ✅ **Fully answerable** — the current pipeline captures all required fields. `sponsoring_member` is available from PRS India for central bills; state bill sponsorship is less consistently captured but available where PRS provides it.

**Analytical complexity:** Clustering/ML — requires the weekly clustering batch job (Phase 3B Query 2). Once clusters are computed, the output query is a straightforward database join.

---

### Q2. Are there organisations (industry bodies, think tanks, central government ministries) whose fingerprints appear across clusters of similar state bills?

**Data required:**
- Bill cluster data from Q1
- `bills.ministry_department` — identifies central government sponsor
- External data: known affiliations between think tanks/lobby groups and specific bills (not currently in pipeline)

**Pipeline functions:**
- Phase 3B Query 2 (clusters)
- Supplementary: LLM-based named-entity extraction on bill preambles and statement of objects, identifying references to draft reports, recommendations, or model legislation sources

**Output format:** A network graph where nodes are organisations/ministries and bills, and edges represent "this organisation is associated with this bill." Clusters of similar bills connected to the same organisation node reveal systematic drafting.

**Data availability:** ⚠️ **Partially answerable** — `ministry_department` is captured for central government bills. Identification of external organisations (industry bodies, think tanks) requires additional LLM extraction on bill text. Model legislation references (e.g., "in accordance with the recommendations of the..." or "based on the draft prepared by...") can be extracted from bill preambles using a targeted LLM prompt. This is not in the current Phase 2 pipeline but is a feasible addition.

**Analytical complexity:** Requires ML (NER/LLM extraction) + external data (known affiliations). Straightforward once extraction is added to the pipeline.

---

### Q3. How long does it typically take for a central government bill or policy to be replicated in state legislation? Does this lag differ by ruling-party alignment?

**Data required:**
- `bill_clusters` and `bill_cluster_members` tables
- `bills.legislature_type` (to distinguish central from state bills)
- `bills.ruling_party` and `legislature_party` external table (see Phase 3D Query 1)

**Pipeline functions:**
- Phase 3B Query 2 (clusters) — to identify central→state adoption patterns
- Phase 3D Query 1 (ruling party alignment) — for party-lag correlation

**Output format:** Two outputs:
1. A distribution histogram of adoption lag (in days) from central bill introduction to first state adoption, across all clusters where the template is a central bill
2. A grouped bar chart: mean adoption lag for same-party vs. different-party states

**Data availability:** ✅ **Fully answerable** — all required fields are in the pipeline, subject to `ruling_party` being populated via the external party affiliation table.

**Analytical complexity:** Moderate — requires clustering output + joins. The party-lag correlation is a simple group-by after clustering.

---

## On Political Patterns

---

### Q4. Do BJP-governed states pass significantly more similar legislation to each other than to non-BJP-governed states, and vice versa for Congress and regional parties?

**Data required:**
- Bill embeddings in Qdrant
- `bills.ruling_party`
- `legislature_party` external table mapping state legislatures to ruling parties by date

**Pipeline functions:**
- Phase 3B Query 4 (Cross-State Similarity Matrix) — computed separately for each ruling-party grouping
- Phase 3D Query 1 (Ruling Party Alignment) — the primary function for this question

**Output format:**
1. A 2×2 similarity matrix: same-BJP vs. cross-party vs. same-Congress vs. same-regional
2. Box plots showing distribution of pairwise similarities within each party grouping
3. Statistical significance test (Mann-Whitney U) results

**Data availability:** ✅ **Fully answerable** — subject to `ruling_party` data being complete and current.

**Analytical complexity:** Moderate — standard statistical comparison after embeddings are computed. The key challenge is maintaining accurate `ruling_party` data, particularly when coalition governments change composition mid-term.

---

### Q5. When a state's ruling party changes, how does its legislative output change in the following two years?

**Data required:**
- `bills` table: all bills for target states across the study period
- `legislature_party` external table: `(state_code, party_change_date, old_party, new_party)`
- `bills.policy_domains` — to track domain shifts
- Bill word counts and section counts — for complexity tracking
- Bill mortality rates — for passage rate comparison

**Pipeline functions:**
- Phase 3D Query 2 (Legislative Velocity) — compare velocity before and after government change
- Phase 3D Query 5 (Bill Mortality Rate) — compare passage rates before and after
- Phase 3B Query 4 (Cross-State Similarity Matrix) — compare similarity to other BJP-states vs. other states before and after

**Output format:** A before/after comparison for each government change event, showing:
- Legislative velocity by domain (bills per quarter before vs. after)
- Similarity to same-old-party states vs. same-new-party states
- Shift in policy domain mix
- Change in bill complexity (word count)

**Data availability:** ⚠️ **Partially answerable** — the pipeline has all bill-level data. Requires the external `legislature_party` change-event table, which must be populated from election results databases (Election Commission of India data is publicly available).

**Analytical complexity:** Requires external data (election results) + multiple pipeline queries. Straightforward once the external table is populated.

---

### Q6. Which states are net legislative exporters (their bills are copied elsewhere) vs. net importers?

**Data required:**
- `bill_clusters` and `bill_cluster_members`
- `bills.legislature_type` and `bills.state_code`

**Pipeline functions:**
- Phase 3B Query 2 (clusters)
- Phase 3B Query 4 (early adopter identification)

**Logic:** For each cluster, the template bill's legislature is the "exporter." All adopters are "importers." Compute: for each state, `export_count = times it is the template in a multi-member cluster` and `import_count = times it is an adopter`. Net position = `export_count - import_count`.

**Output format:** A ranked table of states showing: export count, import count, net position, and representative exported/imported bills. A directed network graph with nodes = states and arrow thickness = number of bill adoptions in that direction.

**Data availability:** ✅ **Fully answerable** by the current pipeline.

**Analytical complexity:** Straightforward query on cluster data.

---

## On Policy Trends

---

### Q7. What policy domains are seeing a sudden increase in bill introductions across multiple states simultaneously? (Early warning system for emerging legislative priorities)

**Data required:**
- `bills` table: `date_introduced`, `state_code`, `policy_domains`
- Historical baseline of bills per domain per quarter

**Pipeline functions:**
- Phase 3D Query 2 (Legislative Velocity) — run separately for each domain and each state
- Aggregated across states: sum of velocity scores per domain per quarter

**Logic:** For each policy domain:
1. Compute the total bill introductions across all target states per quarter
2. Compare the current quarter to the trailing 8-quarter average
3. Flag domains where current quarter exceeds `mean + 1.5 × std_dev` across multiple states simultaneously (require ≥3 states to be in "sprint" for a domain to be flagged as system-wide)

**Output format:** A ranked list of "hot domains" with: domain name, states in which the sprint is occurring, current quarter count vs. historical average, example bills.

**Data availability:** ✅ **Fully answerable** — requires at least 2 years of historical data to establish a meaningful baseline.

**Analytical complexity:** Straightforward time-series aggregation once bills are classified and dated.

---

### Q8. Are there domains where bills are introduced frequently but almost never passed?

**Data required:**
- `bills` table: `policy_domains`, `current_status`, `date_introduced`, `legislature`

**Pipeline functions:**
- Phase 3D Query 5 (Bill Mortality Rate) — computed per domain rather than per legislature

**Logic:** Cross-tabulate `policy_domain` with `current_status`. For domains with `mortality_rate > 0.85` and `introduced_count > 20`, flag as "high introduction, low passage" domains.

**Output format:** A table of domains with: domain name, total bills introduced, enacted count, lapsed count, mortality rate. Separate columns for central vs. state legislature mortality rates.

**Data availability:** ✅ **Fully answerable**, subject to completeness of `current_status` field (which is more complete for central bills than state bills).

**Analytical complexity:** Straightforward SQL aggregation.

---

### Q9. How has the balance between government bills and private member bills shifted over time?

**Data required:**
- `bills` table: `bill_type`, `date_introduced`, `legislature`, `current_status`

**Pipeline functions:** No similarity/embedding functions required — this is a pure SQL aggregation.

**Logic:** Group bills by `(legislature, year, bill_type)` and compute:
- Count of government bills vs. private member bills per year
- Passage rate for each type per year
- Session-to-session trend

**Output format:** Line chart: government bills (solid line) vs. private member bills (dashed line) per year, per legislature. Separate charts for central Parliament and the six target states. Highlight inflection points (years where the ratio changed significantly).

**Data availability:** ✅ **Fully answerable** — `bill_type` is well-captured from PRS India for most bills.

**Analytical complexity:** Straightforward time-series SQL.

---

## On Legislative Process and Quality

---

### Q10. Which legislatures have the highest bill mortality rate?

**Data required:**
- `bills` table: `legislature`, `current_status`, `bill_type`, `date_introduced`

**Pipeline functions:**
- Phase 3D Query 5 (Bill Mortality Rate) applied to all target legislatures

**Output format:** A ranked table of legislatures by mortality rate, with separate columns for government bill mortality and private member bill mortality. Include total bill counts and a "data completeness" flag.

**Data availability:** ✅ **Fully answerable** for central Parliament (complete data). State legislature mortality rates depend on `current_status` accuracy, which is lower.

**Analytical complexity:** Straightforward aggregation.

---

### Q11. How does bill complexity (word count, section count) correlate with passage rate?

**Data required:**
- `bills` table: `word_count_english`, `section_count`, `current_status`, `legislature`, `bill_type`

**Pipeline functions:** No embedding functions required.

**Logic:**
1. For each bill with known outcome (`current_status IN ('enacted', 'lapsed')`), record `(word_count, section_count, status)`
2. Run logistic regression: `P(enacted) ~ word_count + section_count + bill_type + legislature`
3. Report odds ratios and confidence intervals

**Output format:** A scatter plot of word count vs. passage rate (binned), with a fitted logistic curve. A coefficient table from the regression showing the marginal effect of each complexity measure on passage probability.

**Data availability:** ✅ **Fully answerable** — word count and section count are extracted during Phase 2 text processing.

**Analytical complexity:** Requires basic statistical analysis (logistic regression). Not a machine learning task — standard statistics library (scikit-learn or statsmodels) suffices.

---

### Q12. Are bills introduced in the final session before an election systematically different in domain or complexity from those introduced at other times?

**Data required:**
- `bills` table: all fields
- `session_name`, `date_introduced` — to identify pre-election sessions
- External data: Indian state and central election dates (publicly available from Election Commission of India)

**Pipeline functions:**
- Phase 3D Query 2 (Legislative Velocity by domain) — compare election-session vs. non-election-session quarters
- Phase 3B Query 4 (cross-state similarity matrix) — are election-session bills unusually similar across states? (Possible indicator of coordinated political messaging via legislation)

**Logic:**
1. Classify each session as "pre-election" (within 6 months of a scheduled election) or "non-election"
2. Compare domain distribution of bills in each category (chi-square test on domain frequencies)
3. Compare mean word count and section count between categories
4. Compare the cross-state similarity scores for election-session bills vs. non-election-session bills

**Output format:**
1. A domain distribution bar chart: pre-election sessions vs. other sessions (showing if certain domains dominate pre-election legislating)
2. A complexity comparison (box plots of word count by session type)
3. A similarity score comparison

**Data availability:** ⚠️ **Partially answerable** — requires external election date data to classify sessions. Once election dates are in the database, the comparison is straightforward.

**Analytical complexity:** Moderate — straightforward statistics + requires external election calendar data.

---

## On Language and Accessibility

---

### Q13. What proportion of state bills are published only in regional languages with no English version?

**Data required:**
- `bills` table: `state_code`, `original_language`, `has_english_text`, `date_introduced`

**Pipeline functions:** No similarity functions required — straightforward SQL.

**Logic:** For each state: `COUNT(*) WHERE NOT has_english_text AND legislature_type = 'state'` divided by total state bills.

**Output format:** A stacked bar chart per state showing English-available vs. regional-language-only bills, broken down by year. A trend line showing whether English availability is increasing over time.

**Data availability:** ✅ **Fully answerable** by the current pipeline. `has_english_text` is set during Phase 2 extraction.

**Analytical complexity:** Straightforward SQL.

---

### Q14. Are bills published only in regional languages substantively different by domain or complexity from those published bilingually?

**Data required:**
- `bills` table: `has_english_text`, `original_language`, `word_count_english`, `section_count`, `policy_domains`, `bill_type`, `current_status`

**Pipeline functions:**
- Phase 3B Query 1 (Find Similar Bills) — compare similarity scores between English-only and regional-language-only bills to test if they are topically distinct

**Logic:**
1. Compare domain distribution: `has_english_text = TRUE` vs. `has_english_text = FALSE` (chi-square test per state)
2. Compare complexity: mean word count and section count for each group
3. Compare passage rates: `current_status = 'enacted'` rate for each group

**Hypothesis being tested:** A plausible hypothesis is that regional-language-only bills are more likely to address locally specific or politically sensitive topics (e.g., land records, religious endowments) that the state government does not wish to highlight in English-language media. The data can test this hypothesis.

**Output format:** A side-by-side domain distribution chart for each state comparing English-available vs. regional-language-only bills. A complexity and passage rate comparison table. A narrative assessment of whether the hypothesis is supported.

**Data availability:** ✅ **Fully answerable** by the current pipeline — `has_english_text`, domain classifications, and bill metrics are all captured.

**Analytical complexity:** Moderate — standard statistical comparison. No embedding or ML required.

---

## Data Availability and Complexity Summary

| Q# | Question (abbreviated) | Pipeline Ready? | Complexity |
|---|---|---|---|
| Q1 | Near-identical bills across 3+ states | ✅ Yes | Clustering |
| Q2 | Organisation fingerprints across clusters | ⚠️ Partial (NER extraction needed) | ML + External |
| Q3 | Central→state replication lag | ✅ Yes (party data required) | Moderate |
| Q4 | BJP-state vs. cross-party similarity | ✅ Yes (party data required) | Moderate |
| Q5 | Legislative change after government change | ⚠️ Partial (election data required) | External Data |
| Q6 | Net legislative exporters vs. importers | ✅ Yes | Straightforward |
| Q7 | Emerging legislative domains (early warning) | ✅ Yes (2+ years data required) | Straightforward |
| Q8 | High-introduction, low-passage domains | ✅ Yes | Straightforward |
| Q9 | Government vs. private member bills over time | ✅ Yes | Straightforward |
| Q10 | Legislature mortality rates | ✅ Yes (state data completeness varies) | Straightforward |
| Q11 | Bill complexity vs. passage rate | ✅ Yes | Basic statistics |
| Q12 | Pre-election session bill characteristics | ⚠️ Partial (election calendar needed) | Moderate + External |
| Q13 | Regional-language-only bills proportion | ✅ Yes | Straightforward |
| Q14 | Language availability vs. bill substance | ✅ Yes | Moderate |

### External Data Sources Required (Not in Current Pipeline)

The following external datasets need to be compiled to enable the party-political analytical questions:

1. **Ruling party by state legislature by date:** A table mapping each state legislature to its governing party/coalition for each date range. Source: Election Commission of India (eci.gov.in) election results. Needs to be compiled manually for historical periods.

2. **State and central election dates:** Complete calendar of Lok Sabha, Rajya Sabha (biennial), and all state Vidhan Sabha elections. Source: ECI election archive. Largely automatable from ECI data.

3. **Think tank / lobby group bill associations:** A manually curated dataset of bills known to be drafted by or associated with specific external organisations. Source: investigative journalism, RTI responses, parliamentary committee testimony. This cannot be automated and requires editorial curation.

---

*End of Phase 4 Report*
