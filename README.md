# AI Productivity: When AI Enters the Workflow

**Team Members:**  
Amanda Ambrosone
Giulio Presaghi
Beatrice Rossi

---

## Table of Contents

1. [Introduction](#introduction)
   - [](#data-cleaning-pipeline)
2. [Methods](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Experimental Design](#usage-examples)
4. [Results](#troubleshooting)
5. [Conslusions](#conclusion)

## Section 1 — Introduction

**Research Question:** *Beyond which threshold of AI usage does rework erode operational margins?*

This project investigates the relationship between AI tool adoption and operational profitability within a digital agency context, using a dataset of 3,248 tasks provided by Alkemy. Each task carries operational metadata (team, seniority, task type, complexity), effort metrics (hours spent, billable hours, rework hours), AI usage measurements (`ai_usage_pct`, `ai_assisted`), and financial outcomes (revenue, cost, profit).

The central hypothesis is that while AI tools can accelerate individual task execution, their interaction with quality control — specifically rework — may erode profit margins beyond a certain usage threshold. The project combines rigorous data cleaning, exploratory analysis, feature engineering, and machine learning modeling to locate this threshold and quantify its operational implications.

---

## Section 2 — Methods

### Dataset

The dataset (`ai_productivity_dataset_final.csv`) contains 3,248 rows and 34 columns covering tasks executed between July 2025 and May 2026 across four teams (Content, Design, Media, SEO), seven task types (ticket, ad, article, design, report, dev, release), and three seniority levels (junior, mid, senior).

### Environment

The project was implemented in Python 3. Key libraries include:

- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling and evaluation
- `scipy` — statistical testing 

To recreate the environment:

```bash (windows)
py -m venv name-of-environment
```

Or using a unix system:

```bash (unix)
python3 -m venv name-of-environment
```
Activating the Environment:

```bash (unix)
env\Scripts\activate
```

### Data Cleaning Pipeline

The cleaning phase addressed the following issues, each documented in a dedicated notebook section:

**Duplicates (§4.1):** 48 `task_id` pairs (96 rows) with identical business data but diverging metadata. Strategy: keep the most recently `updated_at` row per task.

**Dirty categoricals (§4.5–4.6):** `team` had 15 variants for 4 actual teams; `task_type` had 29 variants for 7 canonical categories. Both were normalized via explicit mapping dictionaries covering typos, case inconsistencies, and synonyms.

**Missing value imputation:** Different strategies were applied based on the nature of each variable:

| Variable | Missingness | Strategy |
|---|---|---|
| `brief_quality_score` | 2.1% | Group median on `team × seniority × task_type` |
| `ai_usage_pct` | 4.4% | Hierarchical group median (L1: 4 vars → L2: 3 vars) |
| `rework_hours` | 2.2% | Group median on `scope_change × complexity × deadline` |
| `outcome_score` | 4.1% | Linear regression (CV R² ≈ 0.35) |
| `billable_hours` | 2.5% | `hours_spent × pricing_model recovery rate` |
| `delivered_at` | 1.2% | `created_at + sla_days`; fallback: `updated_at` |
| `sla_days` | 1.1% | Reverse-engineered from `actual_days + sla_breach` |

**`hours_spent` anomalies (§4.17):** A diagnostic-based criterion using `billable_ratio = billable_hours / hours_spent` identified 56 corrupted records. Those with `billable_ratio < 0.15` (inflated `hours_spent`) or `billable_ratio > 5` (deflated) were reconstructed via `billable_hours / recovery_rate(pricing_model)`. One irrecoverable record (both fields corrupted) was dropped.

**Date consistency (§4.29):** 14 records had `created_at > delivered_at`. A swap test was applied: 8 consistent records had their dates transposed; 6 ambiguous records were dropped.

### Variable Interpretation: Validating Task-Level Hours

A key methodological step (§3.7) established that `hours_spent`, `billable_hours`, and `rework_hours` are **task-level aggregates** — not per-phase measurements. Three tests confirmed this:

1. **Duplicate snapshot test:** Among 36 task pairs captured at different `workflow_stage` values, 34/36 showed identical `hours_spent`. This rules out both per-phase and cumulative interpretations.
2. **Variance decomposition (η²):** `task_type` explains ~16.5% of `hours_spent` variance; `workflow_stage` explains only 0.16% — a two-order-of-magnitude gap.
3. **Face validity (Kruskal-Wallis, p ≈ 10⁻¹⁴³):** The task-type ranking `ticket < ad < article < design < report < dev < release` is semantically coherent with agency operations.

This interpretation means `workflow_stage` and `task_status` are independent administrative labels and should not be treated as ordinal predictors of effort.

### Feature Engineering

Four derived features were constructed:

```python
df['margin']           = df['profit'] / df['revenue']
df['efficiency_ratio'] = df['billable_hours'] / df['hours_spent']
df['rework_ratio']     = df['rework_hours'] / df['hours_spent']
df['delivery_speed']   = df['actual_days'] / df['sla_days']
```

These transform raw operational fields into interpretable KPIs directly linked to the research question. `rework_ratio` is the critical mediator: the hypothesized causal chain is AI usage → rework ratio → efficiency ratio → profit margin.

---

## Section 3 — Experimental Design

### Experiment 1 — Locating the AI-Rework Threshold

**Purpose:** Identify the `ai_usage_pct` level at which rework cost begins to erode profit margins.

**Baseline:** A naïve model predicting `margin` from task-intrinsic variables only (task type, complexity, seniority), ignoring AI usage.

**Evaluation Metrics:**
- **R²** (explained variance in `margin`)
- **RMSE** (prediction error in margin units)
- Threshold identified via breakpoint analysis on the `ai_usage_pct` → `rework_ratio` curve

### Experiment 2 — AI Impact Across Task Types

**Purpose:** Test whether the AI-rework relationship differs by task type (short atomic tasks vs. long structured deliverables).

**Baseline:** A single global model without task-type interaction terms.

**Evaluation Metrics:**
- **η² per task type** (variance explained by AI usage within each type)
- **Negative-profit rate** as a function of `ai_usage_pct` bin and `task_type`

---

## Section 4 — Results

### Main Findings

**Finding 1 — Rework ratio is structurally higher for short task types**

![Rework ratio by task type](images/fig_task_level_validation.png)

Short tasks (ticket: 23.3%, ad: 20.9%) carry proportionally more rework than long structured tasks (release: 12.4%, dev: 13.2%). This is a first empirical hint of the AI productivity paradox: the tasks most amenable to AI assistance are also the ones most burdened by rework relative to their size.

**Finding 2 — AI usage is driven by seniority, complexity, and deadline pressure**

![AI usage imputation segmentation](images/fig_ai_usage_segmentation.png)

The four-variable combination `seniority × task_complexity_score × team × deadline_pressure` explains 21.8% of `ai_usage_pct` variance — three times more than any single variable. Junior staff under high deadline pressure shows the highest AI adoption, consistent with AI being used as a coping mechanism for time constraints.

**Finding 3 — Rework is driven by process variables, not who performs the task**

![Rework hours segmentation](images/fig_rework_segmentation.png)

`scope_change_flag × task_complexity_score × deadline_pressure` explains 8.4% of `rework_hours` variance, while `seniority` alone explains virtually nothing. Rework is determined by task circumstances, not operator experience — which means AI-induced rework may be systematic rather than correctable through staffing.

**Finding 4 — 25% of tasks are delivered at a loss**

The negative-profit rate varies by task type: ticket (34.4%), release (31.4%), ad (24.5%), report (23.5%). These are precisely the task types most likely to involve AI assistance, establishing the business relevance of the AI-margin relationship.

| Metric | Value |
|---|---|
| Dataset size (after cleaning) | 3,193 rows × 38 columns |
| Rows dropped | 55 (1.7% of original) |
| Missing values imputed | 651 across 8 variables |
| Corrupted records reconstructed | 96 (billable + hours anomalies) |
| Median profit margin | 29.1% |
| Tasks at a loss | 25.1% |

---

## Section 5 — Conclusions

### Summary

This project developed a complete data science pipeline — audit, cleaning, variable interpretation, feature engineering, and exploratory modeling — to investigate whether AI tool usage erodes operational margins through increased rework in a digital agency context. The key finding is that the AI-rework relationship is **task-type dependent**: short, atomic tasks (ticket, ad) already carry disproportionately high rework ratios, and are the same task types where AI adoption is highest. This creates a risk concentration that is not visible when aggregating across the entire dataset.

A second finding challenges naive assumptions about AI and seniority: junior staff under deadline pressure are the highest adopters of AI, but the analysis shows that rework is driven by task circumstances (scope changes, complexity, deadline pressure) rather than operator experience. This suggests that AI-induced rework, if it exists, would be systematic rather than correctable through staffing decisions alone.

The threshold question — *beyond which AI usage level does rework erode margins?* — remains partially open: the EDA established the structural conditions under which the paradox would manifest, and provided the engineered features (`rework_ratio`, `margin`, `efficiency_ratio`) needed to test it quantitatively in the modeling phase.

### Limitations and Future Work

Several questions are not fully answered by this work. First, the modeling phase is incomplete: the threshold analysis on `ai_usage_pct` → `margin` requires fitting non-linear models (piecewise regression, GAMs, or tree-based methods with monotonicity constraints) that were not executed in the current notebook. Second, the causal direction is not established: high `rework_ratio` in high-AI tasks may reflect self-selection (AI is used on harder tasks) rather than AI causing rework. A proper causal analysis would require instrumental variables or a natural experiment.

Natural next steps include: (1) fitting a threshold model on `ai_usage_pct` with `margin` as outcome, controlling for `task_type` and `complexity`; (2) testing the `release × client_review` operational bottleneck identified in the heatmap analysis (§3.7.4); and (3) extending the analysis to include `legacy_ai_flag` as a moderator — if legacy AI tools produce systematically different rework patterns than modern ones, the threshold may shift depending on tool generation.