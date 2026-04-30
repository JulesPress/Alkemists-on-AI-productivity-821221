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


## [Section 1] Introduction
 
This project investigates how AI usage affects the operational margins of Alkemy, a digital agency. The central research question is: **"Beyond which threshold of AI usage does rework erode operational margins?"**
 
The dataset provided by Alkemy contains 3,248 tasks and 34 variables covering the full task lifecycle — inputs (briefing quality, complexity), process (hours spent, AI usage percentage), output (quality score, rework) and economic value (revenue, cost, profit). The operation spans four teams (Content, Design, Media, SEO), seven task types (ad, article, design, dev, release, report, ticket), three seniority levels, and three pricing models (hourly 48%, fixed 38%, value_based 14%).
 
The project addresses four main analytical questions and three advanced ones:
 
**Main questions:**
1. Where is value created? — In which task-type and AI-usage segments does AI simultaneously reduce effort, improve margin, and preserve quality?
2. Where are losses incurred? — Where do rework, quality erosion, and margin losses concentrate?
3. Is AI driving quality or just speed? — What economic mechanism underlies the observed margin lift?
4. When does AI become negative? — At what AI usage level do quality and rework turn against the operation?
**Advanced questions:**
1. Is the speed gain real, or an accounting illusion? — Does `hours_spent` tell the true story, or does rework absorb the apparent saving?
2. When does rework actually destroy margin? — Is there a rework intensity threshold beyond which profitability collapses?
3. When does the hourly pricing model become unsustainable? — How does the structural mismatch between AI efficiency gains and hourly billing evolve as AI adoption grows?

---

## [Section 2] Methods

The project follows a four-stage pipeline. We start with a preliminary EDA on the raw data to understand its structure, identify anomalies, and settle foundational questions about variable semantics before touching anything. We then clean each variable individually, documenting every decision with an explicit justification. Once the data is clean, we engineer the features needed to answer the research questions — profitability ratios, effort metrics, AI usage bins. Finally, we run the analysis across seven questions, moving from value creation to loss localization to threshold detection. Each stage informs the next: cleaning decisions depend on what the EDA revealed, and feature definitions depend on what the analysis requires.

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
 
### Dataset semantics: a foundational investigation
 
Before any cleaning or modeling, we conducted a formal investigation into the semantics of three pairs of variables whose interpretation was not self-evident. Getting this wrong would have invalidated the entire downstream strategy.
 
**Hours variables: task-level or per-phase?**
 
The dataset includes `hours_spent`, `billable_hours`, and `rework_hours` alongside two categorical variables describing task state: `workflow_stage` (from the PM tool) and `task_status` (from the ticketing system). A critical question was whether the hour variables record effort for the entire task or only for the phase registered at the moment of the export.
 
We answered this with three sequential tests. First, we exploited the 48 duplicate `task_id` records — the same task observed twice with different `workflow_stage` labels — as a natural experiment: if hours were per-phase, they would differ between snapshots; if task-level, they would be identical. Across 36 pairs with a stage change, 34/36 showed perfectly identical `hours_spent` and 36/36 identical `billable_hours`, decisively ruling out both the per-phase and cumulative interpretations. Hours are task-level aggregates, replicated identically across snapshots.
 
Second, we measured variance explained (η²) of `hours_spent` by each candidate grouping variable. `task_type` explains 16.5% of the variance and `task_complexity_score` another 9.3%, while `workflow_stage` (0.18%) and `task_status` (0.05%) sit at noise level — a two-order-of-magnitude gap. A within-task_type control confirmed that the flat-administrative pattern is structural and not a confounding artifact.
 
Third, we verified face validity: the ranking ticket (7.7h) < ad (8.7h) < article (10.5h) < design (11.5h) < report (12.2h) < dev (13.8h) < release (15.0h) is statistically significant (Kruskal-Wallis p ≈ 10⁻¹⁴³) and operationally coherent with how a digital agency works. Two-dimensional heatmaps confirmed that the dominant pattern in `hours_spent` is vertical (between task types), not horizontal (within task type across administrative labels).
 
This investigation established that `hours_spent`, `billable_hours`, and `rework_hours` are three **independent compartments**, not nested subsets. The correct total effort measure is `total_effort = hours_spent + rework_hours`. `workflow_stage` and `task_status` operate on independent axes (PM tool vs. ticketing system) and carry no effort signal.
 
**`rework_hours` anomalies:** we also noted a first empirical hint of the AI paradox during this analysis. Short tasks (ticket, ad) carry proportionally higher rework-to-hours ratios (20-23%) than long tasks (dev, release: 12-13%), suggesting that the task types where AI provides the largest apparent speed gain also bear the most rework overhead relative to their size.

---

### Data Cleaning
 
All 34 columns were inspected individually. Key decisions:
 
**Duplicates:** 48 `task_id` pairs (96 rows) showed identical business data but diverging metadata. The most recent `updated_at` row was kept per pair, dropping 48 rows.
 
**Categorical normalization:** `team` had 15 variants (case errors, typos, subcategories) mapped to 4 canonical teams; `task_type` had 29 variants mapped to 7 canonical types. The result is a perfectly balanced team distribution (~25% each) and a well-balanced task type distribution (~443-478 per category).
 
**`hours_spent` anomalies:** we replaced a naive fixed-threshold approach with a diagnostic-based criterion. A record is flagged as corrupted if its `billable_ratio = billable_hours / hours_spent` falls below 0.15 or above 5 — thresholds corresponding to roughly 4.6× and 7× the population median respectively. This identified ~39 records with internally inconsistent hour accounting, cross-validated by a second independent diagnostic (`cost_per_hour < 15 €/h` vs. population median ~58 €/h). Corrupted values were imputed using `hours_spent = billable_hours / recovery_rate(pricing_model)`, where recovery rates are pricing-model-specific medians computed on clean records. Records with both fields corrupted simultaneously (irrecoverable) were dropped.
 
**`billable_hours` anomalies:** 82 missing values and 17 negative values (small write-offs at task closure, −1.90 to −0.28h) were imputed symmetrically using `billable_hours = hours_spent × recovery_rate(pricing_model)`. The same `efficiency_ratio` anchors imputation in both directions depending on which field is corrupted.
 
**`ai_usage_pct`:** 142 missing values imputed using a hierarchical group-based median strategy. A bivariate η² comparison across all candidate predictors identified `seniority × task_complexity_score × team × deadline_pressure` as the combination explaining the most variance (21.8%). To avoid degenerate medians from sparse cells, a three-level hierarchy was applied: L1 (4 variables, N ≥ 5) → L2 (3 variables, drop `team`) → L3 (2 variables). Approximately 95% of NaN records were imputed at L1; none required the L3 fallback.
 
**`rework_hours`:** 71 missing values imputed with a single-level group median on `scope_change_flag × task_complexity_score × deadline_pressure` (η² = 8.4%, the highest among tested combinations). Process-context variables — scope changes, complexity, deadline pressure — proved far more predictive of rework than team or seniority, consistent with rework being driven by task-specific events rather than worker characteristics. All 30 resulting groups had N ≥ 8, so no fallback was needed.
 
**`outcome_score`:** 132 missing values imputed using a linear regression model (5-fold CV R² ≈ 0.35) rather than group-based median, because strong continuous predictors were available (`errors` r = −0.48, `brief_quality_score` r = +0.35) and group-median imputation would have thrown away the slopes of these relationships.
 
**Date anomalies:** 14 records with `delivered_at < created_at` were resolved using a swap test. If swapping the two dates produces an `actual_days` consistent with the recorded `sla_breach`, the dates are swapped; otherwise the record is irrecoverable. 8 records were recovered by swap, 6 dropped.
 
**`ai_assisted` consistency:** a system-level rule was identified: `ai_assisted = False` when `ai_usage_pct < 0.20`. Records violating this rule in both directions were corrected, making `ai_assisted` a clean deterministic projection of `ai_usage_pct` above the 0.20 threshold.

### Feature Engineering
 
Nine features were derived from the clean dataset:
 
- `profit_margin = profit / revenue` — core profitability KPI; the primary target for threshold analysis
- `efficiency_ratio = billable_hours / hours_spent` — billing efficiency; share of work hours actually billed
- `rework_ratio = rework_hours / hours_spent` — quality cost; share of time spent on corrections
- `cost_per_hour = cost / hours_spent` — operational cost per hour worked
- `revenue_per_hour = revenue / hours_spent` — revenue generated per hour worked
- `total_effort = hours_spent + rework_hours` — true effort accounting for rework as an additive compartment
- `profit_per_hour = profit / total_effort` — true profitability per unit of effort actually invested; computed on `total_effort` rather than `hours_spent` to capture the rework cost that hourly metrics hide
- `delivery_speed = delivery_days_actual / sla_days` — SLA performance ratio; < 1 means on time
- `ai_usage_bin` — five business-driven bands aligned with Alkemy's operational categorization: low [0, 20%), low_medium [20, 40%), medium_high [40, 60%), high [60, 80%), very_high [80, 100]
The 20% boundary is fixed rather than sample-driven because it has a hard system-level meaning (`ai_assisted` threshold). Sample-driven binning would obscure exactly the threshold dynamic the research question targets.

---
 
## [Section 3] Experimental Design
 
All analytical experiments are conducted in Section 6 of the notebook. Each experiment targets one of the project's seven research questions using the cleaned and feature-engineered dataset.
 
### Q1 — Where is value created? (Section 6.1)
 
**Purpose:** Identify which combinations of task type and AI usage level generate a genuine, simultaneous improvement across all three value dimensions: less effort, higher profit margin, and stable or improved quality.
 
**Baseline:** A binary comparison between low-AI tasks (ai_usage_pct < 20%) and all other tasks. This aggregate view showed a +50% median margin lift for AI-assisted tasks but no reduction in effort and no improvement in quality — a misleading result that motivates the finer-grained segmentation.
 
**Evaluation metrics:** Δ% `total_effort`, Δpp `profit_margin`, and Δ `outcome_score` computed per task_type × ai_usage_bin cell, each relative to the same task type's own low-AI baseline. A cell qualifies as value-creating only if all three metrics move simultaneously in the favorable direction and the cell contains at least 30 observations. Statistical significance on the three axes is tested with Mann-Whitney U (two-sided).
 
### Q2 — Where are losses incurred? (Section 6.2)
 
**Purpose:** Locate where rework costs, quality erosion, and margin losses concentrate across the task_type × ai_usage_bin grid, and determine whether these losses are driven by AI usage or by task type.
 
**Baseline:** Overall dataset loss rate (share of tasks with profit_margin < 0) and overall rework_ratio median, used as reference for each bin and task type.
 
**Evaluation metrics:** Loss rate (% of tasks with negative profit_margin), median `rework_ratio`, `outcome_score` trend, and `errors` count across cells. Chi-square test for independence between ai_usage_bin and loss rate.
 
### Q3 — Quality or just speed? (Section 6.3)
 
**Purpose:** Decompose the source of the observed margin lift — determining how much comes from cost reduction (fewer or cheaper hours) versus revenue increase (higher quality output attracting better billing).
 
**Baseline:** Naive assumption that margin improvement reflects genuine productivity gains.
 
**Evaluation metrics:** Δ`cost`, Δ`revenue`, and Δ`total_effort` across ai_usage_bin, with the relative magnitude of cost and revenue contributions to the total margin lift quantified in percentage terms.
 
### Q4 — When does AI become negative? (Section 6.4)
 
**Purpose:** Identify the AI usage threshold beyond which quality degrades and rework accelerates, both in aggregate and separately for each of the four value-creating task types.
 
**Baseline:** Aggregate curves across the full dataset; per-task-type curves used as refinement to detect type-specific ceilings.
 
**Evaluation metrics:** Trend direction of `outcome_score`, `rework_ratio`, and `profit_margin` as continuous functions of `ai_usage_pct`. The threshold is defined as the first AI level at which quality or rework begins to move adversely with non-trivial magnitude.
 
### Advanced 1 — Is the speed gain real? (Section 6.7)
 
**Purpose:** Test whether the reduction in `hours_spent` at higher AI usage reflects a genuine time saving or whether the saved hours return as rework, leaving `total_effort` unchanged.
 
**Baseline:** `hours_spent` alone as the productivity metric, which is what headline delivery metrics typically report.
 
**Evaluation metrics:** Comparison of `hours_spent` and `total_effort` trends across ai_usage_bin. The gap between the two curves — expressed as absolute hours and as a share of the apparent saving — measures the illusion component.
 
### Advanced 2 — When does rework destroy margin? (Section 6.8)
 
**Purpose:** Determine whether there is a rework intensity level (rework_ratio) at which profit_margin systematically turns negative, i.e., a rework-driven margin collapse threshold.
 
**Baseline:** Assumption that higher rework erodes margin monotonically, as the project brief implies.
 
**Evaluation metrics:** Median `profit_margin` and loss rate (% profit_margin < 0) across `rework_ratio` quantile bins. A threshold would be visible as a discrete drop in median margin or a sharp increase in loss rate.
 
### Advanced 3 — When does the hourly model become unsustainable? (Section 6.9)
 
**Purpose:** Quantify how the profit disadvantage of hourly pricing evolves as AI adoption increases, and whether the disadvantage is modulated by seniority.
 
**Baseline:** Aggregate pricing-model comparison at the overall dataset level (profit_per_hour by pricing_model).
 
**Evaluation metrics:** Median `profit_per_hour` by `pricing_model × ai_usage_bin`; gap in €/hour between hourly and alternative pricing models at each AI bin; within-hourly breakdown by seniority to test whether specific seniority-pricing combinations become structurally unprofitable.
 
---

## [Section 4] Results

Looking at AI usage as a single number is misleading. AI-assisted tasks show a median profit margin about 50% higher than non-AI tasks, but effort and quality are flat. That aggregate hides very different outcomes depending on task type and AI intensity.
 
When we segment by task type and AI usage level, the picture becomes clear. AI genuinely creates value — effort down, margin up, quality stable — only in four task types: design, dev, release, and report, at moderate AI usage (25–75%). These represent about 30% of the dataset. In the best cases, effort drops by 8–20% and profit margin improves by 8–25 percentage points, with quality holding or improving slightly. Creative tasks (ad, article) never meet all three conditions at any AI level. Ticket work gains margin but saves no effort. Above 80% AI usage, quality drops sharply — up to 17 outcome points below the low-AI baseline — in every task type with enough observations.
 
On losses, the result is counterintuitive. The share of unprofitable tasks falls as AI usage rises, from 31.5% at low AI down to 14.0% at very high AI. Losses are driven by task type, not AI: ticket tasks are unprofitable in 30–42% of cases regardless of AI level. Rework grows with AI, but does not directly cause margin losses — both rework and margin rise together because both are driven by AI adoption, which also compresses costs.
 
Q3 has two parts. On the work side, AI is primarily a quality tool: the most common outcome is quality improvement without time savings (28% of tasks), while pure speed gains are rare — just 3.1% of tasks. On the financial side, the margin gains come mostly from cost reduction, not revenue growth. Cost falls by roughly €211 per task from low to high AI, while revenue rises only ~€47. This happens because AI allows junior staff to take on work that would otherwise require senior staff, which lowers the cost per task. So quality goes up and cost goes down at the same time — these two findings together explain where the margin lift comes from.
 
Q4 identifies where AI starts to hurt. The safe ceiling is around 40% AI for dev, 38–45% for design and release, and 60–65% for report — the only task type with room to push further. Beyond these points, quality scores decline and rework accelerates. The problem is that profit margin keeps rising past these ceilings, because cost compression continues. This makes margin a lagging indicator: it looks fine while quality and rework are already moving in the wrong direction.
 
The three advanced analyses add three specific findings. First, roughly half the apparent time saving from AI disappears when rework is included. At very high AI usage the saving is fully absorbed: billed hours are ~2 hours below baseline but rework adds those same 2 hours back, leaving total effort unchanged. Second, a rework-driven margin collapse never appears in the data — rework and margin both rise together, so the binding limit is quality, not rework volume. Third, pricing model is the clearest structural constraint. Under hourly billing, profit per hour stays flat at €8–15 no matter how much AI is used. Under fixed pricing it rises from €17 to €40; under value_based from €31 to over €100. When AI saves time on an hourly contract, those saved hours simply cannot be billed — the gain goes to the client. Within hourly pricing, senior tasks lose money per hour at every AI level (−2 to −11 €/h); juniors are the only group that stays profitable. The gap between hourly and non-hourly pricing gets wider as AI usage increases.

| Key metric | Value |
|---|---|
| Share of tasks where AI creates joint-axis value | ~30% |
| Optimal AI range for technical task types | 25–75% |
| Safe ceiling for dev / design / release | ~38–45% |
| Safe ceiling for report | ~60–65% |
| Best margin lift (design × high AI) | +24.8 pp |
| Speed saving absorbed by rework at high AI | ~49% |
| Speed saving absorbed by rework at very_high AI | ~100% |
| profit_per_hour gap: value_based vs. hourly at medium_high AI | +€45.9/h |
| Loss rate for ticket tasks across all AI levels | 30–42% |

----

## [Section 5] Conclusions
 
### Summary
 
AI is not a uniformly value-creating technology at Alkemy. The data is internally consistent across seven independent analytical angles and points to a single coherent picture: AI creates real, measurable value in roughly 30% of the operational mix — technical task types (dev, design, release, report) at moderate AI usage (25–75%), under non-hourly pricing — and generates no net benefit or active harm in the remaining 70%. The economic mechanism is cost-arbitrage: AI reduces the cost of production by enabling juniors to handle work previously done by seniors, compressing internal cost per hour. This is not a speed or quality story — it is a labor composition story. The safe operational ceiling is ~40–45% AI usage for most technical tasks; pushing beyond it erodes quality while the P&L continues to show improvement, making the degradation invisible until it has already occurred. The hourly pricing model is the structural bottleneck: it transfers every efficiency gain AI generates directly to clients through fewer billable hours, and this transfer grows monotonically with AI adoption. The strategic implication is therefore not "deploy more AI" but rather "stabilize what is working before scaling" — cap AI usage at the measured thresholds per task type, push harder only on report (under-utilized), disengage on creative and support work, and migrate hourly contracts toward fixed or value_based pricing before any further AI rollout.
 

### Open questions and future work DA RIVEDERE TUTTO
 
Several important questions remain unanswered by this dataset. First, the causal mechanism identified — cost-arbitrage through seniority composition shift — is inferred from observational data; it has not been tested with a proper causal design. A difference-in-differences or propensity score matching approach would be needed to rule out selection bias (simpler tasks may be assigned to both juniors and AI simultaneously). Second, the quality erosion above 40% AI usage is measured by `outcome_score` at delivery; what happens to client retention and repeat business is not in the data, so the full economic cost of quality degradation may be substantially larger than visible here. Third, the dataset lacks key variables that would sharpen the analysis: the specific AI tool used (different tools may have very different quality profiles), client satisfaction measured post-delivery rather than at handoff, the number of iterations with the client vs. internal revisions, and the marginal cost per AI tool usage per task. Fourth, the modeling section (Section 7, to be completed) will move from the descriptive threshold analysis conducted here to predictive models — regression on `profit_margin` and classification of profitable vs. loss-making tasks — which will allow threshold detection via SHAP values and partial dependence plots rather than bin-level aggregations, providing finer-grained and more actionable thresholds.


## AI Usage Log
 
We used Claude (Anthropic) as our primary AI assistant throughout the project. In line with Alkemy's tracking requirements, this section documents how and why we used it.
 
**What we did ourselves.** The analytical roadmap was defined entirely by the team: which questions to ask, which variables to investigate, which methodological choices to justify, and how to interpret results. All key decisions — the diagnostic-based criterion for `hours_spent` anomalies, the swap test for date inconsistencies, the hierarchical imputation strategy, the joint three-axis definition of AI value creation, and the interpretation of every finding — were reasoned through by the team before being implemented.
 
**What we used AI for.** We used Claude primarily for two things: writing clean, well-structured Python code for complex visualizations and multi-step analytical operations (heatmaps, LOWESS curves, bootstrap confidence bands, η² comparisons), and formatting outputs in Markdown. In both cases, we provided the logic, the variables, and the expected output; the AI translated that into working code or structured prose.
 
**How we prompted.** Our prompts were always problem-specific and grounded in prior results. We described the analytical goal, specified which variables and transformations to use, and asked for code or text that implemented a decision we had already made. When AI output was incorrect or inconsistent with the data, we identified the error and asked for a correction with an explicit explanation of what was wrong. We never used AI to generate analytical decisions or interpret results on our behalf.
 
**What this means for reproducibility.** Every piece of code in the notebook reflects a design choice made by the team. The AI was a tool for implementation, not a substitute for reasoning.
 




