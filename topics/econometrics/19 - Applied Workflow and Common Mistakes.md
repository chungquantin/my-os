---
title: Applied Workflow and Common Mistakes
tags: [econometrics, workflow, reproducibility, best-practices, checklist]
---

# Applied Workflow and Common Mistakes

A practical companion to the theory notes. Most of these points come directly from Hansen's applied advice scattered across *Econometrics*; the rest are the standard hard-won lessons of applied work.

## The workflow

### 1. Write down the question before touching the data

State: the outcome, the treatment or predictor of interest, the population, and whether you are doing description, prediction, or causal inference. Write the estimand in one sentence. If you cannot, you are not ready to estimate anything.

### 2. Understand the data before modeling

- How was it collected? Sampling scheme determines your inference method more than anything else.
- What is the unit of observation, and what is the unit of treatment assignment? These differ more often than people notice, and the gap is where clustering decisions live.
- Summary statistics for every variable: mean, SD, min, max, count of missing. Look for impossible values (negative wages, ages of 999, timestamps in 1970).
- Plot everything. Histograms of each variable, scatter of `Y` against the key `X`, time series if there is a time dimension. A surprisingly large fraction of empirical mistakes are visible in a plot and invisible in a regression table.

### 3. Handle missing data deliberately

The default in every package is listwise deletion — silently dropping rows. Consequences:

- Your sample changes when you add a variable, so coefficient changes across specifications mix "controlling for X" with "using a different sample". Always report `N` for every column of every table.
- If missingness is related to the outcome, dropping induces selection bias. See [[17 - Limited Dependent Variables]].

Report the missingness pattern. Consider whether a missing-indicator approach, multiple imputation, or an explicit selection model is warranted.

### 4. Specify, do not search

Decide the specification from theory and the design. If you must explore, say so, and separate confirmatory from exploratory results. Silent specification search is the leading cause of results that do not replicate.

### 5. Choose the right standard errors

The single most common technical error in applied work. Defaults are almost always wrong. See [[06 - Standard Errors and Clustering]].

### 6. Report properly

- Point estimate, standard error, confidence interval.
- `N`, and `G` (number of clusters) if clustered.
- Which standard errors and at what level.
- Interpretation in economic units.
- No asterisks. Hansen: report p-values, avoid stars.

### 7. Check robustness honestly

Report the main specification plus a small number of pre-committed variants. A table of thirty specifications is not robustness; it is a specification search with the search hidden.

## Check your code

Hansen devotes a whole section to this under the heading "Check Your Code".

The case: Donohue and Levitt (2001) argued that legalized abortion reduced crime, with `β̂ = -0.028` implying a 15-25% crime reduction. Foote and Goetz (2008), replicating, found the code had **inadvertently omitted the state-year interaction terms** the paper claimed to include. Corrected, `β̂ = -0.010` — the substantive effect cut to roughly a third.

Hansen's conclusion is worth taking literally:

> Computation errors are pervasive in applied economic work. It is very easy to make errors; it is very difficult to clean them out of lengthy code. Errors in most papers are ignored as the details receive minor attention. Important and influential papers, however, are scrutinized. The solution is to be pro-active and vigilant.

Practical defenses:

- **Verify on a case you can compute by hand.** Card-Krueger's DiD estimate should equal the difference of the four cell means. If your regression does not reproduce the table of means, something is wrong.
- **Assert invariants**: row counts after merges, sums that should match, ranges of key variables. Fail loudly.
- **Simulate**: generate data from a known DGP where you know the answer, and check your code recovers it. This catches sign errors, index errors, and wrong-variance bugs that no amount of staring finds.
- **Reproduce a published result** with your pipeline before running your own analysis on the same data.
- **Version control everything.** Scripts, not point-and-click. A result you cannot regenerate from a script is not a result.
- **Set random seeds** for bootstrap, cross-validation, and sample splitting.
- **Have someone else read the code**, or re-implement the key regression independently.

## Common mistakes, catalogued

### Interpretation

| Mistake | Correction |
| --- | --- |
| "X causes Y" from a regression coefficient | Say "associated with", unless you have an identification argument |
| "No effect" from a non-significant coefficient | "Cannot reject zero; the interval is [a, b]" |
| Treating p as P(H₀ true) | It is P(data this extreme \| H₀) |
| Reporting stars, not magnitudes | Report the effect size in economic units |
| Reading probit/logit coefficients as marginal effects | Compute average marginal effects |
| Comparing probit coefficients across specifications | The scale changes; compare marginal effects |
| Interpreting a log-log coefficient as a level effect | It is an elasticity |
| Extrapolating outside the support of the data | The linear fit is a local approximation |

### Specification

| Mistake | Correction |
| --- | --- |
| Kitchen-sink controls | Distinguish confounders, mediators, colliders |
| Controlling for post-treatment variables | Drop them; they block the effect or induce bias |
| Adding controls until significance appears | This is p-hacking |
| Sample changing across specifications | Fix the sample; report `N` per column |
| Ignoring functional form | Check with a flexible specification or a plot |
| Global high-order polynomials in RDD | Use local linear |
| Bare TWFE with staggered adoption | Use a modern DiD estimator |

### Inference

| Mistake | Correction |
| --- | --- |
| Default (homoskedastic) standard errors | Use robust; the classical formula can be off by a factor of 30 in wage data |
| Not clustering grouped data | Cluster at the assignment/sampling level; the Moulton factor can be 12× |
| Clustering with very few clusters | Report `G`; use the wild cluster bootstrap |
| Ignoring multiple testing | Bonferroni or FDR; disclose how many tests |
| Delta method for a ratio | Use test inversion or the bootstrap |
| Bootstrap statistic centered at the null | Center at `θ̂` — this is the classic bootstrap bug |
| Bootstrap standard errors for just-identified 2SLS | The moments do not exist; use quantile-based intervals |
| Naive t-tests after lasso selection | Use double selection, partialling-out, or DML |
| Manual two-stage IV | Use a proper IV command; the manual SEs are wrong |

### Data

| Mistake | Correction |
| --- | --- |
| Silent listwise deletion | Report and examine missingness |
| Merge that silently duplicates rows | Assert row counts before and after |
| Look-ahead bias in time series features | All features must be computable at prediction time |
| Survivorship bias in the sample | Reconstruct the population as it was, not as it survived |
| Outliers driving the result | Check leverage `hᵢᵢ`; report with and without |
| Units and scaling errors | Sanity-check coefficient magnitudes against known facts |

## The reader's checklist

When evaluating someone else's empirical paper:

1. What is the estimand, and does the method deliver it?
2. Where does the identifying variation come from?
3. What must be true for the identification to hold, and what is the most plausible violation?
4. Is the sample the population they claim to speak about?
5. Are standard errors appropriate to the data structure? Is `G` reported?
6. How many specifications were run, and were they pre-committed?
7. Are the effect sizes economically meaningful, not just statistically distinguishable from zero?
8. Are code and data available?

## Reproducibility

- One script (or pipeline) that runs from raw data to final tables with no manual steps.
- Raw data never modified; all cleaning in code.
- Environment pinned (package versions).
- Seeds set and recorded.
- Every number in the paper traceable to a line of code.

Hansen notes that most journals require data and code availability but few require simulation code, and encourages posting it anyway: "This invites others to build on and use your results, leading to possible collaboration, citation, and/or advancement."

Related:

- [[06 - Standard Errors and Clustering]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[10 - Causality and Identification]]
- [[14 - Difference in Differences]]
- [[18 - Model Selection and Machine Learning]]
