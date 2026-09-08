---
title: Econometrics Hub
tags: [econometrics, statistics, causal-inference, hub, reading-map]
---

# Econometrics Hub

Econometrics is the craft of learning about the economy — or any decision problem — from data that you did not get to design. A chemist can randomize a lab experiment. An economist usually gets whatever the world produced, with all its self-selection, feedback loops, and missing variables. Econometrics is the set of tools for extracting honest answers anyway, and for being explicit about how much you actually know.

These notes summarize two books by Bruce Hansen:

- *Probability and Statistics for Economists* (Princeton, 2022) — the foundations: probability, distributions, estimation, testing.
- *Econometrics* (Princeton, 2022) — the applied and theoretical machinery: regression, causality, instruments, panels, time series, nonparametrics, machine learning.

They are written to be read in that order. These notes are written for someone starting from scratch: intuition first, formula second, worked example third, and a "what to actually do" section wherever it helps.

## The one idea to hold onto

Almost everything in these notes is a variation on one loop:

1. **Define the thing you want to know** as a feature of a *population* (a number that exists whether or not you sample: an average, a slope, a treatment effect). This is the **estimand**.
2. **Build a sample analog** of it from your data. This is the **estimator**.
3. **Ask how far off the estimator is likely to be.** This is the **sampling distribution**, and it gives you standard errors, confidence intervals, and tests.
4. **Ask whether the estimand is the thing you actually care about.** This is **identification** — the step where causal claims live or die.

Most beginner mistakes are skipping step 1 or step 4. Software will happily do steps 2 and 3 for you and print stars next to numbers that answer no useful question.

## How to read these notes

**Absolute beginner path** (you have algebra, some calculus, no statistics):

1. [[01 - What Econometrics Is]]
2. [[02 - Probability Foundations]]
3. [[03 - Statistical Inference Foundations]]
4. [[04 - Conditional Expectation and Projection]]
5. [[05 - Least Squares Mechanics]]
6. [[06 - Standard Errors and Clustering]]
7. [[08 - Hypothesis Testing and Confidence Intervals]]
8. [[19 - Applied Workflow and Common Mistakes]]

Keep [[98 - Formula Cheatsheet]] open alongside whichever path you take.

That is enough to read and criticize a regression table honestly.

**Causal inference path** (you can already run a regression):

1. [[10 - Causality and Identification]]
2. [[11 - Instrumental Variables]]
3. [[13 - Panel Data]]
4. [[14 - Difference in Differences]]
5. [[16 - Nonparametrics, Quantiles, and RDD]] (regression discontinuity section)

**Prediction and modern methods path**:

1. [[07 - Asymptotic Theory]]
2. [[09 - Bootstrap and Resampling]]
3. [[18 - Model Selection and Machine Learning]]
4. [[15 - Time Series]]

**Theory path** (you want to know why any of it works):

1. [[02 - Probability Foundations]]
2. [[07 - Asymptotic Theory]]
3. [[12 - GMM and Minimum Distance]]

## Note index

| Note | Covers |
| --- | --- |
| [[01 - What Econometrics Is]] | Populations vs samples, estimands, the shape of the whole subject |
| [[02 - Probability Foundations]] | Random variables, expectation, variance, key distributions, LLN, CLT |
| [[03 - Statistical Inference Foundations]] | Estimators, bias, variance, MSE, method of moments, maximum likelihood |
| [[04 - Conditional Expectation and Projection]] | The CEF, the best predictor, linear projection, omitted variable bias |
| [[05 - Least Squares Mechanics]] | OLS algebra, fitted values, residuals, R², leverage, Frisch-Waugh-Lovell |
| [[06 - Standard Errors and Clustering]] | Heteroskedasticity, HC0-HC3, cluster-robust errors, the Moulton problem |
| [[07 - Asymptotic Theory]] | Consistency, asymptotic normality, the delta method, why large n helps |
| [[08 - Hypothesis Testing and Confidence Intervals]] | t-tests, Wald tests, p-values, power, multiple testing, testing abuse |
| [[09 - Bootstrap and Resampling]] | Jackknife, bootstrap variants, percentile-t, when resampling fails |
| [[10 - Causality and Identification]] | Potential outcomes, ATE/ATT/LATE, selection bias, conditional independence |
| [[11 - Instrumental Variables]] | Endogeneity, 2SLS, LATE, weak and many instruments |
| [[12 - GMM and Minimum Distance]] | Moment conditions as a general estimation language, overidentification |
| [[13 - Panel Data]] | Pooled, random effects, fixed effects, first differences, dynamic panels |
| [[14 - Difference in Differences]] | Two-way fixed effects, parallel trends, event studies, inference traps |
| [[15 - Time Series]] | Stationarity, AR/MA, HAC errors, forecasting, unit roots, cointegration |
| [[16 - Nonparametrics, Quantiles, and RDD]] | Kernels, series regression, quantile regression, regression discontinuity |
| [[17 - Limited Dependent Variables]] | Probit/logit, count data, censoring, sample selection |
| [[18 - Model Selection and Machine Learning]] | AIC, cross-validation, ridge, lasso, double selection, double ML |
| [[19 - Applied Workflow and Common Mistakes]] | A practical checklist from data loading to reporting |
| [[98 - Formula Cheatsheet]] | Every formula in these notes, grouped by topic |
| [[99 - Econometrics Glossary]] | Terms and notation in one place |

## Notation used throughout

Econometrics notation is compact and it trips up beginners more than the ideas do. The conventions here follow Hansen:

- `Y` — the outcome (dependent variable) you are trying to explain or predict.
- `X` — the regressors (independent variables, covariates, controls). Usually a vector.
- `D` — a treatment variable, often binary (0/1).
- `Z` — an instrument.
- `e` or `ε` — the error term: whatever is in `Y` that `X` does not explain.
- `β` — a population parameter (unknown, fixed, what you want).
- `β̂` — an estimate of it from the sample (known, random, what you have).
- `E[Y]` — the population mean of `Y`; `E[Y | X]` — the mean of `Y` among units with that value of `X`.
- `n` — sample size; subscript `i` indexes an observation.
- `X'β` — a linear combination `β₁X₁ + β₂X₂ + ...`. The prime means transpose.
- `→p` — converges in probability (the estimator homes in on a number as `n` grows).
- `→d` — converges in distribution (the rescaled estimator's shape approaches a known distribution).

The hat is the single most important symbol: **hat means "computed from a sample, therefore random"**. Everything about standard errors flows from that.

Related:

- [[01 - What Econometrics Is]]
- [[19 - Applied Workflow and Common Mistakes]]
- [[98 - Formula Cheatsheet]]
- [[99 - Econometrics Glossary]]
