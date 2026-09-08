---
title: Standard Errors and Clustering
tags: [econometrics, inference, standard-errors, robust, clustering, beginner]
---

# Standard Errors and Clustering

Source: Hansen, *Econometrics*, chapter 4 (sections 4.12-4.23) and chapter 7.

If you only read one applied note, read this one. Coefficient estimates are usually fine; standard errors are where applied work quietly goes wrong.

## What a standard error is

A **standard error** `s(β̂)` is an estimate of the standard deviation of the sampling distribution of `β̂`. It answers: *if I re-drew the sample, how much would this estimate bounce around?*

It is an estimate, so it is itself random and can be wrong. When people say "the standard errors are wrong", they mean the formula used does not match how the data were actually generated.

## The general variance formula

For OLS, conditional on `X`:

```
V = (X'X)⁻¹ (X'DX) (X'X)⁻¹        where D = diag(σ₁², ..., σₙ²)
```

This "sandwich" — bread, meat, bread — is the master formula. Every standard error method is a different way of estimating the meat.

## Homoskedastic standard errors (the default, usually wrong)

**Homoskedasticity** means `E[e² | X] = σ²`: the error variance is the same for every observation. Then `D = σ²I` and the sandwich collapses:

```
V⁰ = (X'X)⁻¹ σ²      estimated by  V̂⁰ = (X'X)⁻¹ s²
```

This is the classical formula, and it is the **default in most software including Stata's `regress` and R's `lm`**. It is only valid under homoskedasticity.

**How wrong can it be?** Hansen's calculation: with a single regressor and `σᵢ² = Xᵢ²`, the ratio of the true variance to the expected classical variance is the standardized fourth moment `κ = E[X⁴]/(E[X²])²`. For normal `X`, `κ = 3` — the true variance is *three times* the classical estimate. For the `wage` variable in the CPS data, `κ ≈ 30`. That is a standard error understated by a factor of `√30 ≈ 5.5`.

Heteroskedasticity is the norm, not the exception, in cross-sectional economic data. Variance in wages grows with education. Variance in firm outcomes grows with firm size. Variance in spreads grows with volatility.

## Heteroskedasticity-robust (HC) standard errors

Replace the unknown `σᵢ²` with the squared residual `êᵢ²`. Four variants, differing only in a small-sample correction:

```
HC0:  (X'X)⁻¹ ( Σ XᵢXᵢ' êᵢ² ) (X'X)⁻¹
HC1:  (n/(n-k)) × HC0
HC2:  (X'X)⁻¹ ( Σ XᵢXᵢ' êᵢ²/(1-hᵢᵢ) ) (X'X)⁻¹
HC3:  (X'X)⁻¹ ( Σ XᵢXᵢ' êᵢ²/(1-hᵢᵢ)² ) (X'X)⁻¹
```

where `hᵢᵢ` is leverage. Collectively these are called **robust**, **heteroskedasticity-consistent**, or **Eicker-White** standard errors.

Properties:

- Ordering: `HC0 < HC2 < HC3` (as matrices). HC0 is biased toward zero; HC3 is conservative — in expectation weakly larger than the true variance for any `X`.
- HC2 is exactly unbiased under homoskedasticity.
- HC3 comes from the jackknife / leave-one-out principle.

**Hansen's recommendation**: HC2 or HC3 are preferred; HC1 is the most common in practice (it is what Stata's `, r` option gives). In most applications all three are similar and the choice does not matter. They diverge substantially exactly when some observation has leverage `hᵢᵢ` near 1 — because `1/(1-hᵢᵢ)` and `1/(1-hᵢᵢ)²` blow up. That is precisely the situation where you should worry.

**Practical rule**: use robust standard errors by default in cross-sectional work. Do not use them as a substitute for thinking about clustering.

## The sparse dummy problem

A special failure case worth knowing. If a regressor is a dummy that equals 1 for very few observations, robust standard errors are badly biased downward — in the extreme case of one treated observation, the residual for that observation is exactly zero, so the robust variance for that coefficient is estimated as (nearly) zero. You get a spuriously precise-looking estimate of something you actually know almost nothing about.

This is not a corner case: it is the same algebra as difference-in-differences with one treated state. See [[14 - Difference in Differences]].

## Clustered standard errors

**The setup.** Observations arrive in groups: students within schools, workers within firms, repeated observations of the same individual, all trades in one day. Within a group, errors are correlated; across groups, independent.

Notation: `g = 1,...,G` indexes clusters, `nᵍ` is the size of cluster `g`, `n = Σ nᵍ`.

**Why it matters — the Moulton problem.** Suppose all clusters have `N` observations, within-cluster error correlation is `ρ`, and the regressor is constant within cluster (like a school-level policy). Then the exact variance is

```
V = (X'X)⁻¹ σ² (1 + ρ(N - 1))
```

The classical formula is off by the factor `1 + ρ(N-1)`. Hansen's example: in the Kenyan schooling data (Duflo-Dupas-Kremer 2011), average cluster size is 48. With `ρ = 0.25`, the variance is understated by a factor of about **twelve**, so standard errors are understated by a factor of about **three**.

Notice: even a small `ρ` produces a large inflation when `N` is large. This is why panel and grouped data demand clustering.

**The estimator (Arellano / Liang-Zeger):**

```
Ω̂ = Σ_g ( Σ_i Xᵢg êᵢg ) ( Σ_ℓ X_ℓg ê_ℓg )'
V̂ = a_n (X'X)⁻¹ Ω̂ (X'X)⁻¹
```

The key structure: sum the `X·ê` products *within* each cluster first, then take the outer product across clusters. Stata's finite-sample adjustment is `a_n = ((n-1)/(n-k))·(G/(G-1))`.

There is also a CR3 variant, analogous to HC3, built from cluster-level leave-one-out prediction errors. It is conservative and more expensive to compute.

**Empirical illustration.** The Kenyan tracking regression:

```
TestScore_ig = -0.071 + 0.138·Tracking_g + e_ig
                (0.019)   (0.026)          conventional robust SE
                [0.054]   [0.078]          clustered by school
```

Clustered standard errors are about **three times** the robust ones. Any conclusion about statistical significance depends entirely on this choice.

## How to think about clustered inference

Hansen is unusually blunt here, and these points are the ones people miss:

**1. The effective sample size is `G`, not `n`.** The cluster-robust estimator treats each cluster as one observation and estimates the variance from variation across cluster means. If you have `G = 50` clusters, your inference is (at best) as reliable as heteroskedasticity-robust inference with 50 observations — regardless of having 500,000 rows.

**2. Small `G` is common and treacherous.** Clustering by U.S. state means `G ≈ 50`. If you also estimate 20 coefficients, the covariance matrix is estimated very imprecisely. With `G = 20`, treat it as a very small sample. If you estimate more than `G` coefficients, the clustered covariance matrix is not even full rank.

**3. Heterogeneous cluster sizes make it worse.** Most cluster-robust theory assumes equal cluster sizes. When sizes vary a lot (as they do across U.S. states), clustered inference behaves like heteroskedasticity-robust inference with extremely heteroskedastic observations. Cluster sums have variance proportional to cluster size.

**4. Few treated clusters is the worst case.** If treatment is assigned at the cluster level and only a handful of clusters are treated — in the extreme, one — the reported standard error can be dramatically too small while the estimate is dramatically imprecise. Algebraically identical to the sparse-dummy problem above.

## At what level should you cluster?

There is no formal answer, and Hansen says so emphatically. What is known:

- **Cluster too finely** (households instead of villages): variance estimators are biased because you omit correlation terms. Since correlations are typically positive, standard errors are **too small** and you get spurious significance.
- **Cluster too coarsely** (states instead of villages): no bias, but many extra terms to estimate, so the variance estimator is noisy. Standard errors are **imprecise** — more random than necessary.

So it is a bias-variance tradeoff in estimating the variance itself, with no clean rule.

Practical guidance:

- Cluster at the level at which treatment is assigned. If the policy varies by state, cluster by state.
- Cluster at the level at which sampling was done. If the survey sampled villages, cluster by village.
- Report `G` explicitly. A reader cannot assess your inference without it.
- Be suspicious of the reasoning "clustering changed the standard errors a lot, therefore clustering is important". As Hansen notes, the change could be bias reduction (good) or added sampling noise (bad), and you generally cannot tell which.
- With small `G`, consider the wild cluster bootstrap. See [[09 - Bootstrap and Resampling]].

In the Kenyan example, clustering at school level (`G = 111`, cluster sizes 19-62) is a well-balanced choice. Clustering at the "zone" level would give `G = 9` — far too few.

## Serial correlation in time series

The time-series analogue is **HAC** (heteroskedasticity and autocorrelation consistent) standard errors — Newey-West and friends. Same sandwich, meat built from a weighted sum of autocovariances up to a bandwidth. See [[15 - Time Series]].

## Decision table

| Data structure | Use |
| --- | --- |
| i.i.d. cross-section | HC2 or HC3 robust |
| Grouped / repeated observations of a unit | Cluster-robust at group level |
| Panel (units over time) | Cluster by unit |
| Treatment assigned at group level | Cluster by that group level |
| Two dimensions of dependence (e.g. firm and year) | Two-way clustering |
| Time series | HAC (Newey-West) |
| Few clusters (`G` < 30-40) | Wild cluster bootstrap; report `G` |
| Non-i.i.d. structure you cannot characterize | Say so; do not pretend |

## Reporting

Always state, in the table notes: what standard errors are used, at what level clustering is applied, and how many clusters there are. Papers that omit this are unassessable.

Related:

- [[05 - Least Squares Mechanics]]
- [[07 - Asymptotic Theory]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[09 - Bootstrap and Resampling]]
- [[13 - Panel Data]]
- [[14 - Difference in Differences]]
