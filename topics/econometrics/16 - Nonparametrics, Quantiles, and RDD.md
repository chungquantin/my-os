---
title: Nonparametrics, Quantiles, and RDD
tags: [econometrics, nonparametric, kernel, quantile-regression, rdd, causal-inference]
---

# Nonparametrics, Quantiles, and RDD

Source: Hansen, *Econometrics*, chapters 19-24.

Three related topics: estimating relationships without assuming a functional form, estimating effects on distributions rather than means, and one design (regression discontinuity) that is built on local nonparametric estimation.

## Why go nonparametric

A linear regression assumes the CEF is (approximately) a line. Nonparametric methods let the data determine the shape. The price is precision: you converge more slowly and you need more data, especially in higher dimensions.

## Kernel regression

To estimate `m(x) = E[Y | X = x]` at a point `x`, take a weighted average of nearby `Y` values:

**Nadaraya-Watson (local constant)**:
```
m̂(x) = Σᵢ K((Xᵢ - x)/h) Yᵢ  /  Σᵢ K((Xᵢ - x)/h)
```

- `K` is a **kernel** — a weight function that is large near zero and decays. Gaussian, Epanechnikov, uniform. The choice barely matters.
- `h` is the **bandwidth** — the width of the neighborhood. The choice matters enormously.

**Local linear regression** fits a weighted line rather than a weighted average in each neighborhood. It is strictly better and should be the default: it has smaller bias, especially at the edges of the data, where the local-constant estimator is badly biased because the neighborhood is one-sided. **Boundary bias is precisely why RDD uses local linear estimation.**

### The bandwidth is the whole ballgame

- Small `h`: few observations per neighborhood → low bias, high variance. Wiggly, overfit.
- Large `h`: many observations → low variance, high bias. Smooth, oversmoothed, approaching a global fit.

This is the bias-variance tradeoff made visible. The MSE-optimal bandwidth balances squared bias against variance and shrinks like `n^{-1/5}` for local linear estimation, giving a convergence rate of `n^{-2/5}` — slower than the parametric `n^{-1/2}`.

Selection methods: cross-validation (minimize leave-one-out prediction error), plug-in rules (estimate the unknown curvature that appears in the optimal formula), and Hansen's discussion of the fact that the MSE-optimal bandwidth is *not* the right bandwidth for constructing confidence intervals — at the optimal bandwidth the bias is of the same order as the standard error, so intervals are miscentered. The standard fixes are **undersmoothing** (use a smaller bandwidth than optimal) or **explicit bias correction** with a robust variance, which is what modern RDD packages implement.

### The curse of dimensionality

With `d` regressors the optimal rate degrades to `n^{-2/(4+d)}`. With `d = 5`, achieving the accuracy you would get from 100 observations in one dimension requires astronomically more data. Neighborhoods in high dimensions are almost empty.

This is why fully nonparametric methods are used mainly with one or two continuous variables, and why applied work relies on **semiparametric** models — flexible in the dimension you care about, parametric in the rest. The partially linear model `Y = D'θ + g(X) + e` is the canonical example, and it is exactly the structure that double machine learning exploits. See [[18 - Model Selection and Machine Learning]].

## Series and sieve regression

An alternative to kernels: approximate `m(x)` by a linear combination of basis functions and estimate by OLS.

Bases: polynomials, **splines** (piecewise polynomials joined smoothly at knots), B-splines, wavelets, Fourier terms.

```
m(x) ≈ β₀ + β₁ψ₁(x) + ... + βₖψₖ(x)
```

The number of terms `K` plays the role of the bandwidth: more terms means less bias and more variance. Choose by cross-validation or an information criterion.

Advantages over kernels: it is just OLS, so all the standard regression machinery (F tests, robust standard errors, fixed effects) applies directly; it handles multiple regressors more gracefully; and it is easy to impose additivity.

Practical advice: **splines beat high-order global polynomials.** A degree-10 polynomial oscillates wildly at the edges of the data (Runge's phenomenon). This matters concretely in RDD, where global high-order polynomial fits have been shown to produce spurious discontinuities.

## Quantile regression

Least squares estimates the conditional *mean*. Quantile regression estimates the conditional **quantile**:

```
Q_τ(Y | X) = X'β(τ)
```

Estimated by minimizing the asymmetrically weighted absolute loss (the "check function"):
```
min_b Σᵢ ρ_τ(Yᵢ - Xᵢ'b),    ρ_τ(u) = u(τ - 1{u < 0})
```

At `τ = 0.5` this is **median regression** (least absolute deviations).

Why it is useful:

- **Distributional effects.** A policy may not move the mean while sharply compressing the lower tail. Only quantile regression sees that.
- **Robustness.** Median regression is far less sensitive to outliers than the mean.
- **Heterogeneity.** `β(τ)` varying with `τ` is direct evidence that the effect is not uniform across the distribution.
- **Risk applications.** Value-at-Risk is a conditional quantile by definition.

Details: the objective is convex but non-differentiable, solved by linear programming. Asymptotic variance involves the conditional density of the error at the quantile, which is hard to estimate — so the **bootstrap** is the standard route to standard errors. See [[09 - Bootstrap and Resampling]].

**Interpretation warning**: `β(τ)` describes how the `τ`-quantile of the conditional distribution shifts with `X`. It does **not** track the same individuals across `τ` — the person at the 10th percentile with low `X` may not be the person at the 10th percentile with high `X`. "Quantile treatment effects" have a causal reading only under additional assumptions (rank invariance, or an instrument in the IV quantile framework).

## Regression discontinuity design (RDD)

The most credible non-experimental design in applied economics.

**Setup**: treatment is assigned by a rule based on a **running variable** `X` crossing a **cutoff** `c`:
```
D = 1{X ≥ c}
```

Examples: a scholarship for test scores above a threshold; a program for firms below an employee count; a class-size rule that triggers a split at 40 students; a regulatory tick-size regime that changes above a price level.

**The identifying idea**: units just below and just above the cutoff are nearly identical in everything except treatment. Compare them.

**Identifying assumption**: all other determinants of the outcome are **continuous** at `c`. Then any jump in `Y` at `c` must be caused by the treatment.

```
τ = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]
```

This is a **local** effect: the ATE for units at the cutoff. It says nothing about units far from it. That is the fundamental limitation of RDD — high internal validity, narrow external validity.

**Sharp vs fuzzy**:

- **Sharp**: crossing the cutoff determines treatment perfectly. The jump in `Y` is the effect.
- **Fuzzy**: crossing changes the *probability* of treatment. Then divide the jump in `Y` by the jump in treatment probability — this is exactly IV with `1{X ≥ c}` as the instrument, and it delivers a LATE for compliers at the cutoff. See [[11 - Instrumental Variables]].

### Estimation

Local linear regression on each side of the cutoff, within a bandwidth `h`, with a triangular kernel. Modern practice:

- Use the MSE-optimal bandwidth (Imbens-Kalyanaraman / Calonico-Cattaneo-Titiunik).
- Use **bias-corrected estimates with robust confidence intervals** (CCT), because at the optimal bandwidth the bias is not negligible.
- **Do not use global high-order polynomials.** Gelman and Imbens showed they produce noisy, unreliable estimates and spurious discontinuities.
- Software: `rdrobust` (Stata, R, Python).

### Validity checks

These are what referees look for, and they are genuinely informative:

1. **McCrary density test.** Is the density of the running variable smooth at the cutoff? A jump indicates **manipulation** — units sorting themselves across the threshold. If people can precisely control their test score, the design is dead.
2. **Covariate balance.** Pre-determined characteristics should show no discontinuity at `c`. Run the RDD with each covariate as the outcome; nothing should jump.
3. **Placebo cutoffs.** Run the analysis at fake thresholds. You should find nothing.
4. **Bandwidth sensitivity.** Plot the estimate against bandwidth. A result that only appears in a narrow window is suspicious.
5. **Donut hole.** Drop observations immediately at the cutoff (where manipulation concentrates) and re-estimate.
6. **Plot the data.** A binned scatter of `Y` against `X` with the local fits overlaid. If a reader cannot see the jump, the estimate is doing the work rather than the design.

### Regression kink design

A variant: instead of a jump in the level, the *slope* of treatment intensity changes at the threshold (common with benefit formulas). Identification comes from the kink, and the same local-polynomial machinery applies with one extra derivative.

## Other semiparametric models worth knowing

- **Partially linear model** `Y = D'θ + g(X) + e` — parametric in the variable of interest, nonparametric in controls. Robinson's estimator partials out `g(X)` nonparametrically; the lasso version is the partialling-out estimator of [[18 - Model Selection and Machine Learning]].
- **Single index model** `Y = g(X'β) + e` — the index is linear, the link is free.
- **Additive models** `Y = g₁(X₁) + ... + g_d(X_d) + e` — sidesteps the curse of dimensionality by ruling out interactions.
- **Propensity score methods** — match or weight on `P(D = 1 | X)` to estimate treatment effects under the CIA. Inverse probability weighting, matching, and doubly-robust estimators. All rely on the CIA plus overlap; check the overlap explicitly.

Related:

- [[10 - Causality and Identification]]
- [[11 - Instrumental Variables]]
- [[09 - Bootstrap and Resampling]]
- [[18 - Model Selection and Machine Learning]]
