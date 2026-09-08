---
title: Asymptotic Theory
tags: [econometrics, asymptotics, consistency, delta-method, theory]
---

# Asymptotic Theory

Source: Hansen, *Econometrics*, chapters 6-8; *Probability and Statistics for Economists*, chapters 7-9.

## Why asymptotics at all

For almost any estimator beyond the sample mean, the exact finite-sample distribution is unknown or intractable. Asymptotic theory asks a different question: *what does the distribution look like as the sample grows without bound?* The answers are simple (usually normal) and turn out to be good approximations at realistic sample sizes.

Asymptotics is a tool for approximation, not a claim that your sample is infinite. When someone says "the estimator is asymptotically normal", read it as: "for large enough `n`, treating it as normal gives roughly correct confidence intervals." How large is "large enough" is an empirical question, usually answered with Monte Carlo simulation.

## Consistency

`θ̂ →p θ` — with enough data, the estimator lands arbitrarily close to the truth.

The recipe for proving consistency of an estimator built from sample moments:

1. Write the estimator as a continuous function of sample averages.
2. Apply the WLLN: each sample average converges to its population counterpart.
3. Apply the continuous mapping theorem: the function of the averages converges to the function of the limits.

For OLS:
```
β̂ = ( (1/n)Σ XᵢXᵢ' )⁻¹ ( (1/n)Σ XᵢYᵢ )  →p  E[XX']⁻¹E[XY] = β
```
Requirements: finite second moments, `E[XX']` invertible, and `E[Xe] = 0`. Note what is **not** required: homoskedasticity, normality, or a correctly specified CEF. OLS is consistent for the *linear projection coefficient* under very weak conditions. Whether that projection coefficient is what you want is a separate question — see [[04 - Conditional Expectation and Projection]].

**Inconsistency** is the serious failure. If `E[Xe] ≠ 0` (endogeneity), `β̂ →p β + E[XX']⁻¹E[Xe] ≠ β`. More data makes you more confident about the wrong number. No amount of clever standard errors fixes it; you need a different identification strategy. See [[11 - Instrumental Variables]].

## Asymptotic normality

For OLS:
```
√n(β̂ - β) →d N(0, V)      V = Q⁻¹ Ω Q⁻¹
```
with `Q = E[XX']` and `Ω = E[XX'e²]`. The sandwich again. Under homoskedasticity `Ω = Qσ²` and `V = Q⁻¹σ²`.

Proof sketch: write `√n(β̂ - β) = ((1/n)ΣXᵢXᵢ')⁻¹ ((1/√n)ΣXᵢeᵢ)`. The first factor →p `Q⁻¹` by the WLLN; the second →d `N(0, Ω)` by the CLT. Slutsky's theorem combines them.

Requirements: finite fourth moments of the regressors (needed for `Ω` to exist), and i.i.d. or otherwise well-behaved sampling.

**Practical reading**: the `√n` scaling means the estimator's standard deviation shrinks like `1/√n`. To halve a confidence interval you need four times the data.

## The delta method

You estimated `β̂`. You want `θ = g(β)` — an elasticity, a ratio, a peak of a quadratic, a marginal effect. How precise is `g(β̂)`?

**Delta method**: if `√n(β̂ - β) →d N(0, V)` and `g` is continuously differentiable with gradient `G = ∂g/∂β`, then

```
√n(g(β̂) - g(β)) →d N(0, G'VG)
```

so the standard error of `g(β̂)` is `√(Ĝ'V̂Ĝ / n)`.

Intuition: locally, a smooth function looks linear; a linear function of a normal is normal.

*Example.* The peak of a wage-experience profile `log(wage) = β₁·exp + β₂·exp²/100 + ...` occurs at `θ = -50β₁/β₂`. The delta method gives a standard error for `θ̂` from the covariance matrix of `(β̂₁, β̂₂)`.

**When the delta method misleads.** It is a first-order approximation and can be poor when:

- `g` is strongly nonlinear near `β`.
- The denominator of a ratio is close to zero — this is the killer. Ratios with imprecise denominators have heavy-tailed distributions that no normal approximation captures.

In the wage-peak example, the delta-method interval is `[29.8, 29.9]` while the test-inversion interval is `[29.1, 30.6]`. The delta interval is far too narrow. **Prefer test inversion or the bootstrap for nonlinear parameters.** See [[08 - Hypothesis Testing and Confidence Intervals]] and [[09 - Bootstrap and Resampling]].

## Consistent variance estimation

To use asymptotic normality you need `V̂ →p V`. For OLS this is exactly the HC estimators of [[06 - Standard Errors and Clustering]]. Consistency of `V̂` follows from the WLLN applied to `(1/n)ΣXᵢXᵢ'êᵢ²`, plus the fact that residuals converge to errors.

Slutsky then delivers the t-statistic result:
```
T = (β̂ⱼ - βⱼ)/s(β̂ⱼ) →d N(0, 1)
```

which is why the 1.96 critical value works regardless of the underlying distribution of the data.

## Rates of convergence

Not everything converges at `√n`.

| Estimator | Rate |
| --- | --- |
| Sample mean, OLS, MLE, GMM | `√n` (parametric rate) |
| Kernel regression at a point | `√(nh)`, slower; `h` is bandwidth |
| Nonparametric density estimation | slower still, worse in higher dimensions |
| Unit root / near-integrated time series coefficient | `n` (superconsistent) |
| Estimators at a boundary | nonstandard, often non-normal |

Slower rates are the formal statement of the **curse of dimensionality**: to estimate a flexible function of many variables you need exponentially more data. It is the reason applied work uses parametric approximations even when it does not believe them. See [[16 - Nonparametrics, Quantiles, and RDD]].

## When asymptotic normality fails

These are the cases where standard software output is actively misleading:

- **Parameter on a boundary** (e.g. testing a variance equals zero). The limiting distribution is a mixture, not normal.
- **Weak identification** (weak instruments). The limit is non-normal and the estimator is inconsistent. See [[11 - Instrumental Variables]].
- **Unidentified parameters** (a nuisance parameter present only under the alternative). Nonstandard limits.
- **Infinite variance** in the data. CLT does not apply.
- **Post-model-selection estimation**. If you chose the model using the same data, the "estimator" is a discontinuous function of the data and its distribution is not the naive one. Confidence intervals after stepwise selection or lasso have coverage far below nominal. See [[18 - Model Selection and Machine Learning]].

The last one is the most common in modern practice and the least often acknowledged.

## Monte Carlo simulation

The practical way to check whether asymptotic approximations work at your sample size:

1. Choose a data generating process (DGP) with known parameters.
2. Draw a sample of size `n`, compute your estimator and test statistic.
3. Repeat `B` times.
4. Summarize: bias, MSE, actual rejection rate of a nominal 5% test, actual coverage of a nominal 95% interval.

Because a Monte Carlo result is itself an estimate from a random sample of size `B`, it has its own standard error. For a rejection probability near 5%, `s(p̂) ≈ 0.22/√B` — so `B = 100` gives `±0.022`, `B = 1000` gives `±0.007`, `B = 10,000` gives `±0.003`. Hansen's rule of thumb: **use `B = 10,000` whenever feasible**, and never conclude that two simulations "disagree" without computing the standard error of the difference.

Related:

- [[02 - Probability Foundations]]
- [[05 - Least Squares Mechanics]]
- [[06 - Standard Errors and Clustering]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[09 - Bootstrap and Resampling]]
