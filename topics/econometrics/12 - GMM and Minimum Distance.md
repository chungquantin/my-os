---
title: GMM and Minimum Distance
tags: [econometrics, gmm, moments, estimation, theory]
---

# GMM and Minimum Distance

Source: Hansen, *Econometrics*, chapters 8 and 13.

## The unifying idea

Nearly every estimator in econometrics can be written as: *choose the parameter that makes some sample average equal (or as close as possible) to zero.*

Population statement — a **moment condition**:
```
E[ g(Wᵢ, β) ] = 0    holds at the true β
```

Sample analog:
```
ḡ(β) = (1/n) Σ g(Wᵢ, β)
```

If there are as many moment conditions as parameters (**just identified**), solve `ḡ(β̂) = 0` exactly. If there are more (**overidentified**), you cannot set them all to zero, so minimize a quadratic form:

```
J(β) = n · ḡ(β)' W ḡ(β)
```

for a positive definite weight matrix `W`. This is **generalized method of moments (GMM)**.

## Examples in this language

| Estimator | Moment condition |
| --- | --- |
| Sample mean | `E[X - μ] = 0` |
| OLS | `E[X(Y - X'β)] = 0` |
| IV / 2SLS | `E[Z(Y - X'β)] = 0` |
| MLE | `E[∂ log f(X\|θ)/∂θ] = 0` (the score) |
| Nonlinear least squares | `E[(∂m/∂β)(Y - m(X,β))] = 0` |
| Euler equation models | `E[Z(β(1+r)u'(c₁)/u'(c₀) - 1)] = 0` |

Seeing that these are one framework is genuinely useful: one asymptotic theory, one set of test statistics, one set of standard error formulas covers all of them.

## Asymptotics

Under regularity conditions:
```
√n(β̂ - β) →d N(0, V)
V = (G'WG)⁻¹ (G'WΩWG) (G'WG)⁻¹
```
where `G = E[∂g/∂β']` (how sensitive the moments are to the parameter) and `Ω = E[gg']` (the variance of the moments). The sandwich once again.

## Efficient GMM

The variance is minimized by setting `W = Ω⁻¹`, giving

```
V = (G'Ω⁻¹G)⁻¹
```

Intuition: weight each moment condition inversely to how noisy it is, and account for correlation between moments. Noisy moments get less say.

`Ω` is unknown, so **two-step GMM**:

1. Estimate `β̂₁` with any reasonable weight matrix (often `W = I` or `(Z'Z)⁻¹`).
2. Estimate `Ω̂` from the residual moments at `β̂₁`.
3. Re-estimate with `W = Ω̂⁻¹`.

Variants: **iterated GMM** (repeat to convergence) and **continuously updated GMM** (CUE, where `Ω(β)` is updated inside the minimization). CUE has better finite-sample centering but a nastier objective function to optimize.

**Under homoskedasticity, efficient GMM applied to the IV moment conditions is exactly 2SLS.** Under heteroskedasticity, efficient GMM improves on 2SLS. That is the precise relationship between the two.

**Practical warning**: two-step efficient GMM can behave poorly in finite samples with many moment conditions, because `Ω̂` is estimated with error and the estimator "overfits" the weight matrix. With many moments, one-step GMM or CUE is often better behaved. This is the same family of problems as many instruments in [[11 - Instrumental Variables]].

## The J test of overidentifying restrictions

The minimized objective is itself a test statistic:

```
J = n · ḡ(β̂)' Ω̂⁻¹ ḡ(β̂)  →d  χ²_{ℓ-k}
```

under the null that all moment conditions hold. `ℓ - k` is the number of surplus conditions.

Same interpretation caveats as the Sargan test: rejection means *something* is wrong without saying what; non-rejection is weak evidence because power is low and common flaws are invisible to it.

## Minimum distance

A closely related construction. Suppose you have an unrestricted estimator `β̂` with covariance `V̂`, and a theory implies `β = r(θ)` for a lower-dimensional `θ`. Minimize

```
J(θ) = n (β̂ - r(θ))' W (β̂ - r(θ))
```

Efficient choice is `W = V̂⁻¹`. The minimized value is a test of the restriction, `→d χ²_q`.

This is a general way to impose theoretical restrictions on an estimated model, and it is the population idea behind restricted least squares and behind indirect inference.

## Criterion-based testing, generally

Given any criterion `J(β)` minimized for estimation, the statistic

```
J = min_{β ∈ B₀} J(β) - min_{β ∈ B} J(β)
```

tests the restriction `β ∈ B₀`. Names by context: **likelihood ratio** (MLE), **minimum distance** (least squares), **distance statistic** (GMM). Under the null, `→d χ²_q`.

Compared to Wald tests, criterion-based tests are **invariant to how the restriction is written**. Wald statistics are not: testing `β₁ = β₂` versus `β₁/β₂ = 1` gives different answers, sometimes very different. That is a real defect of Wald tests for nonlinear hypotheses, and a reason to prefer criterion-based or bootstrap tests there. See [[08 - Hypothesis Testing and Confidence Intervals]].

## Empirical likelihood

An alternative to GMM: instead of a quadratic form, choose probability weights on the observations to maximize a nonparametric likelihood subject to the moment conditions holding exactly. Properties:

- Same first-order asymptotics as efficient GMM.
- Better higher-order properties — smaller bias, better-behaved tests.
- Invariant to reparameterization of the moments.
- Computationally harder; can fail to have a solution when the moment conditions are far from compatible with the data.

Related variants: exponential tilting, continuously updated GMM, generalized empirical likelihood.

## When to reach for GMM

- Your model implies moment conditions but not a full likelihood (asset pricing Euler equations, rational expectations models).
- You have more instruments than endogenous variables and heteroskedasticity.
- You want a unified framework for testing restrictions across a system of equations.
- Dynamic panel estimators (Arellano-Bond) — these are GMM with lagged levels as instruments for differences. See [[13 - Panel Data]].

**When not to**: if a simple just-identified IV or OLS answers your question, GMM adds machinery and finite-sample fragility without adding credibility. The identification argument, not the estimator's efficiency, is what makes an empirical paper convincing.

Related:

- [[03 - Statistical Inference Foundations]]
- [[07 - Asymptotic Theory]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[11 - Instrumental Variables]]
- [[13 - Panel Data]]
