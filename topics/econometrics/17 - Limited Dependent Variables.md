---
title: Limited Dependent Variables
tags: [econometrics, probit, logit, tobit, selection, mle]
---

# Limited Dependent Variables

Source: Hansen, *Econometrics*, chapters 25-27.

When the outcome is not a continuous, unbounded number, linear regression may be awkward or invalid. This note covers binary, multinomial, count, censored, and selected outcomes.

## Binary outcomes

`Y ∈ {0, 1}`: employed or not, defaulted or not, order filled or not. The CEF is a probability:
```
E[Y | X] = P(Y = 1 | X)
```

### Linear probability model (LPM)

Just run OLS of `Y` on `X`. The fitted values estimate `P(Y = 1 | X)`.

Pros: coefficients are marginal effects on the probability, directly interpretable; extends trivially to fixed effects, IV, and DiD; robust to distributional assumptions since it is estimating the linear projection of the CEF.

Cons: fitted probabilities can fall outside `[0,1]`; the error is inherently heteroskedastic (variance `p(1-p)`, so **always use robust standard errors**); and the linear approximation is poor when many predicted probabilities are near 0 or 1.

Hansen's framing applies: the LPM estimates the best linear approximation to the CEF, which is a well-defined and often useful object. For estimating an average marginal effect in the middle of the distribution it is frequently fine, and its transparency in causal designs is a real advantage.

### Probit and logit

Model the probability through a link function:
```
P(Y = 1 | X) = G(X'β)
```
- **Probit**: `G = Φ`, the normal CDF.
- **Logit**: `G = Λ(u) = exp(u)/(1 + exp(u))`, the logistic CDF.

Both are estimated by maximum likelihood ([[03 - Statistical Inference Foundations]]). The log-likelihood is globally concave, so optimization is well behaved.

**Latent variable interpretation**: imagine `Y* = X'β + e` with `Y = 1{Y* > 0}`. Probit assumes `e` is standard normal, logit assumes it is logistic. This is how these models arise from utility maximization: `Y*` is the net utility of choosing 1.

**Interpreting coefficients — this is where beginners go wrong.** The coefficient `βⱼ` is *not* a marginal effect. The marginal effect is
```
∂P/∂Xⱼ = g(X'β) · βⱼ
```
which depends on where you evaluate it. Report either:

- **Average marginal effect (AME)**: average `g(Xᵢ'β̂)βⱼ` over the sample. Usually the right choice.
- **Marginal effect at the mean (MEM)**: evaluate at `X̄`. Less meaningful, since the "average person" may not exist.

Standard errors for marginal effects come from the delta method ([[07 - Asymptotic Theory]]).

For logit, `exp(βⱼ)` is an **odds ratio**. Odds ratios are not probabilities and are routinely misreported as such.

Probit and logit give nearly identical marginal effects in practice; logit coefficients are roughly 1.6-1.8 times probit coefficients. The choice rarely matters.

**Caution**: unlike OLS, probit/logit coefficients are not comparable across specifications or samples, because the scale is normalized by the (unidentified) error variance. Adding a control changes the scale, so a coefficient change does not have the OVB interpretation it has in a linear model. Compare **marginal effects**, not coefficients.

## Multinomial and ordered outcomes

**Multinomial logit** for unordered choices (which mode of transport, which product). Relies on the **independence of irrelevant alternatives (IIA)** — the classic "red bus / blue bus" problem: adding a near-identical alternative should not reallocate choice probabilities proportionally, but MNL forces it to. Test with Hausman-McFadden.

**Conditional logit (McFadden)**: the workhorse of discrete-choice demand estimation. Regressors are attributes of the alternatives, not just of the chooser.

**Nested logit** and **mixed / random-coefficients logit** relax IIA. Mixed logit is the modern standard and is estimated by simulated maximum likelihood.

**Ordered probit/logit** for ordered categories (credit ratings, survey scales). Estimates one coefficient vector plus a set of cutpoints. Do not run OLS on an ordinal scale as if the spacing between categories were meaningful — though for many purposes it is a reasonable approximation, and it is far easier to embed in a causal design.

## Count data

`Y ∈ {0, 1, 2, ...}`: number of patents, number of trades, number of doctor visits.

- **Poisson regression**: `E[Y | X] = exp(X'β)`. Coefficients are semi-elasticities — a one-unit change in `Xⱼ` multiplies the expected count by `exp(βⱼ)`.
- Poisson imposes `var = mean` (equidispersion), almost always violated. **Overdispersion** is the norm.
- **Fix**: use **quasi-maximum likelihood Poisson with robust standard errors**. This is consistent for `β` whenever the conditional mean is correctly specified, regardless of the true distribution. This is a genuinely underused result — Poisson QMLE is the right default for non-negative outcomes, including continuous ones (it handles zeros, unlike `log(Y)`).
- **Negative binomial** adds a dispersion parameter. **Zero-inflated** models add a separate process generating excess zeros.

## Censoring and truncation

**Censoring**: you observe the variable only within a range but keep the observation. Wages top-coded at a maximum; hours worked bounded below at zero.

**Truncation**: observations outside the range are absent entirely.

**Tobit** handles censoring at zero via a latent-variable model estimated by MLE. Its weaknesses: it assumes normality and homoskedasticity, and it is **inconsistent** if either fails. It also forces the same parameters to govern both the probability of a positive value and its magnitude, which is often economically implausible.

Alternatives: two-part models (a probit for `Y > 0`, then a regression on the positive values), or Poisson QMLE, which handles zeros without the normality assumption.

## Sample selection

The deepest problem in this chapter. You observe `Y` only for a **selected** subsample, and selection depends on unobservables that also affect `Y`.

Classic case: wages are observed only for people who work. If people work when their offered wage exceeds their reservation wage, the observed sample is not representative of the population, and OLS on it is biased.

Structure:
```
Y  = X'β + e                    outcome equation (observed only when S = 1)
S  = 1{Z'γ + u > 0}             selection equation
```

If `e` and `u` are correlated, then `E[e | S = 1, X] ≠ 0` and OLS on the selected sample is inconsistent.

**Heckman two-step (Heckit)**:

1. Estimate the selection probit; compute the **inverse Mills ratio** `λ̂ᵢ = φ(Zᵢ'γ̂)/Φ(Zᵢ'γ̂)`.
2. Regress `Y` on `X` and `λ̂` in the selected sample. The coefficient on `λ̂` estimates the selection correlation, and its t-test is a test for selection bias.

Second-step standard errors must account for the first-stage estimation (analytic correction or bootstrap).

**The critical practical point**: identification is technically achieved by the nonlinearity of the Mills ratio alone, but this is **very fragile** — the Mills ratio is close to linear over much of its range, and identification off functional form gives unstable, collinearity-plagued estimates. You need an **exclusion restriction**: a variable in `Z` that affects selection but not the outcome. This is an instrument for selection, and it is as hard to find and as much in need of justification as any other instrument. See [[11 - Instrumental Variables]].

Papers that run Heckit with `Z = X` should be read very sceptically.

## Panel versions and the incidental parameters problem

Adding fixed effects to a nonlinear model is not straightforward. With `N` individual effects and small `T`, the fixed effects are estimated inconsistently (each from only `T` observations), and this inconsistency **contaminates** `β̂`. This is the **incidental parameters problem**. For fixed-effects probit with `T = 2`, the bias in `β̂` is roughly 100%.

Workarounds:

- **Conditional logit** (Chamberlain): conditioning on the number of ones per individual removes the fixed effects exactly. Only works for logit, and only uses individuals who switch. Marginal effects are not recoverable without further assumptions.
- **Poisson fixed effects**: no incidental parameters problem — Poisson QMLE with fixed effects is consistent for fixed `T`. Another reason it is a good default.
- **Bias-correction methods** for large `T`.
- **Linear probability model with fixed effects**: biased for the true nonlinear model but transparent and consistent for the linear projection. In causal designs, often the pragmatic choice.

## Practical guidance

1. For a **causal** design (DiD, IV, RDD) with a binary outcome, the LPM with robust or clustered standard errors is usually the right choice. Transparency and compatibility with the design beat distributional fidelity.
2. For **prediction** of probabilities, use logit or probit and evaluate out of sample.
3. Always report **marginal effects**, not raw index coefficients, and say which kind (AME or MEM).
4. For non-negative outcomes including counts, **Poisson QMLE with robust standard errors** is a strong default — no distributional assumption, handles zeros, works with fixed effects.
5. For sample selection, an exclusion restriction is not optional. State it and defend it.
6. Do not compare probit/logit coefficients across models or samples.

Related:

- [[03 - Statistical Inference Foundations]]
- [[10 - Causality and Identification]]
- [[11 - Instrumental Variables]]
- [[13 - Panel Data]]
