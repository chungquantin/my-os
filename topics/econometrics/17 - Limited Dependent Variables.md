---
title: Limited Dependent Variables
tags: [econometrics, probit, logit, tobit, selection, mle]
---

# Limited Dependent Variables

Source: Hansen, *Econometrics*, chapters 25-27.

When the outcome is not a continuous, unbounded number, linear regression may be awkward or invalid. This note covers binary, multinomial, count, censored, and selected outcomes.

## Binary outcomes

`Y ∈ {0, 1}`: employed or not, defaulted or not, order filled or not. The object of interest is the **response probability**
```
P(x) = P[Y = 1 | X = x] = E[Y | X = x]
```
and its derivative, the **marginal effect** `∂P(x)/∂x`, which is just the regression derivative.

The error has a two-point conditional distribution (`1 - P(X)` with probability `P(X)`, else `-P(X)`) and conditional variance
```
var[e | X] = P(X)(1 - P(X))
```
so binary-outcome regressions are **inherently heteroskedastic**. Also: scatterplots of binary `Y` against `X` are useless, since the points lie only on two horizontal lines. Plot empirical frequencies by bin instead.

### Linear probability model (LPM)

`P(x) = x'β`, estimated by OLS.

Pros: coefficients *are* the marginal effects; estimation is least squares, so fixed effects, IV, and DiD all carry over directly.

Con: it does not respect the `[0,1]` boundary, and in practice this bites. Hansen's marriage example (CPS, college-educated men, `Y` = married, `X` = age): the fitted LPM violates `[0,1]` for men above 67 and says an **80-year-old is married with probability 113%**. His verdict is blunt: *"Overall, the linear probability model is a poor choice for calculation of probabilities."*

So: use the LPM when you want a transparent linear causal estimate embedded in a design (DiD, IV, RDD), always with robust or clustered standard errors — and do not use it to report fitted probabilities.

### Index models

```
P(x) = G(x'β)
```
`G` is the **link function**, `x'β` the **linear index**. `G` is a distribution function, so `0 ≤ G ≤ 1` is automatic. Marginal effect: `∂P/∂x = βg(x'β)`.

- **Probit**: `G = Φ`, the standard normal CDF.
- **Logit**: `G = Λ(u) = (1 + exp(-u))⁻¹`, the logistic CDF. Closed form, so faster.

**Latent variable interpretation.** `Y* = X'β + e` with `e ~ G` symmetric about zero, and `Y = 1{Y* > 0}`. Then `P(x) = G(x'β)`. Economically, `Y*` is the relative utility of the two options and the individual picks the higher — so a discrete-choice model derived from utility maximization *is* an index model whose link is the error distribution.

### The two series models — and Hansen's actual recommendation

Two more options that people skip and shouldn't:

- **Linear series model**: `P(x) = x_K'β_K` where `x_K` is a vector of transformations (splines, polynomials) of `x`. Flexible, linear, but still not boundary-respecting.
- **Index series model** (e.g. **probit series**): `P(x) = G(x_K'β_K)`. Flexible *and* boundary-respecting.

In Hansen's marriage example the plain probit is "quite poor" — nearly as bad as the LPM up to age 60 — while the linear series model puts a 19-year-old's marriage probability at **-27%**. Both series models fit well above age 25; only the probit series is good everywhere.

His summary: **the probit series model has several excellent features — simple, globally approximates any continuous response probability, respects `[0,1]`.** A linear series model is a reasonable second choice with the boundary caveat.

The practical lesson: reaching for probit instead of the LPM does not by itself solve the fit problem. **Getting the functional form of the index right matters far more than the choice of link.**

### Estimation

Both are estimated by maximum likelihood ([[03 - Statistical Inference Foundations]]). Writing `Z = X` if `Y = 1` and `Z = -X` if `Y = 0`, the log-likelihood collapses to

```
ℓₙ(β) = Σᵢ log G(Zᵢ'β)
```

For both probit and logit `G` is **log concave**, which makes the Hessian positive definite and the log-likelihood **globally concave** — the MLE is unique and numerical optimization is straightforward. Stata: `probit`, `logit`. R: `glm(Y~X, family=binomial(link="probit"))`.

Under misspecification the estimator converges to a **pseudo-true value** — the best-fitting `β` with respect to the expected log mass function — and the sandwich variance `V = Q⁻¹ΩQ⁻¹` applies. Under correct specification the information matrix equality gives the simplification `V = Q⁻¹`.

**Practical warning**: Stata and R report the **non-robust** `V̂⁰ = Q̂⁻¹` by default. For the sandwich form use `vce(robust)` in Stata or the `sandwich` package in R. Same discipline as [[06 - Standard Errors and Clustering]].

### The scale is not identified

Write `e = σε`. Then `P[Y=1|X] = G(x'β/σ) = G(x'β*)` with `β* = β/σ`. **Only the ratio `β/σ` is identified**, and probit and logit simply pick different normalizations: `σ = 1` for probit, `σ = π/√3 ≈ 1.8` for logit.

Consequences:

1. A probit coefficient means `β/σ`; a logit coefficient means `β/ν` with `ν = σ√3/π`. To put them on the same scale, **multiply probit coefficients by 1.8, or divide logit coefficients by 1.8**.
2. Coefficients are **not comparable across specifications or samples** — adding a control changes `σ` and so rescales everything. A coefficient change has no omitted-variable-bias reading. Compare **marginal effects**, not coefficients.

What *is* identified: scaled coefficients `β/σ`, ratios of coefficients `β₁/β₂`, and marginal effects.

Going further: if you write the latent equation nonparametrically as `Y* = m(X) + e`, then neither `G` nor `m` is separately identified — only `P(x) = 1 - G(-m(x))`. An important implication: **there is no loss of generality in assuming normality, so long as the index is treated flexibly.**

### Marginal effects — this is where beginners go wrong

The coefficient `βⱼ` is *not* a marginal effect. The marginal effect is
```
δ(x) = ∂P/∂x = β g(x'β)
```
which varies with `x`. In the marriage example, the marginal effect of age is about 0.06/year between 20 and 30 and near zero above 40.

Report the **average marginal effect**:
```
AME = E[δ(X)] = β E[g(X'β)],    estimated by  β̂ (1/n) Σᵢ g(Xᵢ'β̂)
```
Stata: `margins, dydx(*)`. Standard errors via the delta method ([[07 - Asymptotic Theory]]).

When `X` contains nonlinear transformations the marginal effect must be built by hand. For `P = G(β₀ + β₁x + ... + β_p x^p)`,
```
δ(x) = (β₁ + ... + pβ_p x^{p-1}) g(β₀ + β₁x + ... + β_p x^p)
```

The **marginal effect at the mean** evaluates at `X̄`; it is less meaningful, since the "average person" may not exist.

For logit, `exp(βⱼ)` is an **odds ratio** — not a probability, and routinely misreported as one.

### Worked example: probability of marriage

CPS, men aged 19-35, `n = 9137`, robust standard errors:

| | Logit coef | Logit AME | Probit coef | Probit AME |
| --- | --- | --- | --- | --- |
| age | 0.217 (.006) | **0.044** (.001) | 0.132 (.006) | **0.045** (.001) |
| education | 0.014 (.010) | 0.003 (.002) | 0.009 (.006) | 0.003 (.002) |
| Black | -0.767 (.092) | -0.156 (.018) | -0.454 (.054) | -0.153 (.018) |
| Hispanic | -0.084 (.063) | -0.017 (.013) | -0.048 (.038) | -0.017 (.013) |
| West | 0.383 (.072) | 0.078 (.015) | 0.228 (.044) | 0.077 (.015) |

Two lessons Hansen draws:

1. **Logit vs probit is unimportant.** The coefficients differ by roughly the 1.8 scale factor; the AMEs are essentially identical (0.044 vs 0.045).
2. **Specification is critical.** The sample was restricted to ages 19-35 precisely so a single linear age term would fit. Estimated on the full sample with a linear age term, the AME of age comes out at **1% per year rather than 4.5%** — a large mis-estimate.

### Semiparametric binary choice

`P[Y=1|X] = G(X'β)` with `G` unknown. Manski's (1975) **maximum score estimator** identifies `β` up to scale under only `med[e|X] = 0`, but cannot recover response probabilities or marginal effects. Klein and Spady (1993) add independence of `e` and `X`, which identifies `G` too, and estimate both by a nested criterion.

Hansen's assessment of the modern view: these rely on the *linear index* assumption. Relax that to a nonparametric `m(x)` and neither `G` nor `m` is identified — only the composite. So the sensible route is to treat `P(x)` as nonparametrically identified and estimate it by **series approximation with a linear, probit, or logit link**. "There is no gain from the restriction to the function form `G(x'β)`, and hence no gain from the semiparametric approach."

### IV probit

When a regressor is endogenous, the latent-variable system

```
Y₁* = X'β₁ + Y₂β₂ + e₁,   Y₂ = X'γ₁ + Z'γ₂ + e₂,   Y₁ = 1{Y₁* > 0}
```

is estimated by MLE under joint normality of `(e₁, e₂)`. The normality assumption does real work: it factors the joint density into the conditional distribution of `Y₁` given `Y₂` and the marginal of `Y₂`. No comparable factorization exists in a logit framework. Stata: `ivprobit`.

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

Three distinct objects, and keeping them apart is most of the battle:

- **Uncensored** `Y*` — the latent variable you care about.
- **Censored** `Y = max(Y*, 0)` — negative values piled up at the boundary, observations retained.
- **Truncated** `Y#` — the censored observations *deleted* from the sample.

Hansen's example: `tabroad` (transfers from abroad) in a Philippine household survey is exactly 0 for 80% of households and continuously distributed with a thick right tail for the other 20%.

### Tobit

Tobin (1958), also called censored regression or the **Type 1 Tobit model**:
```
Y* = X'β + e,   e | X ~ N(0, σ²),   Y = max(Y*, 0)
```

The three conditional means are

```
m*(x) = E[Y*|X] = X'β                                       uncensored
m(x)  = E[Y |X] = X'β Φ(X'β/σ) + σ φ(X'β/σ)                 censored
m#(x) = E[Y#|X] = X'β + σ λ(X'β/σ)                          truncated
```

with `λ(u) = φ(u)/Φ(u)` the **inverse Mills ratio**. They are ordered

```
m*(x) ≤ m(x) ≤ m#(x)
```

with strict inequality wherever censoring occurs. So **dropping the zeros is worse than keeping them** — truncation has the largest bias. That settles the natural first question ("should I just drop the zeros?"): no.

**How large is the OLS bias?** Greene (1981) gives a usable rule for normally distributed regressors:

```
β_BLP = β(1 - π),        π = P[Y = 0], the censoring proportion
```

Least squares slopes are shrunk toward zero **in proportion to the censoring percentage**, which you can read straight off the data. Hansen's practical guidance: if `π` is small enough that the implied bias is acceptable (say under 5%), plain least squares on the full sample is defensible; if `π` is large (say 10% or more), use a method that corrects for censoring.

**Estimation.** MLE on the mixed discrete/continuous density
```
ℓₙ(β,σ²) = Σ_{Yᵢ=0} log Φ(-Xᵢ'β/σ) + Σ_{Yᵢ>0} log[ σ⁻¹φ((Yᵢ - Xᵢ'β)/σ) ]
```
Reparameterizing as `γ = β/σ`, `ν = 1/σ` (Olsen 1978) makes the log-likelihood **globally concave**, so Newton-based optimizers converge reliably. Stata: `tobit`. R: `AER::tobit`.

**Which assumptions matter.** Linearity is not critical — read `X'β` as a series approximation. **Independence of the error from `X` and normality are both critical**: heteroskedasticity changes the censoring process, and normality is hard to justify from first principles. Nonparametrically, `m(x)` and `F(e)` are identified when `e ⊥ X` and `m(X)` has full support; if the support fails, `E[e]` and hence the intercept of `m(x)` are not identified.

### CLAD and CQR — the robust alternative

Powell (1984, 1986) found the way around normality, and this is the part most treatments omit.

Quantiles are **equivariant to monotone transformations**, so censoring commutes with taking quantiles:
```
Q_τ[Y | X] = max(q_τ(X), 0)
```
The conditional quantile of the *censored* variable is the censored conditional quantile of the latent variable. This means `q_τ(x)` is identified wherever `q_τ(x) > 0`, **with no assumption that the error is independent of `X` and no distributional assumption at all**. Hansen calls it "an important conceptual breakthrough".

The estimators follow directly:

```
CLAD:  M̂ₙ(β) = (1/n) Σ |Yᵢ - max(Xᵢ'β, 0)|                censored least absolute deviations
CQR:   M̂ₙ(β;τ) = (1/n) Σ ρ_τ(Yᵢ - max(Xᵢ'β, 0))            censored quantile regression
```

Caveats: the criteria are **not globally convex**, so optimizers can land on a local minimum; and identification needs a positive fraction of the population with `X'β > 0` plus full rank of the design on that sub-population — enough variation in the uncensored region.

Stata: `clad` (add-on). R: `crq` in `quantreg`.

**Hansen's worked comparison** (Philippine transfers on income, 18% censoring, four estimators): the shape of the relationship — slope near -1 at low income, flat at high income, break at 20,000 pesos — is robust across all four. What differs is the **level**: OLS sits several thousand pesos above the others, and Tobit is also shifted up. Since the regression is negatively sloped, the censoring probability rises with income, so the OLS bias is positive and increasing. His conclusion: **"the CLAD estimates are the preferred choice because they are robust to both censoring and non-normality."**

Other alternatives worth knowing: two-part models (a probit for `Y > 0`, then a regression on positives), and Poisson QMLE, which handles zeros with no distributional assumption.

## Sample selection

The deepest problem in this chapter. You observe `Y` only for a **selected** subsample, and selection depends on unobservables that also affect `Y`.

Classic case: wages are observed only for people who work. If people work when their offered wage exceeds their reservation wage, the observed sample is not representative of the population, and OLS on it is biased.

Hansen's four canonical cases: **wage regressions** (wages observed only for those who work), **program evaluation** with volunteers rather than random assignment, **surveys** with low response rates, and **product ratings**, where only people with strong opinions bother to respond.

View sampling as two stages: `(Y, X)` is drawn, then the pair is either selected (`S = 1`) or not. With `Y = X'β + e` and `E[e|X] = 0`, the CEF *in the selected sample* is

```
E[Y | X, S=1] = X'β + E[e | X, S=1]
```

and selection bias is the second term. Modelling selection as `S = 1{X'γ + u > 0}` and projecting `e = ρu + ε`:

```
E[Y | X, S=1] = X'β + ρ λ(X'γ)
```

**Read this as omitted variable bias.** A regression of `Y` on `X` in the selected sample omits `ρλ(X'γ)`, which is correlated with `X`. So sample selection bias arises **if and only if `ρ ≠ 0`** — if and only if selection is correlated with the equation error. If selection is exogenous (`ρ = 0`), the term vanishes and OLS on the selected sample is fine.

Two refinements that are easy to miss:

1. **If `X'γ = γ₀` is constant** — selection does not depend on the regressors — then `λ(X'γ)` is a constant too and selection only shifts the **intercept**. Marginal effects are unaffected. *Sample selection bias distorts slopes only when the selection equation depends non-trivially on `X`.*
2. Unlike censoring, which always **attenuates** the regression function, selection can **steepen or flatten** it. The direction of the distortion is not predictable in general.

### Heckman's model and the two-step estimator

```
Y* = X'β + e
S* = Z'γ + u,   S = 1{S* > 0}
Y  = Y* if S = 1, missing otherwise
(e, u) ~ N(0, [[σ², σ₂₁], [σ₂₁, 1]])
```

The variance of `u` is not identified, so it is normalized to 1. In Heckman's original example `Y*` is the wage a person would earn if employed, `S` is employment status, `e` is unobserved ability and `u` is whatever determines employment — and those two are very likely correlated.

The selected-sample CEF is
```
E[Y | X, Z, S=1] = X'β + σ₂₁ λ(Z'γ)
```

**Two-step (Heckit)**:

1. Construct `S` from the observed data.
2. Probit regression of `S` on `Z` → `γ̂`.
3. Form `λ̂ᵢ = λ(Zᵢ'γ̂) = φ(Zᵢ'γ̂)/Φ(Zᵢ'γ̂)`.
4. Least squares of `Yᵢ` on `(Xᵢ, λ̂ᵢ)` on the sub-sample with `Sᵢ = 1` → `(β̂, σ̂₂₁)`.

`λ̂` is a **generated regressor**, so second-step standard errors must be corrected for the first-stage estimation (analytic correction or bootstrap).

**A useful by-product**: `σ̂₂₁` measures the degree of selection endogeneity. The t-statistic on `σ̂₂₁` is a **test of the null of exogenous selection**.

An alternative to the two-step is **joint maximum likelihood** on the mixed density of `(S, Y)` — more efficient under the normality assumption, more fragile without it.

**The critical practical point**: `λ(·)` is a nonlinear function, so `β` is technically identified even when `Z = X`. But the Mills ratio is close to linear over much of its range, so identification off functional form alone produces unstable, collinearity-plagued estimates. Standard practice — and the right practice — is to insist on an **exclusion restriction**: a variable in `Z` that affects selection but not the outcome. That is an instrument for selection, and it needs the same defence as any other instrument. See [[11 - Instrumental Variables]].

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
5. For **censored** outcomes: compute the censoring proportion `π` first. Greene's rule says the OLS slope bias is roughly `π`, so small `π` means you can ignore it. When you cannot, prefer **CLAD/CQR** over Tobit — same robustness to censoring, plus robustness to non-normality.
6. Never fix censoring by dropping the censored observations — truncation is worse than censoring.
7. For sample selection, an exclusion restriction is not optional. State it and defend it. Report the t-statistic on the selection covariance as a test of exogenous selection.
8. Do not compare probit/logit coefficients across models or samples.

Related:

- [[03 - Statistical Inference Foundations]]
- [[10 - Causality and Identification]]
- [[11 - Instrumental Variables]]
- [[13 - Panel Data]]
