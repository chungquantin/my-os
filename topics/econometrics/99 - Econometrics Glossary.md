---
title: Econometrics Glossary
tags: [econometrics, glossary, reference, notation]
---

# Econometrics Glossary

Terms as used in Hansen's *Econometrics* and *Probability and Statistics for Economists*.

## Notation

| Symbol | Meaning |
| --- | --- |
| `Y` | Outcome / dependent variable |
| `X` | Regressors / covariates (usually a vector) |
| `D` | Treatment, often binary |
| `Z` | Instrument |
| `e`, `ε` | Error term |
| `ê` | Residual (computed from a sample) |
| `β` | Population parameter |
| `β̂` | Estimator / estimate |
| `β̃` | Restricted or alternative estimator |
| `θ₀` | True parameter value |
| `n` | Sample size |
| `k` | Number of regressors |
| `G` | Number of clusters |
| `N`, `T` | Panel: number of units, number of periods |
| `E[Y]` | Expectation |
| `E[Y \| X]` | Conditional expectation |
| `P[·]` | Probability |
| `X'β` | Linear combination (prime = transpose) |
| `→p` | Converges in probability |
| `→d` | Converges in distribution |
| `Op(1)`, `op(1)` | Bounded in probability / negligible |
| `1{A}` | Indicator: 1 if `A` true, else 0 |
| `hᵢᵢ` | Leverage of observation `i` |
| `P`, `M` | Projection and annihilator matrices |

## Terms

**ACE / ATE** — Average causal effect / average treatment effect. `E[Y(1) - Y(0)]`.

**ATT** — Average treatment effect on the treated. `E[Y(1) - Y(0) | D = 1]`.

**Adjusted R²** — R² with a penalty for the number of regressors. Ad hoc; prefer cross-validation.

**AIC** — Akaike information criterion. `-2 log L + 2k`. Efficient for prediction, not consistent for model selection.

**Annihilator matrix** — `M = I - X(X'X)⁻¹X'`. Maps `Y` to residuals.

**Asymptotic normality** — `√n(θ̂ - θ) →d N(0, V)`. The basis of standard errors.

**Attenuation bias** — Bias toward zero caused by classical measurement error in a regressor.

**Autocorrelation** — Correlation of a series with its own lags.

**BIC** — Bayesian information criterion. `-2 log L + k log n`. Consistent for a true sparse model.

**BLUE** — Best linear unbiased estimator. What OLS is under homoskedasticity (Gauss-Markov).

**Bootstrap** — Resampling the data with replacement to approximate a sampling distribution. See [[09 - Bootstrap and Resampling]].

**CATE** — Conditional average treatment effect, `E[Y(1) - Y(0) | X = x]`.

**CEF** — Conditional expectation function, `m(x) = E[Y | X = x]`. The best predictor under squared loss.

**CIA** — Conditional independence assumption. Conditional on `X`, treatment is independent of unobservables. Licenses a causal reading of regression.

**Cluster-robust** — Standard errors allowing arbitrary correlation within groups. Effective sample size is `G`, not `n`.

**Cointegration** — Two non-stationary series with a stationary linear combination.

**Collider** — A variable caused by both treatment and outcome. Never condition on it.

**Compliers** — Units whose treatment status is changed by the instrument. The population LATE speaks about.

**Consistency** — `θ̂ →p θ`. The estimator converges to the truth with more data.

**Control function** — Including a first-stage residual as a regressor; equivalent to 2SLS in the linear model, generalizes beyond it.

**Cross-fitting** — Estimating nuisance functions on data excluding the fold where they are used. The key step in DML.

**Delta method** — Standard errors for a smooth function of estimators, `var[g(β̂)] ≈ G'VG`.

**DiD** — Difference in differences. See [[14 - Difference in Differences]].

**DML** — Double / debiased machine learning. Orthogonal moments plus cross-fitting for valid inference with ML nuisance estimates.

**Double selection** — Lasso the outcome and the treatment equations separately, take the union of selected controls, then OLS.

**Endogeneity** — `E[Xe] ≠ 0`. Makes OLS inconsistent.

**Ergodicity** — Time averages converge to population averages.

**Estimand / estimator / estimate** — What you want / the formula / the number.

**Exclusion restriction** — The instrument affects the outcome only through the endogenous variable. Untestable when just identified.

**Fixed effects** — Unit-specific intercepts, allowed to correlate with regressors; removed by the within transformation.

**FWL (Frisch-Waugh-Lovell)** — Regression coefficients can be obtained by partialling out other regressors. The engine behind fixed effects and partialling-out lasso.

**GMM** — Generalized method of moments. Estimation by making sample moment conditions close to zero.

**HAC** — Heteroskedasticity and autocorrelation consistent standard errors (Newey-West).

**HC0-HC3** — Heteroskedasticity-consistent variance estimators. HC1 is Stata's `, r`; HC2 and HC3 are preferred.

**Heteroskedasticity** — Error variance varying with `X`. The normal case in cross-sections.

**Homoskedasticity** — Constant error variance. The default assumption in software and rarely true.

**Identification** — Whether the parameter is determined by the population distribution of observables. Logically prior to estimation.

**IIA** — Independence of irrelevant alternatives. The restrictive assumption of multinomial logit.

**Incidental parameters problem** — Inconsistency of nonlinear fixed-effects estimators with small `T`.

**Instrument** — A variable correlated with the endogenous regressor and excluded from the outcome equation.

**Jackknife** — Leave-one-out resampling. Source of HC3 and the BCa acceleration constant.

**Jensen's inequality** — `E[g(X)] ≥ g(E[X])` for convex `g`. Why `E[log Y] ≠ log E[Y]`.

**J statistic / Sargan** — Test of overidentifying restrictions, `→d χ²_{ℓ-k}`.

**Kernel** — Weight function for local averaging; `h` is the bandwidth.

**LATE** — Local average treatment effect. What IV estimates under heterogeneity: the effect on compliers.

**Lasso** — `L1`-penalized regression. Produces exact zeros, so it selects variables.

**Leverage** — `hᵢᵢ`, how much an observation pulls its own fitted value. Average `k/n`.

**LIML** — Limited information maximum likelihood. More robust than 2SLS to weak and many instruments.

**Linear projection** — The best linear approximation to the CEF: `β = E[XX']⁻¹E[XY]`. What OLS estimates.

**LLN** — Law of large numbers. Sample averages converge to population averages.

**LPM** — Linear probability model. OLS with a binary outcome.

**Marginal effect** — `∂P/∂X` in a nonlinear model. Not the coefficient.

**Mean independence** — `E[e | X] = 0`. Stronger than zero correlation.

**Mediator** — A variable on the causal path from treatment to outcome. Do not control for it if you want the total effect.

**MLE** — Maximum likelihood estimation. Efficient under correct specification, fragile otherwise.

**Monotonicity** — No defiers; the instrument never moves anyone the wrong way. Required for LATE.

**Moulton factor** — `1 + ρ(N-1)`. How much clustering inflates the true variance over the classical formula.

**MSE** — Mean squared error, `bias² + variance`.

**Nickell bias** — Bias of order `1/T` in dynamic fixed-effects panels.

**Overidentified** — More moment conditions or instruments than parameters. Permits a specification test.

**Overlap / common support** — Every covariate cell contains both treated and untreated units. Necessary for the CIA to be usable.

**Panel** — Repeated observations on the same units.

**Parallel trends** — The DiD identifying assumption: absent treatment, groups would have moved together.

**Percentile-t** — Bootstrap interval built from the studentized statistic. Achieves an asymptotic refinement.

**Potential outcomes** — `Y(0)`, `Y(1)`. Only one is ever observed per unit.

**Power** — Probability of rejecting a false null.

**Projection matrix** — `P = X(X'X)⁻¹X'`. Maps `Y` to fitted values.

**Propensity score** — `P(D = 1 | X)`. Used for matching and weighting.

**p-value** — The smallest significance level at which the null would be rejected. Not the probability the null is true.

**QMLE** — Quasi-maximum likelihood. Consistent for the conditional mean under misspecification, with robust standard errors. Poisson QMLE is the key case.

**Random effects** — Individual effects assumed uncorrelated with regressors; estimated by GLS.

**RDD** — Regression discontinuity design. Treatment assigned by a cutoff rule; compare units just either side.

**Ridge** — `L2`-penalized regression. Shrinks all coefficients; always beats OLS in MSE for some `λ`.

**Robust standard errors** — Heteroskedasticity-consistent (HC) errors.

**Sandwich variance** — `A⁻¹BA⁻¹`. The general form of the variance of an M-estimator.

**Sample selection** — Outcome observed only for a non-random subsample. Corrected by Heckman-type models, which need an exclusion restriction.

**Saturated model** — Full dummies and interactions. Exactly equals the CEF, so no functional-form assumption.

**Serial correlation** — Autocorrelation of errors over time.

**Slutsky's theorem** — Lets you combine convergence in probability with convergence in distribution.

**Sparsity** — Assumption that only a few coefficients are non-zero. What makes lasso theory work.

**Spurious regression** — High R² and large t-statistics between unrelated non-stationary series.

**Stationarity** — Distributional properties invariant to time shifts.

**Strict exogeneity** — `E[Xᵢₛεᵢₜ] = 0` for all `s, t`. Required for fixed effects; violated by lagged dependent variables.

**Test inversion** — Building a confidence interval as the set of values a test does not reject. Better than the delta method for nonlinear parameters.

**Two-way fixed effects (TWFE)** — Unit and time fixed effects. The standard DiD regression; problematic under staggered adoption with heterogeneous effects.

**Unit root** — Non-stationary process with permanent shocks; `Yₜ = Yₜ₋₁ + eₜ`.

**Wald test** — `W = (θ̂ - θ₀)'V̂⁻¹(θ̂ - θ₀) →d χ²_q`. Generalizes the t-test to multiple restrictions.

**Weak instruments** — Instruments with a small first-stage relationship. Cause inconsistency, median bias, non-normality, and spuriously small standard errors. First-stage `F > 10` is the rule of thumb.

**Wild cluster bootstrap** — Resampling method for inference with few clusters.

**Within transformation** — Subtracting individual means. Eliminates fixed effects and time-invariant regressors.

Related:

- [[00 - Econometrics Hub]]
