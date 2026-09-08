---
title: Model Selection and Machine Learning
tags: [econometrics, model-selection, lasso, ridge, machine-learning, double-ml]
---

# Model Selection and Machine Learning

Source: Hansen, *Econometrics*, chapters 28-29.

## Two different goals

- **Prediction**: minimize out-of-sample error. Bias is fine if it buys variance. Interpretability optional.
- **Inference on a specific parameter**: get an honest confidence interval for one coefficient of interest, with many nuisance controls.

Machine learning solves the first well. Naively applying it to the second produces confidence intervals with badly wrong coverage. The last third of this note is about the fix.

## Model selection criteria

**AIC** = `-2·log L + 2k`. Estimates out-of-sample prediction error. Asymptotically equivalent to leave-one-out cross-validation. Not consistent for the "true" model — it tends to over-select — but is **efficient** for prediction.

**BIC** = `-2·log L + k·log n`. Heavier penalty. Consistent for the true model if one exists in the candidate set, but can under-select in finite samples.

**Mallows' Cp** — the least-squares analogue of AIC.

Use AIC for forecasting, BIC when you believe in a sparse true model, and — mostly — cross-validation.

## Cross-validation

The most reliable and most general tool.

**K-fold CV**: split the data into `K` folds; for each fold, fit on the other `K-1` and predict the held-out fold; average the prediction errors. `K = 5` or `10` is standard. `K = n` is leave-one-out, which for linear regression has a closed form via leverage — this is exactly the `σ̄²` estimator from [[05 - Least Squares Mechanics]].

Uses: choose a tuning parameter (bandwidth, `λ`, number of series terms), choose between model classes, estimate honest prediction error.

**Do it right**:

- Any data-dependent step — variable screening, standardization, imputation — must happen *inside* each fold. Screening variables on the full data and then cross-validating is a classic leakage error that makes garbage look predictive.
- For time series use **rolling-origin** or blocked CV. Random K-fold on time series lets the model train on the future.
- For clustered data, hold out whole clusters.

## Regularization

**Ridge regression**:
```
β̂ = argmin Σ(Yᵢ - Xᵢ'b)² + λ Σ bⱼ²
```
Closed form `(X'X + λI)⁻¹X'Y`. Shrinks all coefficients toward zero but sets none exactly to zero. Handles collinearity and `p > n`.

The key theoretical result (Hansen's Theorem 29.2): for a range of `λ` strictly between 0 and `2σ²/β'β`, ridge has **strictly lower mean squared error** than OLS. Some shrinkage always helps in MSE terms. This is the bias-variance trade in its purest form.

**Lasso**:
```
β̂ = argmin Σ(Yᵢ - Xᵢ'b)² + λ Σ |bⱼ|
```
The `L1` penalty produces **exact zeros**, so it does variable selection and estimation simultaneously. No closed form; solved by coordinate descent.

Under a **sparsity** assumption (only `s` coefficients are non-zero, `s` small relative to `n`), the lasso is consistent and achieves near-oracle rates, with the theory requiring roughly `s·log(p)/√n → 0`. Note `p` can grow *exponentially* in `n` — that is the remarkable part.

**Elastic net** mixes `L1` and `L2`; better than lasso when predictors are highly correlated in groups.

**Post-lasso**: run lasso to select variables, then run plain OLS on the selected set. Removes the shrinkage bias in the retained coefficients. This is the standard practice inside the inference procedures below.

**Always standardize** regressors before penalizing — otherwise the penalty depends on units — and never penalize the intercept.

## Why naive post-selection inference is broken

Suppose you lasso-select controls, then run OLS on the selected set, then report the t-statistic on your variable of interest.

This is a **post-model-selection (PMS) estimator**, and its sampling distribution is not the distribution of an OLS estimator. Selection is a discontinuous, data-dependent step, and it induces bias. Hansen shows the coverage of standard confidence intervals for the parameter of interest "can be far from the nominal level", with distortion increasing in the correlation between the treatment `D` and the controls `X`.

The intuition: lasso drops controls whose coefficients are small. But a control with a small coefficient in the outcome equation can still be a serious confounder if it is strongly related to `D` — the omitted variable bias is the *product* of two terms ([[04 - Conditional Expectation and Projection]]), and lasso only looks at one of them.

## Three fixes for honest inference

Model:
```
Y = Dθ + X'β + e,     E[e | D, X] = 0
D = X'γ + V,          E[V | X] = 0
```
`θ` is the parameter of interest; `X` is high-dimensional.

### 1. Double selection (Belloni, Chernozhukov, Hansen 2014)

1. Lasso `D` on `X`. Keep selected set `X₁`.
2. Lasso `Y` on `X`. Keep selected set `X₂`.
3. Let `X̄ = X₁ ∪ X₂`.
4. Regress `Y` on `D` and `X̄` by OLS to get `θ̂_DS`.
5. Use conventional heteroskedasticity-robust standard errors.

The idea: **keep a control if it matters for the outcome OR if it is correlated with the treatment.** The second selection is what fixes the confounding problem the naive approach misses. Under approximate sparsity of both equations, `θ̂_DS` and its t-ratio are asymptotically normal and conventional inference is valid.

Stata: `dsregress` or `pdslasso`. R: `hdm`.

### 2. Post-regularization / partialling-out lasso (Chernozhukov, Hansen, Spindler 2015)

Transform the structural equation to remove `X` entirely:
```
Y - E[Y|X] = (D - E[D|X])θ + e
```
Then:

1. Lasso (or post-lasso) `D` on `X`; get residual `V̂ = D - X'γ̂`.
2. Lasso (or post-lasso) `Y` on `X`; get residual `Û = Y - X'η̂`.
3. Regress `Û` on `V̂`. That coefficient is `θ̂_PR`.
4. Conventional robust standard error.

This is Robinson's partially linear estimator with lasso in place of kernel regression, and it is the Frisch-Waugh-Lovell theorem doing the work again ([[05 - Least Squares Mechanics]]).

**Why it is robust** — this is the elegant part. The moment condition for the naive approach is `m(θ,β) = E[D(Y - Dθ - X'β)]`, whose sensitivity to `β` is `-E[DX']`, non-zero when `D` and `X` are correlated. So errors in estimating `β` propagate into `θ̂`. For the partialled-out moment, the sensitivity is `-E[(D - X'γ)X'] = -E[VX'] = 0`. The moment condition is **locally insensitive to errors in the nuisance parameters** — this is called Neyman orthogonality, and it is why model-selection mistakes wash out.

Efficiency-wise, `θ̂_PR` is more parsimonious than double selection (different components of `X` can matter for `D` and for `Y`); `θ̂_DS` is more robust. Both are valid.

Stata: `poregress`. R: `hdm`.

### 3. Double / debiased machine learning (DML)

Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018). Adds **cross-fitting** to the partialling-out estimator:

1. Randomly partition the sample into `K` folds.
2. For each fold `k`: estimate `γ` and `η` using **all observations except fold `k`**.
3. Form residuals for fold `k` using those leave-fold-out estimates: `V̂ₖ = Dₖ - Xₖγ̂₋ₖ` and `Ûₖ = Yₖ - Xₖη̂₋ₖ`.
4. Stack all folds and regress `Û` on `V̂`.
5. Conventional robust standard error.

`K = 10` is recommended; computational cost is roughly proportional to `K`.

**Why cross-fitting helps**: without it, the nuisance estimate and the residual in the same observation are dependent, and the resulting error term is `Op(‖γ‖₀ log p/√n)`. With sample splitting they are independent, so the term is mean zero conditionally and its order drops to `Op(√(‖γ‖₀ log p / n))` — a smaller order. In practice this reduces post-selection bias.

The nuisance functions `E[Y|X]` and `E[D|X]` need not be lasso — random forests, boosting, or neural networks work, as long as they converge fast enough (roughly `n^{-1/4}`). This is why DML is the standard bridge between machine learning and causal inference.

**Disadvantages** Hansen flags: the estimate depends on the random split, so two researchers with the same data get different numbers (mitigated by larger `K` or averaging over splits); and nuisance functions are estimated on smaller samples. DML is best suited to large `n`.

Stata: `xporegress`. R: `DoubleML`.

## Lasso IV

For high-dimensional instruments, lasso the first stage to select instruments, then 2SLS. The split-sample and cross-fit versions ("Lasso SSIV") have weaker rate requirements — `‖Γ‖₀ log p / n → 0` rather than `‖Γ‖₀ log p / √n → 0` — because sample splitting breaks the dependence between instrument selection and the second stage. Lasso SSIV is the preferred variant. Note this does **not** dissolve the weak-instrument problem: selecting instruments from a large pool by fit is exactly how you manufacture spuriously strong first stages. See [[11 - Instrumental Variables]].

Stata: `ivlasso`.

## Where prediction methods legitimately belong in econometrics

- **Estimating nuisance functions** for the procedures above.
- **Propensity score estimation** in high dimensions.
- **Heterogeneous treatment effects**: causal forests, generic ML inference on CATEs.
- **Constructing variables**: extracting features from text, images, or satellite data to use downstream.
- **Genuine forecasting problems** where nobody is claiming causality.

Where they do **not** belong: as a substitute for an identification argument. A random forest that predicts `Y` from `D` extremely well tells you nothing about what would happen if you changed `D`. See [[10 - Causality and Identification]].

## Practical checklist

1. Say which goal you have: prediction or inference on a parameter.
2. For prediction, evaluate honestly out of sample with the correct CV scheme for your data structure.
3. For inference with many controls, use double selection, partialling-out, or DML — never naive post-selection.
4. Standardize before regularizing; do not penalize the intercept; report `λ` and how you chose it.
5. Record random seeds — lasso paths, CV folds, and DML splits are all stochastic.
6. Report sparsity: how many variables were selected, and which ones.
7. Compare against a simple benchmark (OLS with a handful of sensible controls). If ML changes the answer a lot, understand why before believing it.

Related:

- [[05 - Least Squares Mechanics]]
- [[10 - Causality and Identification]]
- [[11 - Instrumental Variables]]
- [[16 - Nonparametrics, Quantiles, and RDD]]
- [[19 - Applied Workflow and Common Mistakes]]
