---
title: Time Series
tags: [econometrics, time-series, forecasting, stationarity, cointegration]
---

# Time Series

Source: Hansen, *Econometrics*, chapters 14 (univariate), 15 (multivariate), 16 (non-stationary).

## What changes when data are ordered

In cross-sections we assume independent observations. In time series, observation `t` is correlated with observation `t-1` by construction — that dependence is the object of interest, not a nuisance. Two things must change:

1. **The law of large numbers and CLT need new conditions.** i.i.d. is replaced by *stationarity* plus a *mixing* or *ergodicity* condition — roughly, "dependence dies out as observations get further apart".
2. **Standard errors must account for serial correlation.** Robust-to-heteroskedasticity is not enough; you need robust-to-autocorrelation as well.

## Stationarity

A series is **strictly stationary** if the joint distribution of any block of observations is unchanged by shifting the block in time.

**Covariance (weak) stationarity** is the working definition: constant mean, constant variance, and autocovariance `cov(Yₜ, Yₜ₋ₖ) = γ(k)` depending only on the lag `k`, not on `t`.

**Why it matters**: without stationarity, "the mean" is not a well-defined thing to estimate. Sample averages of a trending or exploding series do not converge to anything useful, and regressions between two trending series produce **spurious regression** — high `R²`, huge t-statistics, no relationship.

**Ergodicity** adds that time averages converge to population averages, which is what makes estimation from a single realization possible at all.

## Autocorrelation

The **autocorrelation function (ACF)** is `ρ(k) = γ(k)/γ(0)`. The **partial autocorrelation function (PACF)** is the correlation between `Yₜ` and `Yₜ₋ₖ` after removing intermediate lags.

Reading the pair is the classical identification method:

| Pattern | Suggests |
| --- | --- |
| ACF decays geometrically, PACF cuts off after lag `p` | AR(`p`) |
| ACF cuts off after lag `q`, PACF decays | MA(`q`) |
| Both decay | ARMA |
| ACF decays extremely slowly, near 1 for many lags | Unit root / non-stationarity |

## Core models

**Autoregression AR(p)**:
```
Yₜ = α + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + eₜ
```
Estimated by OLS. Stationary iff all roots of the characteristic polynomial lie outside the unit circle; for AR(1), `|φ| < 1`.

The AR(1) coefficient has a direct interpretation: `φ` is the persistence, and the half-life of a shock is `log(0.5)/log(φ)` periods.

**Moving average MA(q)**: `Yₜ = eₜ + θ₁eₜ₋₁ + ... + θ_qeₜ₋q`. Estimated by MLE since the errors are unobserved.

**ARMA(p,q)** combines them. **ARIMA(p,d,q)** differences the series `d` times first.

**Wold decomposition**: any covariance-stationary process can be written as an infinite MA in uncorrelated innovations plus a deterministic part. This is the theoretical justification for the whole ARMA apparatus as an approximation device.

**Lag selection**: use AIC or BIC over a grid of `(p,q)`. AIC tends to over-select and is better for forecasting; BIC is consistent for the true order if one exists. See [[18 - Model Selection and Machine Learning]].

## Inference with serial correlation

For a regression with time-series data, the OLS estimator is still consistent under mild conditions, but its variance is

```
V = Q⁻¹ Ω Q⁻¹     with  Ω = Σ_{k=-∞}^{∞} E[XₜeₜXₜ₋ₖ'eₜ₋ₖ]
```

The **long-run variance** `Ω` sums autocovariances across all lags, not just the contemporaneous term.

**HAC (Newey-West) estimator**:
```
Ω̂ = Γ̂₀ + Σ_{k=1}^{M} w(k)(Γ̂ₖ + Γ̂ₖ')
```
with a kernel weight `w(k)` (Bartlett is standard) and bandwidth `M`. Bandwidth choice matters: too small leaves serial correlation unaccounted for, too large adds noise. Common automatic rules: Andrews (1991), Newey-West (1994). A crude default is `M ≈ 4(T/100)^{2/9}`.

Practical rule: for regressions on time-series data, use HAC standard errors by default, and report the bandwidth and kernel.

## Forecasting

The optimal point forecast under squared loss is the conditional expectation `E[Yₜ₊ₕ | information at t]`.

Practical machinery:

- **Direct vs iterated multi-step forecasts.** Iterated: fit a one-step model and iterate forward. Direct: regress `Yₜ₊ₕ` on time-`t` information. Iterated is more efficient if the model is right; direct is more robust to misspecification.
- **Forecast intervals** need the variance of the multi-step forecast error, which accumulates across horizons.
- **Evaluation**: out-of-sample RMSE on a hold-out period, using a rolling or expanding window. In-sample fit is nearly worthless for forecast comparison.
- **Diebold-Mariano test** compares the predictive accuracy of two forecasts formally.
- Forecast **combination** (simply averaging forecasts) very often beats individual models. This is one of the most robust empirical findings in the field.

**Structural breaks**: a model estimated over one regime forecasts badly in another. Test with Chow tests (known break date) or Andrews' sup-Wald test (unknown break date, non-standard critical values). Rolling-window estimation is a pragmatic partial answer.

## Multivariate time series

**Vector autoregression (VAR)**:
```
Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + eₜ
```
where `Yₜ` is a vector. Each equation is estimated by OLS (they share regressors, so system estimation gains nothing).

Uses:

- **Granger causality**: does `X` help predict `Y` beyond `Y`'s own past? A Wald test on the lags of `X` in the `Y` equation. Note this is a statement about *predictability*, not causality in the [[10 - Causality and Identification]] sense. Two variables both driven by an unobserved anticipation can Granger-cause each other with no structural link.
- **Impulse response functions**: the dynamic path of `Y` after a shock. Requires identifying which shocks are which — a Cholesky (recursive) ordering, sign restrictions, or external instruments. The identification problem here is exactly as serious as elsewhere in econometrics, and the ordering choice is a substantive assumption.
- **Forecast error variance decomposition**: how much of the variance of one variable at each horizon is attributable to each shock.

## Non-stationarity and unit roots

A **unit root** process (random walk) `Yₜ = Yₜ₋₁ + eₜ` has:

- Variance growing linearly with `t` — no fixed mean to converge to.
- Shocks that are permanent, never dying out.
- Non-standard asymptotics: the OLS coefficient converges at rate `T` (superconsistent), and its t-statistic has a Dickey-Fuller distribution, not normal.

**Spurious regression**: regress one independent random walk on another and you typically get a large `R²` and a t-statistic that grows with `T`, with no relationship whatever. This is the reason unit-root testing exists.

**Tests**:

- **Dickey-Fuller / Augmented Dickey-Fuller (ADF)**: null is a unit root. Non-standard critical values, and the version (with/without constant, with/without trend) matters.
- **Phillips-Perron**: nonparametric correction for serial correlation.
- **KPSS**: null is *stationarity* — the reverse. Running both and comparing conclusions is common practice.

Practical caveat: unit-root tests have low power against persistent-but-stationary alternatives (`φ = 0.95` versus `φ = 1`). Do not treat a failure to reject as proof of a unit root. Economic reasoning about whether shocks should be permanent is at least as informative as the test.

## Cointegration

Two series can each be non-stationary while a linear combination of them is stationary. Then they are **cointegrated**, and the stationary combination is a long-run equilibrium relationship they are tied to.

Examples: consumption and income; spot and futures prices of the same asset; prices of the same security on two venues; short and long interest rates.

**Error correction model (ECM)** — the Granger representation theorem says cointegration is equivalent to an ECM:
```
ΔYₜ = α(Yₜ₋₁ - βXₜ₋₁) + (short-run dynamics) + eₜ
```
The term in parentheses is last period's deviation from equilibrium; `α < 0` is the speed at which the system corrects back. This is the natural regression form for pairs trading and long-run relationships generally.

**Estimation and testing**:

- **Engle-Granger two-step**: regress `Y` on `X` (superconsistent), test the residual for a unit root using the ADF with *cointegration-specific* critical values (not the standard ones).
- **Johansen**: full-system maximum likelihood, tests the number of cointegrating relationships. Preferred when there are more than two series.

Warning: cointegration is a statement about a long-run *statistical* relationship. It is not causality, and cointegrating relationships can break down (regime change, structural break) exactly when it matters most.

## Volatility models

For financial and high-frequency data, conditional variance is itself time-varying and forecastable — **volatility clustering**.

- **ARCH(q)**: `σₜ² = ω + Σ αᵢ eₜ₋ᵢ²`.
- **GARCH(1,1)**: `σₜ² = ω + αeₜ₋₁² + βσₜ₋₁²`. Remarkably hard to beat empirically. Persistence is `α + β`, typically near 0.95-0.99 in daily data.
- Extensions: EGARCH and GJR-GARCH for asymmetry (negative returns raise volatility more), stochastic volatility models, and realized volatility from high-frequency data.

Estimated by MLE, usually with quasi-maximum-likelihood standard errors since the conditional distribution is rarely truly normal.

## Practical checklist

1. Plot the series. Always. Look for trends, level shifts, seasonality, changing volatility.
2. Test for stationarity, but weigh economic reasoning too.
3. If the series is non-stationary, either difference it or model the cointegrating relationship — do not regress levels on levels without a cointegration argument.
4. Use HAC standard errors; report kernel and bandwidth.
5. Evaluate forecasts out of sample, with a rolling window.
6. Check for structural breaks; a stable-looking in-sample fit is not evidence of stability.
7. When in doubt, benchmark against a random walk or a simple AR(1). Beating those honestly is harder than it looks.

Related:

- [[02 - Probability Foundations]]
- [[06 - Standard Errors and Clustering]]
- [[07 - Asymptotic Theory]]
- [[09 - Bootstrap and Resampling]]
- [[18 - Model Selection and Machine Learning]]
