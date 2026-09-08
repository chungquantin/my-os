---
title: Statistical Inference Foundations
tags: [econometrics, statistics, estimation, mle, foundations, beginner]
---

# Statistical Inference Foundations

Source: Bruce Hansen, *Probability and Statistics for Economists*, chapters 6-14.

## Estimators and how to judge them

An **estimator** `θ̂` is a function of the data intended to approximate a population quantity `θ`. Because it is a function of random data, it is itself random: it has a distribution, called the **sampling distribution**. Every claim about precision is a claim about that distribution.

Three criteria:

**Bias** = `E[θ̂] - θ`. Systematic error. An estimator is **unbiased** if bias is zero — on average across hypothetical repeated samples it lands on the truth.

**Variance** = `var[θ̂]`. Random error, sample to sample.

**Mean squared error** = `E[(θ̂ - θ)²] = bias² + variance`. The honest summary, because it prices both.

The bias-variance decomposition is the single most useful idea in applied statistics. Unbiasedness is not sacred: a slightly biased estimator with much lower variance can be much better. Ridge regression, shrinkage, and most machine learning methods are deliberate bias-for-variance trades. See [[18 - Model Selection and Machine Learning]].

**Consistency**: `θ̂ →p θ` as `n → ∞`. The estimator converges to the truth with enough data. This is the minimum bar. An inconsistent estimator does not improve with more data — it converges confidently to the wrong answer.

**Asymptotic normality**: `√n(θ̂ - θ) →d N(0, V)`. The basis of all standard errors and tests.

**Efficiency**: among estimators in some class, the one with smallest variance. The **Cramér-Rao bound** gives a floor on the variance of any unbiased estimator; an estimator hitting it is efficient.

## The sample mean as the template

The sample mean `X̄ₙ` demonstrates the whole pattern:

- Unbiased: `E[X̄ₙ] = μ`.
- Variance: `var[X̄ₙ] = σ²/n`. Precision improves linearly in `n`, so *standard* error improves as `√n`.
- Consistent by the WLLN.
- Asymptotically normal by the CLT.
- Its standard error is `s/√n` where `s²` is the sample variance.

Almost every estimator you meet later is either an average, a function of averages, or a solution to an equation involving averages — and inherits this structure. That is the whole point of the "smooth function model" that shows up in bootstrap theory.

## The plug-in (analog) principle

The general recipe for building an estimator:

1. Write the population quantity as a feature of the distribution — usually an expectation.
2. Replace the population expectation with the sample average.

Population mean `E[X]` → sample mean `(1/n)ΣXᵢ`. Population variance `E[(X-μ)²]` → `(1/n)Σ(Xᵢ - X̄)²`. Population covariance → sample covariance. Population regression coefficient `E[XX']⁻¹E[XY]` → `(X'X)⁻¹X'Y`, which is OLS.

This is called the **analog principle** or **method of moments**, and it is why OLS looks the way it does. See [[05 - Least Squares Mechanics]] and its generalization in [[12 - GMM and Minimum Distance]].

## Sample variance and degrees of freedom

Two versions of the sample variance:

```
σ̂² = (1/n)Σ(Xᵢ - X̄)²        biased downward
s²  = (1/(n-1))Σ(Xᵢ - X̄)²    unbiased
```

The `n-1` corrects for the fact that you estimated the mean from the same data, using up one "degree of freedom". In regression the analogous correction is `n - k` where `k` is the number of estimated coefficients. For large `n` the difference is negligible; for small `n` it matters.

## Maximum likelihood estimation

MLE is the other great estimation principle. It applies when you are willing to specify the *entire* probability distribution of the data up to a few parameters — a **parametric model**.

**Setup**: assume `X₁,...,Xₙ` are i.i.d. with density `f(x | θ)` for some unknown `θ`. The model is **correctly specified** if there is a `θ₀` with `f(x | θ₀)` equal to the true density.

**Likelihood function**: the joint density of the observed data viewed as a function of `θ`:
```
Lₙ(θ) = Πᵢ f(Xᵢ | θ)
```
Flip the usual reading: normally a density tells you which data values are likely given the parameter; the likelihood tells you which parameter values are compatible with the data you actually got.

**Log-likelihood** (what you actually maximize, because sums are easier than products and numerically better behaved):
```
ℓₙ(θ) = Σᵢ log f(Xᵢ | θ)
```

**MLE**: `θ̂ = argmax ℓₙ(θ)`.

*Worked example.* Exponential model `f(x | λ) = λ⁻¹exp(-x/λ)`. Then
```
ℓₙ(λ) = -n log λ - n X̄ₙ / λ
```
Setting the derivative to zero: `-n/λ + nX̄ₙ/λ² = 0`, so `λ̂ = X̄ₙ`. The MLE of the exponential mean is the sample mean. The second derivative at `λ̂` is `-n/X̄ₙ² < 0`, confirming a maximum.

*Worked example.* Bernoulli model `π(x | p) = pˣ(1-p)¹⁻ˣ`. Then `ℓₙ(p) = nX̄ₙ log p + n(1-X̄ₙ)log(1-p)`, and the MLE is `p̂ = X̄ₙ` — the sample frequency.

**Why MLE works**: define the expected log density `ℓ(θ) = E[log f(X | θ)]`. When the model is correctly specified, the true `θ₀` maximizes `ℓ(θ)`. The sample average log-likelihood `(1/n)ℓₙ(θ)` is the sample analog of `ℓ(θ)`, so its maximizer is the analog estimator of `θ₀`. MLE is the analog principle applied to a population maximization problem.

**Properties under correct specification**:

- Consistent.
- Asymptotically normal with variance equal to the inverse **Fisher information** `I(θ)⁻¹`, where `I(θ) = E[(∂ log f/∂θ)(∂ log f/∂θ)']`.
- Asymptotically efficient — it attains the Cramér-Rao bound. No other consistent, asymptotically normal estimator does better.
- **Invariant**: the MLE of `g(θ)` is `g(θ̂)`.

**The catch**: all of this depends on the model being right. Under misspecification the MLE converges to the parameter that minimizes the Kullback-Leibler divergence to the truth — a "pseudo-true" value, which may or may not be interesting — and the classic variance formula is wrong. The fix is a **sandwich** (robust) variance estimator, structurally identical to the heteroskedasticity-robust variance in regression. See [[06 - Standard Errors and Clustering]].

This is the core tradeoff between MLE and least squares: MLE is more efficient when you know the distribution, and can be badly wrong when you do not. Least squares assumes less and asks less.

## Confidence intervals

A **confidence interval** is a data-dependent set that covers the true parameter with a stated probability. The standard asymptotic 95% interval:

```
θ̂ ± 1.96 · s(θ̂)
```

where `s(θ̂)` is the standard error. The interpretation is about the *procedure*, not the particular interval: if you repeated the sampling many times, 95% of the intervals so constructed would contain the truth. It is **not** "there is a 95% probability that θ is in this specific interval" — the truth is fixed, the interval is random.

Hansen's practical advice throughout both books: report estimates with standard errors and confidence intervals, and discuss magnitudes. Confidence intervals convey precision; test decisions do not.

Intervals can also be built by **test inversion**: the set of parameter values not rejected by a test. For nonlinear parameters this is much better behaved than the naive `estimate ± 2 SE` interval. Hansen's example: for the peak of a wage-experience profile (`-50β₁/β₂`), the delta-method interval is `[29.8, 29.9]` while the test-inversion interval is `[29.1, 30.6]` — the first is badly too narrow, and the second is the one to trust. See [[08 - Hypothesis Testing and Confidence Intervals]].

## Testing in one paragraph

You have a **null hypothesis** `H₀` (usually "this coefficient is zero"), a test statistic whose distribution under `H₀` you know, and a **significance level** `α`. Reject when the statistic is extreme. **Type I error**: rejecting a true null (probability `α` by construction). **Type II error**: failing to reject a false null. **Power** = 1 - P(Type II) — the probability of catching a real effect. Full treatment in [[08 - Hypothesis Testing and Confidence Intervals]].

## Numerical optimization

Most estimators beyond OLS have no closed form and are computed by iterative optimization (Newton-Raphson, BFGS, Nelder-Mead). Practical consequences:

- Results can depend on starting values. Try several.
- Convergence warnings are not cosmetic. An unconverged optimizer produces meaningless standard errors.
- Non-concave likelihoods can have multiple local maxima. Grid search over a coarse parameter grid before trusting an optimizer.

Related:

- [[02 - Probability Foundations]]
- [[05 - Least Squares Mechanics]]
- [[07 - Asymptotic Theory]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[17 - Limited Dependent Variables]]
