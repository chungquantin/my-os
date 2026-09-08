---
title: Bayesian Methods
tags: [econometrics, bayesian, priors, posterior, credible-sets, beginner]
---

# Bayesian Methods

Source: Hansen, *Probability and Statistics for Economists*, chapter 16.

Everything else in these notes is **frequentist** (or **classical**): the parameter is a fixed unknown, the data are random, and probability statements are about the sampling distribution of estimators. Bayesian statistics reverses this: the parameter is treated as **random**, and inference is conditional on the data you actually observed.

## The setup in one paragraph

You specify a probability model `f(x | θ)` — the same parametric model that maximum likelihood uses ([[03 - Statistical Inference Foundations]]) — and additionally a **prior** density `π(θ)` describing beliefs about `θ` before seeing data. Together these give a joint distribution for data and parameters. Bayes' rule turns it around into the **posterior** `π(θ | X)`, the distribution of `θ` given the data. Everything else — point estimates, intervals, tests — is read off the posterior.

## The machinery

Joint density of data and parameter:
```
f(x, θ) = f(x | θ) π(θ)
```

**Marginal density** (also called the marginal likelihood, or the evidence) — integrate the parameter out:
```
m(x) = ∫_Θ f(x, θ) dθ = ∫_Θ Lₙ(x | θ) π(θ) dθ
```

**Posterior density** by Bayes' rule:
```
π(θ | X) = Lₙ(X | θ) π(θ) / m(X)
```

Read it as: *posterior ∝ likelihood × prior*. The denominator is just a normalizing constant, which is why most Bayesian algebra works with proportionality.

Computing `m(X)` is often the hard part. In textbook models it is analytic; beyond them it requires numerical methods (MCMC, and its modern variants), which is why Bayesian econometrics is a computational field.

*Small example.* `X ~ N(θ, 1)` with prior `θ ~ N(0, 1)`. Then `(X, θ)` is bivariate normal with covariance `[[2,1],[1,1]]`, the marginal of `X` is `N(0,2)`, and the posterior is `θ ~ N(X/2, 1/2)`. Notice: the posterior mean `X/2` shrinks the observation halfway toward the prior mean.

## The Bayes estimator

The standard point estimate is the **posterior mean**:
```
θ̂_Bayes = ∫_Θ θ π(θ | X) dθ = ∫ θ Lₙ(X|θ)π(θ)dθ / m(X)
```

Why this and not something else? Because of **Bayes risk**. Given a loss `ℓ(T, θ)`, the Bayes risk of an estimator `T` is `R(T | X) = ∫ ℓ(T, θ)π(θ|X)dθ` — average loss weighted by the posterior. The optimal estimator minimizes it:

- **Quadratic loss** `ℓ = (T - θ)'(T - θ)`: the minimizer is the **posterior mean**. (Expand the risk; the first-order condition gives `T = θ̂_Bayes` directly.)
- **Absolute loss** `ℓ = |T - θ|`: the minimizer is the **posterior median**.

Posterior mean is the common choice, posterior median the second most common.

## Choosing a prior

Hansen identifies three philosophies, and it is worth knowing which one an applied paper is using:

- **Subjectivist** — the prior should reflect the analyst's actual beliefs, informed by prior studies. Useful for personal decisions, business decisions, and policy where the state of knowledge is well articulated. Awkward for scientific discourse, where the reader may not share the author's beliefs.
- **Objectivist** — the prior should be non-informative, so the posterior reflects the data. The scientific stance. The difficulty is that "non-informative" has multiple non-equivalent definitions.
- **Shrinkage** — the prior is a *regularization device*, chosen to improve estimation precision. Deliberately similar in spirit to Stein-Rule estimation ([[21 - Shrinkage and Model Averaging]]), and more flexible than Stein theory in what it can be applied to. Downside: no guidelines for how much shrinkage.

Priors are usually parametric, `π(θ | α)`, with `α` controlling **centering** and **spread**. The parameter space constrains the family:

| Parameter lives in | Natural prior |
| --- | --- |
| `[0,1]` (a probability) | Beta |
| `ℝ₊` (a variance or precision) | Gamma or inverse-gamma |
| `ℝ` (a mean) | Normal |

An objectivist sets the prior center at the middle of the parameter space and the spread large. A subjectivist matches both to prior knowledge. A shrinkage Bayesian centers on "default" values and uses the spread to tune how hard the estimate is pulled toward them.

## Conjugate priors

A prior is **conjugate** to a likelihood if prior and posterior belong to the same parametric family. This makes the posterior analytic and the estimates easy to compute.

Two conditions produce conjugacy: the likelihood, viewed as a function of `θ`, must be proportional to `π(θ|α)` for some `α`; and products of two densities from the family must stay in the family.

Useful product rules:

```
beta(α₁,β₁) × beta(α₂,β₂) ∝ beta(α₁+α₂, β₁+β₂)
gamma(α₁,β₁) × gamma(α₂,β₂) ∝ gamma(α₁+α₂-1, β₁+β₂)
N(μ₁,1/ν₁) × N(μ₂,1/ν₂) ∝ N(μ̄, 1/ν̄),   μ̄ = (ν₁μ₁+ν₂μ₂)/(ν₁+ν₂),  ν̄ = ν₁+ν₂
```

Note the normal is parameterized by **precision** `ν = 1/σ²` rather than variance — standard in Bayesian work because it makes the algebra additive.

### Bernoulli sampling (the cleanest example)

Likelihood `Lₙ(X|p) = p^{Sₙ}(1-p)^{n-Sₙ}` with `Sₙ = ΣXᵢ`. Beta prior `π(p|α,β) ∝ p^{α-1}(1-p)^{β-1}`. Product:

```
π(p | X) = beta(Sₙ + α, n - Sₙ + β)
```

Posterior mean:
```
p̂_Bayes = (Sₙ + α)/(n + α + β)      versus      p̂_mle = Sₙ/n
```

Read it: the Bayes estimator acts as if you had observed `α` extra successes and `β` extra failures. The prior is worth `α + β` pseudo-observations.

Hansen's illustration: prior `beta(5,5)` (centered at 0.5), MLE `p̂ = 0.8`. Posterior mean is 0.65 at `n = 10` and 0.725 at `n = 30` — the estimate is shrunk toward the prior, and the shrinkage weakens as data accumulate.

### Normal sampling

Writing the model in terms of precision `ν = σ⁻²`:

**Mean with known precision.** Prior `μ ~ N(μ̄, 1/ν̄)` gives posterior `N( (nνX̄ₙ + ν̄μ̄)/(nν + ν̄), 1/(nν + ν̄) )`, so

```
μ̂_Bayes = (nν X̄ₙ + ν̄ μ̄) / (nν + ν̄)
```

A weighted average of the sample mean and the prior mean. Weight goes to the sample mean when `n` is large or the prior is diffuse. Interpretation of `ν̄`: it is equivalent to adding `N = ν̄/ν` extra observations all equal to `μ̄`.

**Precision with known mean.** Gamma prior `ν ~ gamma(α,β)` gives posterior `gamma(n/2 + α, nσ̃²/2 + β)` and

```
σ̂²_Bayes = (nσ̃²_mle + 2ασ̄²)/(n + 2α)
```

Again a weighted average, with `2α` interpretable as prior pseudo-observations.

**Both unknown.** The conjugate prior is the **NormalGamma** distribution `NormalGamma(μ̄, λ, α, β)`, and the posterior is NormalGamma too:

```
π(μ, ν | X) = NormalGamma( (nX̄ₙ + λμ̄)/(n+λ),  n+λ,  (n-1)/2 + α,  nσ̃²_mle/2 + β )

μ̂_Bayes = (nX̄ₙ + λμ̄)/(n + λ)
σ̂²_Bayes = ((n-1)s² + 2ασ̄²)/(n - 1 + 2α)
```

`λ` is the number of pseudo-observations backing the prior on the mean; `2α` the number backing the prior on the variance. The marginal posterior for `μ` is a scaled Student t with `n - 1 + 2α` degrees of freedom. Setting `λ = α = β = 0` recovers exactly the classical sampling distributions in the normal model — the diffuse-prior limit of the Bayesian answer is the frequentist answer.

## Credible sets

The Bayesian interval estimator.

**Definition.** A `1-η` **credible interval** is `C = [L, U]` with `P[θ ∈ C | X] = 1 - η`, computed from the posterior.

The contrast with a confidence interval is exact and worth stating carefully:

| | Confidence interval | Credible interval |
| --- | --- | --- |
| What is random | The interval | The parameter |
| What is fixed | The parameter | The interval (conditional on data) |
| Probability statement | Over repeated samples | Over the posterior |
| "95% probability θ is in here" | **False** | **True** (given the prior) |

The everyday misinterpretation of a confidence interval is in fact the correct interpretation of a credible interval. That is the appeal — and the cost is that the statement is conditional on a prior the reader may not accept.

**HPD intervals.** The standard construction is the **highest posterior density** set: the interval `C` with `∫_C π(θ|X)dθ = 1-η` such that the posterior density is higher everywhere inside than outside. HPD intervals have the **shortest length** among all credible sets of the same level. For a symmetric unimodal posterior the HPD set is symmetric about the mode; in general it is found numerically by solving `f(L) - f(Q(1-η+F(L))) = 0` for the lower endpoint.

*Normal mean example.* The marginal posterior is scaled Student t, symmetric, so
```
C = [ μ̂_Bayes ± q_{1-η/2} / √((n+λ)ν̂_Bayes) ]
```
with `q` a Student t quantile on `n - 1 + 2α` degrees of freedom. When `λ` and `α` are small this is close to the classical interval.

*Normal variance example.* The posterior is `χ²`-based and asymmetric, so the credible set is asymmetric too. A nice property: transformations apply directly to the endpoints. With `L = 0.17, U = 0.96` for the precision, the interval for `σ² = 1/ν` is `[1/.96, 1/.17] = [1.04, 5.88]` and for `σ` is `[1.02, 2.43]`. Frequentist intervals do not transform this cleanly.

In Hansen's illustrations, the Bayes credible interval is **shorter** than the classical confidence interval because the prior adds information.

## Bayesian hypothesis testing

Very different from Neyman-Pearson. There is no null and no alternative — models are treated symmetrically, and you pick the one with the highest probability of being true.

Given models `H₁,...,H_J` with prior probabilities `πⱼ` summing to one and marginal likelihoods `mⱼ(X)`:

```
πⱼ(X) = P[Hⱼ | X] = πⱼ mⱼ(X) / Σᵢ πᵢ mᵢ(X)
```

A **Bayes test** selects the model with the largest `πⱼmⱼ(X)`. With equal prior probabilities, that is the model with the largest marginal likelihood.

For two models:

```
prior odds     = π₂/π₁
posterior odds = π₂(X)/π₁(X)
Bayes Factor   = m₂(X)/m₁(X)
```

Select `H₂` when the posterior odds exceed 1 — equivalently when prior odds × Bayes Factor > 1, or (with equal priors) when the Bayes Factor exceeds 1.

No significance levels, no p-values, no asymmetry between hypotheses. Bayesian econometricians generally do less hypothesis testing than frequentists.

## Frequentist properties of Bayes estimators

You can evaluate a Bayes estimator by classical criteria. In the normal model with `μ̂_Bayes = (nX̄ₙ + λμ̄)/(n+λ)`:

**Theorem 16.1**:
```
E[μ̂_Bayes] = (nμ + λμ̄)/(n + λ)
bias        = λ(μ̄ - μ)/(n + λ)
var         = σ²/(n + λ)
μ̂_Bayes ~ N( (nμ + λμ̄)/(n+λ),  σ²/(n+λ) )
```

So a Bayes estimator with `λ > 0` has **lower variance** than the sample mean, and **bias** whenever the prior is not centered on the truth. That is the bias-variance trade again, arrived at from a different direction — which is exactly why the shrinkage interpretation of priors is coherent.

**Asymptotically**, if the MLE is asymptotically normal and the prior is fixed with support containing the true parameter, the posterior converges to a normal distribution and the standardized posterior mean converges to a normal random vector. In large samples Bayesian and MLE inference coincide: the data swamp the prior.

## Advantages and disadvantages

**Advantages**

- A single coherent framework for estimation and inference. Awkward sampling distributions never arise — you compute posterior integrals instead.
- Induces shrinkage similar to James-Stein, which improves precision over the MLE.
- Naturally handles nuisance parameters (integrate them out) and gives exact finite-sample statements.

**Disadvantages** (Hansen is direct about these)

- Computationally burdensome beyond textbook models — closed forms disappear and you need numerical methods. He notes Bayesian econometrics has advanced enough computationally that this "should not be viewed as a barrier".
- **Results depend on the prior by construction.** Most applications choose priors for mathematical convenience, making results a by-product of an arbitrary choice. When the data are uninformative, the posterior resembles the prior — "inferential statements are about the prior, not about the actual world".
- Built for **parametric** models. Bayesian methods are essentially an analog of maximum likelihood, not of the method of moments. Most econometric models are semiparametric or nonparametric.
- Hard to robustify against unmodeled dependence (clustering, serial correlation) or misspecification — precisely the things [[06 - Standard Errors and Clustering]] exists to handle.

That last pair is why frequentist methods dominate applied microeconometrics while Bayesian methods are common in macro (DSGE estimation), structural IO (BLP-style demand), and forecasting, where fully specified parametric models are the norm anyway.

Further reading Hansen recommends: Koop, Poirier and Tobias (2007) for Bayesian econometrics; Lehmann and Casella (1998) and van der Vaart (1998) for theory.

Related:

- [[03 - Statistical Inference Foundations]]
- [[08 - Hypothesis Testing and Confidence Intervals]]
- [[21 - Shrinkage and Model Averaging]]
- [[18 - Model Selection and Machine Learning]]
