---
title: Multivariate Regression and Factor Models
tags: [econometrics, systems, sur, pca, factor-models, dimension-reduction]
---

# Multivariate Regression and Factor Models

Source: Hansen, *Econometrics*, chapter 11.

Everything so far had one equation and one outcome. This note covers systems of equations, and the dimension-reduction tools — principal components and factor models — that let you handle a large number of correlated regressors.

## Systems of equations

A **multivariate regression** has `m` outcome variables observed for each unit:

```
Y = X̄'β + e,   E[e | X] = 0,   E[ee' | X] = Σ
```

`Y` is `m × 1`, and `Σ` is the `m × m` covariance of the errors *across equations* for the same unit. Examples: a demand system (one equation per good), a set of firm outcomes, a portfolio of asset returns.

Two things are new:

1. Errors are correlated across equations for the same unit — an unobserved shock to a household hits all its demand equations.
2. You may want **cross-equation restrictions** — Slutsky symmetry in a demand system, adding-up constraints, or a common coefficient across equations. These cannot be imposed equation by equation.

Estimating each equation separately by OLS is consistent. The question is whether you can do better.

## Seemingly Unrelated Regression (SUR)

Zellner (1962). Since errors are correlated across equations, GLS is more efficient than equation-by-equation OLS:

```
β̂_gls = ( Σᵢ X̄ᵢ'Σ⁻¹X̄ᵢ )⁻¹ ( Σᵢ X̄ᵢ'Σ⁻¹Yᵢ )
```

`Σ` is unknown, so use the feasible version with `Σ̂ = (1/n)Σ êᵢêᵢ'` from a first-pass OLS:

```
β̂_sur = ( Σᵢ X̄ᵢ'Σ̂⁻¹X̄ᵢ )⁻¹ ( Σᵢ X̄ᵢ'Σ̂⁻¹Yᵢ )
```

You can iterate — recompute residuals, recompute `Σ̂`, re-estimate — until convergence. **Iterated SUR equals the MLE under normality.**

Under conditional homoskedasticity, SUR is asymptotically more efficient than OLS (Theorem 11.5).

### The two cases where SUR gives you nothing

This is the practically important part, and it surprises people:

1. **Regressors are common across equations.** If every equation has the same `X`, then `β̂_sur = β̂_ols` exactly — not approximately, algebraically. The GLS weighting cancels.
2. **`Σ` is diagonal.** If errors are uncorrelated across equations, there is nothing to exploit.

So SUR only helps when the equations have *different* regressors and correlated errors. Hansen's framing: a model with common regressors is the unrestricted model, and a model with differing regressors is a restricted one — SUR gains come precisely from those exclusion restrictions.

Stata: `sureg`.

## Reduced rank regression

Sometimes theory says the `k × m` coefficient matrix `B` in `Y = B'X + C'Z + e` has **reduced rank** `r < min(k, m)`. Then `B = GA'` with `A` being `m × r` and `G` being `k × r` — the `k` regressors influence the `m` outcomes only through `r` linear combinations.

Anderson (1951), extended by Johansen (1995). The MLE under normality is found by concentration: given `G`, the rest is least squares; substituting back gives a concentrated likelihood whose maximizer is

```
Ĝ = generalized eigenvectors of X̃'Ỹ(Ỹ'Ỹ)⁻¹Y'X̃  with respect to  X̃'X̃
    corresponding to the r largest generalized eigenvalues
```

where tildes denote residuals after regressing on `Z` (Frisch-Waugh-Lovell again — see [[05 - Least Squares Mechanics]]). The maximized log-likelihood involves `Σⱼ log(1 - λ̂ⱼ)` over the retained eigenvalues, which is exactly the form of the Johansen cointegration test statistic. That is not a coincidence: **cointegration is a reduced-rank restriction on a VAR in differences**. See [[15 - Time Series]].

R package: `RRR`. No standard Stata command.

## Principal component analysis (PCA)

You have `k` correlated variables and want a few summary indices.

**Definition.** The first principal component is `U₁ = h₁'X` where `h₁` maximizes `var[h'X]` subject to `h'h = 1`. The `j`-th maximizes the same variance subject to being orthogonal to all previous ones.

**Theorem 11.8**: the principal components are `Uⱼ = hⱼ'X` where `hⱼ` is the eigenvector of `Σ = var[X]` associated with the `j`-th largest eigenvalue.

Equivalently, from the spectral decomposition `Σ = HDH'`, set `U = H'X`. Then `var[U] = D`, which is diagonal — the components are mutually uncorrelated — and the variance share of the `j`-th component is `dⱼ / tr(Σ)`.

**Scaling matters.** PCA is not invariant to the units of `X`. Standard practice is to scale each variable to mean zero and unit variance first, which makes `Σ` the correlation matrix. Stata's `pca` normalizes by default; R's `prcomp`/`princomp` do not. Check.

### Hansen's worked example (Kenyan first-grade test scores)

Seven section scores: word recognition, sentences, letters, spelling, addition, subtraction, multiplication. Eigenvalues of the sample correlation matrix:

| Component | Eigenvalue | Proportion |
| --- | --- | --- |
| 1 | 4.02 | 0.57 |
| 2 | 1.04 | 0.15 |
| 3 | 0.57 | 0.08 |
| 4 | 0.52 | 0.08 |
| 5 | 0.37 | 0.05 |
| 6 | 0.29 | 0.04 |
| 7 | 0.19 | 0.03 |

Weight vectors:

| | First | Second |
| --- | --- | --- |
| words | 0.41 | -0.32 |
| sentences | 0.32 | -0.49 |
| letters | 0.40 | -0.13 |
| spelling | 0.43 | -0.28 |
| addition | 0.38 | 0.41 |
| subtraction | 0.35 | 0.52 |
| multiplication | 0.33 | 0.36 |

Read it: the first component has all-positive, similar weights — it is essentially a **simple average** of the seven scores, and it explains 57% of the variance. That is a real justification for the practice of using `totalscore`. The second component weights the four literacy scores negatively and the three math scores positively — it is **verbal minus math**. Together they explain 72%.

This is what a good PCA reading looks like: not "we retained two components", but "the components turn out to mean *average performance* and *verbal-vs-math tilt*".

## Factor models

Closely related but a genuine statistical model rather than a decomposition.

**Single factor**: `X = λF + u`, with `λ` the **factor loadings** (`k × 1`), `F` a scalar **common factor**, `u` idiosyncratic error. The factor is individual-specific; the loadings are common across individuals.

**Multiple factor**: `X = ΛF + u` with `Λ` a `k × r` loading matrix and `F` an `r × 1` factor vector, normalized so `E[FF'] = I_r`.

Assuming `u` is mean zero, uncorrelated with `F`, with **diagonal** covariance `Ψ`, the correlation matrix satisfies

```
Σ = ΛΛ' + Ψ
```

`ΛΛ'` is called the **communality** (variance explained by the factors), `Ψ` the **uniqueness** (unexplained).

### Estimation by MLE

Under joint normality, `X ~ N(0, ΛΛ' + Ψ)` and the log-likelihood is

```
ℓₙ(Λ, Ψ) = -(nk/2)log(2π) - (n/2)log det(ΛΛ' + Ψ) - (n/2)tr((ΛΛ' + Ψ)⁻¹Σ̂)
```

No closed form; solved numerically. Note what Hansen points out about its structure: the likelihood depends on the data only through the sample correlation matrix `Σ̂`, and on the parameters only through `ΛΛ' + Ψ`. It is a measure of the match between the sample and model correlation matrices — so it is really a fitting criterion, not reliant on normality.

Stata: `factor, ml factors(r)`. R: `factanal(X, factors=r, rotation="none")`.

### Estimating the factors themselves

Two scoring methods:

```
Bartlett:    F̃ᵢ = (Λ̂'Ψ̂⁻¹Λ̂)⁻¹ Λ̂'Ψ̂⁻¹ Xᵢ
Regression:  F̄ᵢ = Λ̂'Σ̂⁻¹ Xᵢ
```

Bartlett is unbiased (for its idealized version); regression scoring is biased but has lower MSE. Hansen's recommendation is **regression scoring**, on precision grounds. The difference shrinks as `k` grows. In Stata, `predict` with the `bartlett` or `regression` option.

### Two warnings

**Rotation indeterminacy.** `Λ` and `F` are not separately identified: replacing `(Λ, F)` with `(ΛG, G'F)` for any orthonormal `G` gives an identical model. The MLE output is one arbitrary rotation among many. Hansen: "it is unwise to attribute meaning to the individual factor loading estimates." The factor *space* is identified; individual loadings are not.

**Choosing `r`.** No clear guideline. Two approaches: look for a gap between "large" and "small" eigenvalues in the PCA decomposition; or use the likelihood-ratio test of `r` factors against `k` factors (a by-product of the MLE) — rejection is evidence that `r` is too small.

## Approximate factor models

Chamberlain and Rothschild (1983). The MLE above requires `Ψ` diagonal — no correlation at all among idiosyncratic errors — which is rarely credible, and is computationally painful when `k` is large. The **approximate factor model** leaves `Ψ` unrestricted and estimates by least squares, treating the factors as unknown regressors:

```
min  (1/n) Σᵢ (Xᵢ - ΛFᵢ)'(Xᵢ - ΛFᵢ)
```

**Theorem 11.9**: with the normalization `n⁻¹ΣF̂ᵢF̂ᵢ' = I_r`, the solution is:

1. Let `D̂` be the first `r` eigenvalues and `Ĥ` the first `r` eigenvectors of `Σ̂`.
2. `Λ̂ = ĤD̂^{1/2}`.
3. `F̂ᵢ = D̂^{-1/2}Ĥ'Xᵢ`.

That is: **the factor estimates are the principal components, scaled by the eigenvalues.** Hence the name "principal-component method". It is computationally stable even in high dimensions.

**The key asymptotic subtlety.** For *fixed* `k`, `Λ̂` is **inconsistent** — the sample has `nk` observations but the model has `nr + kr - r(r+1)/2` parameters, so the parameter count grows with the sample and you should not expect consistency. The fix is an asymptotic framework where `k → ∞` as well: more variables carry more information about the unobserved factors. Under Assumption 11.1 (`λ_max(Ψ)` bounded, `λ_min(Λ'Λ) → ∞`), the `ΛΛ'` component dominates `Σ = ΛΛ' + Ψ` as `k` grows, and consistency follows (Bai 2003, jointly in `n, k`).

Stata: `factor, pcf factors(r)`, then `predict`.

## Factor-augmented regression

The applied payoff. You have `Y`, a few variables of interest `Z`, and a large block of correlated regressors `X`:

```
Y = F'β + Z'γ + e
X = ΛF + u
```

The idea: the influence of `X` on `Y` runs through a small number of common factors, so replace `k` regressors with `r` factors. This is dimension reduction. Typically `γ` is what you care about and the factors are controls — and since only the factor *space* is identified, **do not interpret `β`**.

Estimation is two-step: estimate the factors (MLE or principal components), then regress `Y` on `F̂` and `Z`.

Asymptotics mirror the approximate factor model: `β̂` has a probability limit but is **inconsistent for fixed `k`**,

```
β* = (I_r + D⁻¹Λ'ΨΛD⁻¹)⁻¹ β
```

which tends to `β` as `k → ∞`. So factor-augmented regression is consistent only when *both* `n` and the dimension of `X` are large. For asymptotic normality you need the stronger condition `k²/n → ∞` — the dimension of `X` must be large relative to `√n`.

This is the FAVAR / diffusion-index approach used in macro forecasting (Stock and Watson), and it is the natural alternative to lasso when your many regressors are highly correlated rather than sparse. Lasso assumes a few regressors matter; factor models assume they all matter a little through a common structure. Which is right is an empirical question. See [[18 - Model Selection and Machine Learning]].

## Practical guidance

- Use SUR only when equations have different regressors; otherwise it is literally OLS.
- Standardize before PCA, and say whether you did.
- Report the eigenvalue table, not just the number of components retained.
- Interpret components by looking at the weight vectors — and stop if they do not admit a clean reading.
- Never interpret individual factor loadings: rotation makes them meaningless.
- Factor-augmented regression needs large `k` *and* large `n`. With a small number of regressors it is inconsistent.

Related:

- [[05 - Least Squares Mechanics]]
- [[15 - Time Series]]
- [[18 - Model Selection and Machine Learning]]
- [[21 - Shrinkage and Model Averaging]]
