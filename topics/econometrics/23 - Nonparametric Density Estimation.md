---
title: Nonparametric Density Estimation
tags: [econometrics, nonparametric, kernel, density, bandwidth]
---

# Nonparametric Density Estimation

Source: Hansen, *Probability and Statistics for Economists*, chapter 17.

Estimating a whole density `f(x)` rather than a mean or a regression function. This is the cleanest setting in which to learn the bias-variance mechanics that also govern kernel regression ([[16 - Nonparametrics, Quantiles, and RDD]]) — the algebra is nearly identical but simpler here.

Densities can take any shape, so they cannot be described by a finite parameter vector. They are inherently nonparametric.

## From histogram to kernel

**Histogram.** Split the range into `B` bins of width `w`, count `nⱼ` observations in bin `j`:
```
f̂(x) = nⱼ / (nw)     for x in bin j
```
The scaling makes the rectangles' areas sum to one, so it is a valid density.

Hansen's illustration uses hourly wages for Asian women in the March 2009 CPS (`n = 1149`). With bin width $10 the histogram is too crude to tell whether $11 or $19 wages are more common. With bin width $1 it is visibly noisy. Neither is satisfactory — hence smoothing.

**Kernel density estimator** (Rosenblatt 1956, Parzen 1962):
```
f̂(x) = (1/nh) Σᵢ K((Xᵢ - x)/h)
```

Two improvements over the histogram: the window is centered at the evaluation point `x` rather than at fixed bin edges, and observations are weighted by distance rather than counted equally.

`K` is the **kernel** (a weight function), `h > 0` the **bandwidth**.

## Kernel functions

**Definition.** A second-order kernel satisfies `0 ≤ K(u) ≤ K̄ < ∞`, `K(u) = K(-u)`, `∫K(u)du = 1`, and finite absolute moments. A **normalized** kernel additionally has `∫u²K(u)du = 1`.

| Kernel | Formula | `R_K` | `C_K` |
| --- | --- | --- | --- |
| Rectangular | `1/(2√3)` for `\|u\| < √3` | `1/(2√3)` | 1.064 |
| Gaussian | `(2π)^{-1/2}exp(-u²/2)` | `1/(2√π)` | 1.059 |
| Epanechnikov | `(3/4√5)(1 - u²/5)` for `\|u\| < √5` | `3√5/25` | 1.049 |
| Triangular | `(1/√6)(1 - \|u\|/√6)` for `\|u\| < √6` | `√6/9` | 1.052 |
| Biweight | `(15/16√7)(1 - u²/7)²` for `\|u\| < √7` | `5√7/49` | 1.050 |

`R_K = ∫K(u)²du` is the kernel's **roughness**.

The kernel density estimator is a valid density: non-negative, and `∫f̂(x)dx = 1` by change of variables.

Hansen advises **against the rectangular kernel** — it produces discontinuous, visibly erratic estimates. Gaussian, Epanechnikov and Biweight give very similar answers in practice. Gaussian is convenient because it is infinitely differentiable and strictly positive everywhere (useful if you need `1/f̂(x)`).

## Bias

```
E[f̂(x)] = ∫ K(u) f(x + hu) du
        = f(x) + (1/2) f''(x) h² + o(h²)
```

Second line by a second-order Taylor expansion, using `∫uK(u)du = 0` and `∫u²K(u)du = 1`.

**Theorem 17.1**: if `f` is continuous near `x`, `E[f̂(x)] = f(x) + o(1)`; if `f''` is continuous, `E[f̂(x)] = f(x) + (1/2)f''(x)h² + o(h²)`.

The bias is `½f''(x)h²` — proportional to the **curvature** of the true density. Consequences worth internalizing:

- At a **peak** (`f'' < 0`) the estimator is biased **downward** — peaks get flattened.
- In a **valley or tail** (`f'' > 0`) it is biased **upward** — troughs get filled in.

This is **smoothing bias**, and it is why a kernel density estimate of a bimodal density has attenuated modes.

## Variance

```
V_f̂ = var[f̂(x)] = (1/nh²) var[K((X-x)/h)]
    = f(x)R_K/(nh) + o(1/(nh))
```

**Theorem 17.2.** Read the leading term: variance is inversely proportional to `nh`, which is the **effective sample size** — the number of observations in the smoothing window. It rises with the height of the density and with kernel roughness.

Standard errors: `V̂_f̂(x)^{1/2}` using either the finite-sample form
```
V̂_f̂(x) = (1/(n-1))( (1/nh²)ΣK((Xᵢ-x)/h)² - f̂(x)² )
```
or the asymptotic form `V̂_f̂(x) = f̂(x)R_K/(nh)`.

## The bias-variance trade-off, explicitly

**Integrated mean squared error** measures precision over all `x`:
```
IMSE = ∫ E[(f̂(x) - f(x))²] dx
     = (1/4)R(f'')h⁴ + R_K/(nh) + o(h⁴) + o(1/(nh))
```

where `R(f'') = ∫(f''(x))²dx` is the roughness of the second derivative. The leading terms give the **AIMSE**:

```
AIMSE = (1/4)R(f'')h⁴ + R_K/(nh)
```

Squared bias **increases** in `h`; variance **decreases** in `h`. Minimizing gives

```
h₀ = ( R_K / R(f'') )^{1/5} · n^{-1/5}
```

and at that bandwidth `AIMSE ~ n^{-4/5}`, so `f̂` converges at rate `n^{-2/5}` — slower than the parametric `n^{-1/2}`. Nonparametric problems are harder; you need more data.

**Hansen's warning about `h₀ ~ n^{-1/5}`**: a common error is to read the rate as a rule and set `h = n^{-1/5}`. That is wrong and can be a large mistake in practice. **The constant matters as much as the rate.**

## The optimal kernel

The kernel enters the AIMSE only through `R_K`, so the AIMSE-optimal kernel is the one minimizing roughness. By a calculus-of-variations argument (Lagrangian on `∫K = 1` and `∫u²K = 1`, giving a truncated quadratic), that is the **Epanechnikov** kernel (Hodges-Lehmann 1956; Epanechnikov 1969) — **Theorem 17.4**.

But the efficiency loss from using another kernel is tiny:

```
( AIMSE₀(Gaussian)/AIMSE₀(Epanechnikov) )^{1/2} = ( R_K(Gauss)/R_K(Epan) )^{2/5} ≈ 1.02
```

**2%.** Kernel choice barely matters; bandwidth choice matters enormously. That is the single most useful fact in this chapter.

## Bandwidth selection

### Reference (Silverman) rule

Compute `h₀` under the simplifying assumption that `f` is normal. Using `R(φ'') = 3/(8√π)σ^{-5}`:

```
h_r = σ C_K n^{-1/5},    C_K = (8√π R_K/3)^{1/5}
```

For the Gaussian kernel `C_K = (4/3)^{1/5} ≈ 1.059`, giving the familiar

```
h_r = 1.06 σ n^{-1/5}
```

`C_K` ranges only from about 1.05 (Epanechnikov) to 1.06 (rectangular), so **the constant 1.06 works for any kernel with unit variance.**

Feasible versions replace `σ`:
```
h_r = 1.06 s n^{-1/5}          sample standard deviation
h_r = 1.06 σ̃ n^{-1/5}          robust scale estimate
h_r = 0.9 σ̃ n^{-1/5}           Silverman's Rule of Thumb
```

Silverman observed that 1.06 oversmooths for thick-tailed or bimodal densities and recommended 0.9 based on simulation. The 0.9 version is what most packages implement.

### Sheather-Jones bandwidth

The reference rule plugs in a normal value for the unknown `R(f'')`. Better: estimate it nonparametrically. Sheather and Jones (1991) solve for `h` in

```
h = ( R_K / Ŝ₂(b̄₂(h)) )^{1/5} n^{-1/5}
```

where `Ŝ₂` is a kernel estimate of `R(f'')` whose own pilot bandwidth `b̄₂(h) = 1.357(S̃₂/S̃₃)^{1/7}h^{5/7}` is tied back to `h`. Solved numerically by Newton-Raphson; for the Gaussian kernel `h = 0.776 / Ŝ₂(b̄₂(h))^{1/5} · n^{-1/5}`.

Theory and simulation both show it outperforms the reference rule. **Hansen's stated preference is Sheather-Jones.**

### The three compared on real data

CPS Asian women, `n = 1149`, `s = 20.6`, robust `σ̃ = 14.0`, `n^{-1/5} = 0.24`:

| Rule | Bandwidth |
| --- | --- |
| Gaussian optimal `1.06·s·n^{-1/5}` | 5.34 |
| Silverman rule-of-thumb `0.9·σ̃·n^{-1/5}` | 3.08 |
| Sheather-Jones | 2.14 |

A factor of 2.5 between the extremes. The Gaussian-optimal estimate is visibly oversmoothed — it understates the main mode and overstates the left tail. The rule-of-thumb and Sheather-Jones estimates are similar; Hansen picks Sheather-Jones "because my preference leans towards detail".

Contrast that with the *kernel* comparison at fixed bandwidth: Gaussian and Epanechnikov estimates are nearly indistinguishable, and only the rectangular kernel looks different (erratic, non-smooth).

### Confidence intervals and undersmoothing

At the AIMSE-optimal bandwidth the bias is the same order as the standard error, so `f̂(x) ± 1.96·se` is **miscentered** and undercovers. The standard fixes are **undersmoothing** (deliberately use `h` smaller than optimal so bias is asymptotically negligible) or explicit bias correction with a robust variance. The same issue drives modern RDD practice — see [[16 - Nonparametrics, Quantiles, and RDD]].

## Practical recommendations

Hansen's, condensed:

1. **Try several bandwidths and use judgment.** Plot the estimates, compare, and choose based on the evidence and your purpose. If your package offers only one rule, perturb it by 20-30% and look at the results.
2. Use roughly **100 grid points** for a plot; more if the density has a steep section.
3. Do not use a kernel density estimator when the variable is close to discrete — Hansen's rule: **at least 50 distinct values** in the data.
4. **Minimum sample size around `n = 100`**, and even then precision may be poor. The `n^{-2/5}` rate is unforgiving.
5. Pick Gaussian or Epanechnikov and stop thinking about it. Avoid rectangular.

Related:

- [[02 - Probability Foundations]]
- [[16 - Nonparametrics, Quantiles, and RDD]]
- [[07 - Asymptotic Theory]]
