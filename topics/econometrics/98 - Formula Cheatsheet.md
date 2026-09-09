---
title: Formula Cheatsheet
tags: [econometrics, cheatsheet, formulas, reference]
---

# Formula Cheatsheet

Every formula from these notes in one place, grouped by topic, each with what it is for. Companion to [[99 - Econometrics Glossary]] (terms) and [[00 - Econometrics Hub]] (concepts).

Reading key: `→p` converges in probability, `→d` converges in distribution, hat = estimated from sample, prime = transpose, `1{A}` = 1 if A true else 0.

---

## 1. Probability

See [[02 - Probability Foundations]].

### Expectation and variance

```
E[X] = Σ x·π(x)                        discrete
E[X] = ∫ x·f(x) dx                     continuous
E[aX + bY + c] = aE[X] + bE[Y] + c     linearity — always true, no independence needed
var[X] = E[(X - E[X])²] = E[X²] - (E[X])²
var[aX + b] = a² var[X]                shifting doesn't change spread
sd[X] = √var[X]                        same units as X — the interpretable one
```

### Two variables

```
cov(X,Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
corr(X,Y) = cov(X,Y) / (sd[X]·sd[Y])                     ∈ [-1, 1]
var[X + Y] = var[X] + var[Y] + 2cov(X,Y)
f(x,y) = f(x)f(y)                                        independence
```

Independence ⟹ zero covariance. Zero covariance ⟹̸ independence.

### Shape

```
skew  = E[(X-μ)³]/σ³
κ     = E[(X-μ)⁴]/σ⁴          kurtosis; = 3 for normal
```

CPS `wage` has `κ ≈ 30` — the reason classical standard errors fail badly there.

### Conditional expectation (the CEF)

```
m(x) = E[Y | X = x]                             the CEF
Y = m(X) + e,   E[e | X] = 0                    CEF decomposition — an identity, not an assumption
E[E[Y|X]] = E[Y]                                law of iterated expectations
E[E[Y|X,Z] | X] = E[Y|X]                        LIE, general form
E[g(X)Y | X] = g(X)·E[Y|X]                      conditioning theorem
var[Y] = var[m(X)] + E[var[Y|X]]                variance decomposition
m = argmin_g E[(Y - g(X))²]                     CEF is the best predictor under squared loss
```

Auto-consequences of `E[e|X] = 0`: `E[e] = 0`, `E[h(X)e] = 0` for any `h`, `cov(X,e) = 0`.

### Limit theorems

```
X̄ₙ →p E[X]                          WLLN (needs E|X| < ∞)
√n(X̄ₙ - μ) →d N(0, σ²)              CLT (needs var[X] = σ² < ∞)
```

The `√n` rate: **4× the data to halve a standard error.**

### Inequalities

```
P[X > a] ≤ E[X]/a                    Markov (X ≥ 0)
P[|X - μ| > a] ≤ var[X]/a²           Chebyshev
E[g(X)] ≥ g(E[X])                    Jensen (g convex) — why E[log Y] ≠ log E[Y]
|E[XY]| ≤ √(E[X²]E[Y²])              Cauchy-Schwarz
P[∪Aⱼ] ≤ Σ P[Aⱼ]                     Boole — basis of Bonferroni
```

### Convergence tools

```
Zₙ →d Z, g continuous  ⟹  g(Zₙ) →d g(Z)              continuous mapping
Zₙ →d Z, Cₙ →p c       ⟹  Zₙ + Cₙ →d Z + c,  CₙZₙ →d cZ    Slutsky
```

---

## 2. Estimation basics

See [[03 - Statistical Inference Foundations]].

```
bias(θ̂) = E[θ̂] - θ
MSE(θ̂)  = E[(θ̂ - θ)²] = bias² + variance          the honest summary
θ̂ →p θ                                             consistency
√n(θ̂ - θ) →d N(0, V)                               asymptotic normality
```

### Sample moments

```
X̄  = (1/n) Σ Xᵢ
var[X̄] = σ²/n                        so se(X̄) = σ/√n
σ̂² = (1/n) Σ(Xᵢ - X̄)²                biased down
s² = (1/(n-1)) Σ(Xᵢ - X̄)²            unbiased
```

### Maximum likelihood

```
Lₙ(θ) = Πᵢ f(Xᵢ | θ)                         likelihood
ℓₙ(θ) = Σᵢ log f(Xᵢ | θ)                     log-likelihood (maximize this)
θ̂ = argmax ℓₙ(θ)
ℓ(θ) = E[log f(X|θ)],  maximized at θ₀       why MLE works (analog principle)
S(θ) = ∂ log f(X|θ)/∂θ                       score;  E[S(θ₀)] = 0
I(θ) = E[S S']= -E[∂²log f/∂θ∂θ']            Fisher information
√n(θ̂ - θ) →d N(0, I(θ)⁻¹)                    asymptotically efficient
V_sandwich = I⁻¹ · E[SS'] · I⁻¹              robust/QMLE variance under misspecification
```

Worked MLEs:

```
Exponential  f(x|λ) = λ⁻¹e^{-x/λ}      →  λ̂ = X̄ₙ
Bernoulli    π(x|p) = pˣ(1-p)^{1-x}    →  p̂ = X̄ₙ
```

### Confidence interval

```
θ̂ ± 1.96 · s(θ̂)                              asymptotic 95%
```

---

## 3. Linear projection and OLS algebra

See [[04 - Conditional Expectation and Projection]], [[05 - Least Squares Mechanics]].

```
β = E[XX']⁻¹ E[XY]                            population linear projection
e = Y - X'β,  E[Xe] = 0                       projection error — uncorrelated, NOT mean-independent
```

### The estimator

```
SSE(b) = Σ (Yᵢ - Xᵢ'b)²
β̂ = (X'X)⁻¹X'Y = (Σ XᵢXᵢ')⁻¹(Σ XᵢYᵢ)         OLS = sample analog of the projection
Ŷᵢ = Xᵢ'β̂                                     fitted value
êᵢ = Yᵢ - Ŷᵢ                                  residual (≠ error e)
```

### Always-true algebra (with an intercept)

```
Σ êᵢ = 0
Σ Xᵢêᵢ = 0                                    orthogonality — arithmetic, not a diagnostic
Ȳ = X̄'β̂                                       line passes through the means
```

### Geometry

```
P = X(X'X)⁻¹X'          projection matrix, Ŷ = PY
M = I - P               annihilator, ê = MY
PP = P,  MM = M,  PM = 0
Y'Y = Ŷ'Ŷ + ê'ê         Pythagoras
```

### Fit

```
R² = 1 - SSE/TSS
R̄² = 1 - (SSE/(n-k)) / (TSS/(n-1))            adjusted
```

### Leverage and influence

```
hᵢᵢ = [P]ᵢᵢ ∈ [0,1],  Σ hᵢᵢ = k               average leverage = k/n
ẽᵢ = êᵢ / (1 - hᵢᵢ)                           leave-one-out prediction residual
β̂₍₋ᵢ₎ = β̂ - (X'X)⁻¹Xᵢ ẽᵢ                      leave-one-out coefficient
E[êᵢ²] = (1 - hᵢᵢ)σᵢ²                          why residuals are "too small"
```

### Error variance

```
σ̂² = (1/n) Σ êᵢ²                  biased down
s²  = (1/(n-k)) Σ êᵢ²             unbiased under homoskedasticity
σ̄²  = (1/n) Σ ẽᵢ²                 estimates out-of-sample MSFE (= LOO cross-validation)
MSFEₙ = σ² + E[X'ₙ₊₁ V_β̂ Xₙ₊₁]
```

### Frisch-Waugh-Lovell

For `Y = X₁'β₁ + X₂'β₂ + e`:

```
Ỹ  = residual of Y on X₂
X̃₁ = residual of X₁ on X₂
β̂₁ = (X̃₁'X̃₁)⁻¹ X̃₁'Ỹ                          identical to full-regression β̂₁
```

The precise meaning of "controlling for `X₂`". Engine behind fixed effects and partialling-out lasso.

### Omitted variable bias

```
long:   Y = X₁'β₁ + X₂'β₂ + e
short:  Y = X₁'γ₁ + u
γ₁ = β₁ + Γβ₂          where Γ = projection coef of X₂ on X₁

bias = (relation of omitted to included) × (effect of omitted)
```

Signs: `Γ > 0, β₂ > 0` ⟹ short regression **overstates**.

### Functional forms

```
Y on X            β₁ = unit change in Y per unit X
log Y on X        ≈ 100·β₁ percent change in Y per unit X   (exact: 100(e^{β₁} - 1))
log Y on log X    β₁ = elasticity (% per %)
Y on log X        β₁/100 = unit change in Y per 1% change in X
```

Regression derivative with interactions: `∂m/∂x₁` depends on the other variables — report at meaningful values.

---

## 4. Variance, standard errors, clustering

See [[06 - Standard Errors and Clustering]].

### The master sandwich

```
V_β̂ = (X'X)⁻¹ (X'DX) (X'X)⁻¹        D = diag(σ₁²,...,σₙ²)
```

### Homoskedastic (software default — usually wrong)

```
V⁰ = (X'X)⁻¹σ²        V̂⁰ = (X'X)⁻¹s²
s(β̂ⱼ) = s·√[(X'X)⁻¹]ⱼⱼ
```

Bias when heteroskedastic: ratio of true to expected classical variance ≈ `κ` (kurtosis). `κ = 3` normal, `κ ≈ 30` for CPS wages.

### Heteroskedasticity-robust (HC)

```
HC0 = (X'X)⁻¹ ( Σ XᵢXᵢ'êᵢ² ) (X'X)⁻¹
HC1 = (n/(n-k)) · HC0                                    Stata `, r`
HC2 = (X'X)⁻¹ ( Σ XᵢXᵢ'êᵢ²/(1-hᵢᵢ) ) (X'X)⁻¹            unbiased under homosk.
HC3 = (X'X)⁻¹ ( Σ XᵢXᵢ'êᵢ²/(1-hᵢᵢ)² ) (X'X)⁻¹           conservative; jackknife
ordering:  HC0 < HC2 < HC3
```

Hansen prefers HC2 / HC3. They diverge from HC1 exactly when some `hᵢᵢ` is near 1.

### Cluster-robust

```
Ω̂ = Σ_g ( Σ_i Xᵢg êᵢg )( Σ_ℓ X_ℓg ê_ℓg )'        sum WITHIN cluster first, then outer product
V̂ = a_n (X'X)⁻¹ Ω̂ (X'X)⁻¹
a_n = ((n-1)/(n-k)) · (G/(G-1))                   Stata's adjustment
```

### Moulton inflation factor

```
V = (X'X)⁻¹σ² (1 + ρ(N - 1))
```

Equal clusters of size `N`, within-cluster correlation `ρ`, cluster-level regressor. `N = 48, ρ = 0.25` ⟹ variance inflated ~12×, standard errors ~3×.

**Effective sample size for clustered inference is `G`, not `n`.**

### HAC (time series)

```
uₜ = Xₜeₜ,   Γ(ℓ) = E[uₜ₋ℓuₜ']
Ω = Σ_{ℓ=-∞}^{∞} Γ(ℓ)                            long-run variance
Ω̂_M = Σ_{ℓ=-M}^{M} Γ̂(ℓ)                          truncated — may be NEGATIVE definite
Ω̂_nw = Σ_{ℓ=-M}^{M} (1 - |ℓ|/(M+1)) Γ̂(ℓ)         Newey-West (Bartlett) — always psd
M = (6ρ²/(1-ρ²)²)^{1/3} n^{1/3}                  Andrews' optimal rule
M = 1.4 n^{1/3}                                  benchmark at ρ = 0.5
consistency needs M → ∞ with M³/n = O(1)
```

Only needed when the model is **misspecified**. A correctly specified AR(p) with MDS errors takes ordinary robust standard errors.

---

## 5. Asymptotics for regression

See [[07 - Asymptotic Theory]].

```
β̂ →p E[XX']⁻¹E[XY]                              consistency
√n(β̂ - β) →d N(0, V),   V = Q⁻¹ Ω Q⁻¹
Q = E[XX'],   Ω = E[XX'e²]
V = Q⁻¹σ²                                        under homoskedasticity
β̂_ols →p β + E[XX']⁻¹E[Xe]                       inconsistency when E[Xe] ≠ 0
T = (β̂ⱼ - βⱼ)/s(β̂ⱼ) →d N(0,1)
```

### Delta method

```
√n(g(β̂) - g(β)) →d N(0, G'VG),   G = ∂g/∂β
s(g(β̂)) = √(Ĝ'V̂Ĝ / n)
```

Fails when `g` is strongly nonlinear or a ratio has a near-zero denominator — use test inversion or the bootstrap.

### Rates

```
√n         mean, OLS, MLE, GMM
√(nh)      kernel regression at a point
n^{-2/(4+d)}   optimal nonparametric MSE rate — the curse of dimensionality
n          unit-root coefficient (superconsistent)
```

### Monte Carlo

```
bias^ = (1/B)Σ(θ̂_b - θ)
mse^  = (1/B)Σ(θ̂_b - θ)²
P̂     = (1/B)Σ 1{|T_b| ≥ 1.96}                  simulated rejection rate
s(P̂)  = √(P(1-P)/B) ≈ 0.22/√B  at P = .05

B = 100 → ±.022,  B = 1000 → ±.007,  B = 10000 → ±.003
```

Rule of thumb: **B = 10,000**.

---

## 6. Testing

See [[08 - Hypothesis Testing and Confidence Intervals]].

```
T = (β̂ⱼ - β⁰ⱼ)/s(β̂ⱼ) →d N(0,1)                  t-test; reject |T| > 1.96 at 5%
W = (θ̂ - θ₀)' V̂θ⁻¹ (θ̂ - θ₀) →d χ²_q             Wald test, q restrictions
W = n(θ̂-θ₀)' V̂θ⁻¹ (θ̂-θ₀)                        equivalent scaling
W = (R'β̂-θ₀)'(R'V̂_β̂R)⁻¹(R'β̂-θ₀)                 linear restrictions R'β = θ₀
W = T²  when q = 1
F = W/q                                          F version, use F_{q,n-k}
p = 1 - G(T)                                     p-value;  p →d U[0,1] under H₀
```

χ² 5% critical values: `q=1` → 3.84, `q=2` → 5.99, `q=3` → 7.82.

### Criterion-based

```
J = min_{β∈B₀} J(β) - min_{β∈B} J(β)  →d χ²_q
J* = n(β̂ - β̃_emd)' V̂_β̂⁻¹ (β̂ - β̃_emd)          efficient minimum distance
```

Invariant to how a nonlinear restriction is written — Wald is not.

### Multiple testing

```
P[min_j pⱼ < α] ≤ kα                             Boole
reject if  min_j pⱼ < α/k                        Bonferroni
familywise p = k · min_j pⱼ
```

### Power

```
power(θ) = 1 - Φ(c - √n·θ/σ)                     one-sided normal-mean example
θₙ = θ₀ + h/√n                                   local alternative
W →d χ²_q(λ)                                     noncentral χ², λ from h and V
```

### Test inversion (for nonlinear parameters)

```
C = { θ : |T(θ)| ≤ 1.96 }
for θ = β₁/β₂:   T(θ) = (β̂₁ - β̂₂θ)/(R'V̂R)^{1/2},   R = (1, -θ)'
```

Grid search over `θ`. Hansen's wage-peak example: delta method `[29.8, 29.9]`, test inversion `[29.1, 30.6]`.

---

## 7. Bootstrap and jackknife

See [[09 - Bootstrap and Resampling]].

```
se_boot = sd of { θ̂*(1), ..., θ̂*(B) }
bias^   = mean(θ̂*) - θ̂
```

### Intervals

```
normal:      θ̂ ± 1.96·se_boot                                    1st order
percentile:  [ q*_{α/2}, q*_{1-α/2} ]                            1st order
BC:          x(α) = Φ(2z₀ + z_α),  interval [q*_{x(α/2)}, q*_{x(1-α/2)}]
BCa:         x(α) = Φ( z₀ + (z_α + z₀)/(1 - a(z_α + z₀)) )       2nd order
percentile-t: [ θ̂ - s(θ̂)q*_{1-α/2},  θ̂ - s(θ̂)q*_{α/2} ]         2nd order — preferred
```

with

```
T* = (θ̂* - θ̂)/s(θ̂*)                             centered at θ̂, NOT at θ₀
z₀ = Φ⁻¹( (1/B)Σ 1{θ̂*(b) ≤ θ̂} )                 median-bias correction
a = Σ(θ̄ - θ̂₍ᵢ₎)³ / [ 6(Σ(θ̄ - θ̂₍ᵢ₎)²)^{3/2} ]    jackknife acceleration
```

### Accuracy

```
one-sided asymptotic CI:   1 - α + O(n^{-1/2})       first-order accurate
percentile-t CI:           1 - α + O(n^{-1})         second-order — an asymptotic refinement
asymptotic test size:      α + O(n^{-1})
bootstrap test size:       α + o(n^{-1})
```

### Bootstrap test

```
p* = (1/B) Σ 1{ |T*(b)| > |T| }                  bootstrap p-value
```

### Jackknife

```
V_jack  = ((n-1)/n) Σ (θ̂₍ᵢ₎ - θ̄)²
bias_jack = (n-1)(θ̄ - θ̂)
```

For OLS, closed form via leverage — this is where HC3 comes from.

---

## 8. Causal inference

See [[10 - Causality and Identification]].

```
Y = h(D, X, U)                                   structural model
Y(0) = h(0,X,U),  Y(1) = h(1,X,U)                potential outcomes
Y = Y(0)(1-D) + Y(1)D                            only one is observed
C = Y(1) - Y(0)                                  individual causal effect
```

### Estimands

```
ATE  = E[Y(1) - Y(0)]
ATT  = E[Y(1) - Y(0) | D = 1]
ATU  = E[Y(1) - Y(0) | D = 0]
CATE = ACE(x) = E[C | X = x]
ATE  = ∫ ACE(x) f(x) dx                          average of conditional effects
LATE = E[Y(1) - Y(0) | X(1) > X(0)]
```

### Selection bias decomposition

```
E[Y|D=1] - E[Y|D=0]
   = E[Y(1)-Y(0) | D=1]                    ATT
   + ( E[Y(0)|D=1] - E[Y(0)|D=0] )         selection bias
```

### Identification under CIA

```
CIA:  D ⊥ U | X
⟹  m(1,x) - m(0,x) = ACE(x)                     regression derivative = causal effect
overlap:  0 < P(D=1 | X=x) < 1
```

**Worked numbers** (Jennifer/George): true ATE = $7.00, naive regression = $8.25, with test score control = $5.50 (low) / $8.50 (high), average = $7.00.

---

## 9. Instrumental variables

See [[11 - Instrumental Variables]].

```
Y = X'β + e,   E[Xe] ≠ 0                         the problem
E[Ze] = 0                                        exclusion / exogeneity
cov(Z,X) ≠ 0                                     relevance
β = cov(Z,Y)/cov(Z,X)                            just-identified IV estimand
β̂_iv = (Z'X)⁻¹ Z'Y
β̂ = ( Ȳ|Z=1 - Ȳ|Z=0 ) / ( X̄|Z=1 - X̄|Z=0 )       Wald estimator, binary Z
```

### 2SLS

```
P_Z = Z(Z'Z)⁻¹Z'
β̂_2sls = (X'P_Z X)⁻¹ (X'P_Z Y)
V = (Q_XZ Ω⁻¹ Q_XZ')⁻¹                           efficient-GMM form
```

### LATE

```
LATE = ( E[Y|Z=1] - E[Y|Z=0] ) / ( E[X|Z=1] - E[X|Z=0] )
```

Requires: `Z ⊥ U` and monotonicity `P[X(1) - X(0) < 0] = 0` (no defiers).

Complier types:

```
             X(0)=0        X(0)=1
X(1)=0    Never Takers    Defiers
X(1)=1    Compliers       Always Takers
```

### Identification failure (γ = 0)

```
β̂_iv - β →d ξ₁/ξ₂ = ρ + ξ₀/ξ₂                    Cauchy — no mean
median bias → ρ  (does not correct OLS centering)
σ̂² →p 0  ⟹  |T| →p ∞                             spuriously tiny SEs, huge t-stats
```

### Weak instruments (local-to-zero: Γ = n^{-1/2}C)

```
β̂_ols →p Σ₂₂⁻¹Σ₂ₑ
β̂_2sls →d ((Q_ZC + ξ₂)'Q_Z⁻¹(Q_ZC + ξ₂))⁻¹ (Q_ZC + ξ₂)'Q_Z⁻¹ξₑ     non-normal, inconsistent
S₁ = ξ|1 + ξ/μ|                                  worst-case (ρ=1) t-stat distribution
F = γ̂²/s(γ̂)² →d χ²₁(μ²)                          first-stage F, noncentral χ²
```

Stock-Yogo, one instrument, tolerating 15% size: threshold `τ² = 1.70`, critical value `F > 8.7` (they report 9.0 with `τ² = 1.82`) → the **"F > 10" rule**.

Equivalent: first-stage `|t| > 2.94` (or 3.16 for F > 10) — **check the reduced-form t-stat on the instrument exceeds 3**.

5% critical values, `k₂ = 1`, maximal size `r`:

| `ℓ₂` | 2SLS r=.10 | .15 | .20 | .25 | LIML r=.10 | .15 | .20 | .25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16.4 | 9.0 | 6.7 | 5.5 | 16.4 | 9.0 | 6.7 | 5.5 |
| 2 | 19.9 | 11.6 | 8.7 | 7.2 | 8.7 | 5.3 | 4.4 | 3.9 |
| 3 | 22.3 | 12.8 | 9.5 | 7.8 | 6.5 | 4.4 | 3.7 | 3.3 |
| 5 | 26.9 | 15.1 | 11.0 | 8.8 | 4.8 | 3.6 | 3.0 | 2.8 |
| 10 | 38.5 | 20.9 | 14.8 | 11.6 | 3.7 | 2.8 | 2.5 | 2.2 |
| 20 | 62.3 | 32.8 | 22.7 | 17.6 | 3.2 | 2.3 | 2.1 | 1.9 |
| 30 | 86.2 | 44.8 | 30.7 | 23.6 | 3.9 | 2.2 | 1.9 | 1.7 |

2SLS needs a *higher* F as instruments multiply; LIML needs a *lower* one.

### Many instruments (ℓ/n → α)

```
β̂_ols  →p β + (H + Σ₂₂)⁻¹ Σ₂ₑ
β̂_2sls →p β + (H + αΣ₂₂)⁻¹ αΣ₂ₑ                  inconsistent, worsening in α
β̂_liml →p β                                      consistent (under homoskedasticity)
α = ℓ/n ≥ 0.05  ⟹  worry; prefer LIML
```

### Overidentification test

```
S = n · ê'Z(Z'Z)⁻¹Z'ê / (ê'ê)  →d χ²_{ℓ-k}       Sargan
S** uses (ê*'Z* - Z'ê) recentering                bootstrap version
```

---

## 10. GMM

See [[12 - GMM and Minimum Distance]].

```
E[g(Wᵢ, β)] = 0                                  moment condition
ḡ(β) = (1/n) Σ g(Wᵢ, β)
J(β) = n · ḡ(β)' W ḡ(β)                          GMM criterion
G = E[∂g/∂β'],   Ω = E[gg']
V = (G'WG)⁻¹ (G'WΩWG) (G'WG)⁻¹                   general
V = (G'Ω⁻¹G)⁻¹                                   efficient GMM, W = Ω⁻¹
J = n ḡ(β̂)' Ω̂⁻¹ ḡ(β̂) →d χ²_{ℓ-k}                overidentification test
J(θ) = n(β̂ - r(θ))' W (β̂ - r(θ))                 minimum distance
```

Moment conditions by estimator:

```
mean       E[X - μ] = 0
OLS        E[X(Y - X'β)] = 0
IV/2SLS    E[Z(Y - X'β)] = 0
MLE        E[∂log f(X|θ)/∂θ] = 0
NLLS       E[(∂m/∂β)(Y - m(X,β))] = 0
```

Efficient GMM on IV moments **= 2SLS under homoskedasticity**.

---

## 11. Panel data

See [[13 - Panel Data]].

```
Yᵢₜ = Xᵢₜ'β + uᵢ + εᵢₜ                            one-way error component
Yᵢₜ = Xᵢₜ'β + uᵢ + vₜ + εᵢₜ                       two-way
```

### Transformations

```
Ȳᵢ = (1/Tᵢ) Σₜ Yᵢₜ                                individual mean
Ẏᵢₜ = Yᵢₜ - Ȳᵢ                                    within (demeaning)
Mᵢ = I - 1ᵢ(1ᵢ'1ᵢ)⁻¹1ᵢ'                           demeaning operator; Ẏᵢ = MᵢYᵢ
ΔYᵢₜ = Yᵢₜ - Yᵢ,ₜ₋₁                               first difference
```

Two-way within: subtract individual mean, subtract time mean, add grand mean.

### Estimators

```
β̂_fe = ( Σᵢ Ẋᵢ'Ẋᵢ )⁻¹ ( Σᵢ Ẋᵢ'Ẏᵢ ) = ( Σᵢ Xᵢ'MᵢXᵢ )⁻¹ ( Σᵢ Xᵢ'MᵢYᵢ )
β̂_Δ  = ( Σᵢ ΔXᵢ'ΔXᵢ )⁻¹ ( Σᵢ ΔXᵢ'ΔYᵢ )
β̂_gls = random effects, Ω = σ_u²11' + σ_ε²I
```

Facts: `β̂_fe` ≡ dummy-variable estimator (FWL). `T = 2` ⟹ `β̂_Δ = β̂_fe`. GLS on first differences with i.i.d. errors = `β̂_fe`.

### Variance

```
V⁰_fe = σ_ε² ( Σᵢ Ẋᵢ'Ẋᵢ )⁻¹                       homoskedastic
σ̂_ε² = (1/(n - N - k)) Σ ε̂ᵢₜ²                    note: subtract N for the fixed effects
V̂_fe^cluster = (Ẋ'Ẋ)⁻¹ ( Σᵢ Ẋᵢ'ε̂ᵢ ε̂ᵢ'Ẋᵢ ) (Ẋ'Ẋ)⁻¹
   × N/(N-1)                                     C. Hansen adjustment (recommended)
   × ((n-1)/(n-N-k))(N/(N-1))                    heavier adjustment, not theory-justified
V⁰_fe ≥ V_pool                                   cost of robustness: FE is less efficient
```

### Assumptions

```
strict exogeneity:  E[Xᵢₛ εᵢₜ] = 0  for all s, t
strict mean indep:  E[εᵢₜ | Xᵢ] = 0
predetermined:      E[Xᵢₛ εᵢₜ] = 0  for s ≤ t only
```

### Dynamic panel

```
Yᵢₜ = αYᵢ,ₜ₋₁ + Xᵢₜ'β + uᵢ + εᵢₜ
Nickell bias = O(1/T), downward
```

Fixes: Anderson-Hsiao (instrument `ΔYᵢ,ₜ₋₁` with `Yᵢ,ₜ₋₂`), Arellano-Bond (difference GMM), Blundell-Bond (system GMM).

---

## 12. Difference in differences

See [[14 - Difference in Differences]].

```
Yᵢₜ = β₀ + β₁Stateᵢ + β₂Timeₜ + θDᵢₜ + εᵢₜ       Dᵢₜ = Stateᵢ × Timeₜ
Yᵢₜ = θDᵢₜ + uᵢ + vₜ + εᵢₜ                        two-way fixed effects form
Yᵢₜ = θDᵢₜ + Xᵢₜ'β + uᵢ + vₜ + εᵢₜ                with controls
Yᵢₜ = θDᵢₜ + Xᵢₜ'β + uᵢ + vₜ + t·wᵢ + εᵢₜ         unit-specific trends (needs T ≥ 4)
Yᵢₜ = Σ_{k ≠ -1} θ_k 1{t - Eᵢ = k} + uᵢ + vₜ + εᵢₜ   event study
```

The 2×2 mapping:

```
                 Treated        Control       Diff
Before        β₀+β₁            β₀             β₁
After         β₀+β₁+β₂+θ       β₀+β₂          β₁+θ
Diff          β₂+θ             β₂             θ    ← DiD
```

Card-Krueger: `(20.90-20.43) - (21.10-23.38) = 0.47 - (-2.28) = 2.75`.
DiTella-Schargrodsky: `(0.035-0.112) - (0.105-0.095) = -0.077 - 0.010 = -0.087`.

Homogeneity tests: add `N₂-1` treated-region × time interactions (treatment homogeneity); add `N₁-1` control-region × time interactions (control homogeneity — the more serious one).

---

## 13. Time series

See [[15 - Time Series]].

```
γ(k) = cov(Yₜ, Yₜ₋ₖ),   ρ(k) = γ(k)/γ(0)         autocovariance, ACF
var[Ȳ] = σ²/n + (2/n)Σ_{ℓ=1}^{n}(1 - ℓ/n)γ(ℓ)    variance of the sample mean
```

Shock hierarchy — `i.i.d. ⊂ MDS ⊂ white noise`:

```
MDS:          E[eₜ | ℱₜ₋₁] = 0                   unforecastable in mean
white noise:  E[eₜ]=0, E[eₜ²]=σ², cov(eₜ,eₜ₋ₖ)=0  serially uncorrelated
MDS CLT:      n^{-1/2}Σuₜ →d N(0,Σ)              stationary ergodic MDS, 2 moments
mixing CLT:   n^{-1/2}Σuₜ →d N(0,Ω)              r > 2 moments + Σα(ℓ)^{1-2/r} < ∞
```

Growth rates:

```
Qₜ = 100(Yₜ/Yₜ₋₁ - 1) ≈ 100 Δlog Yₜ              one-period (recommended)
Aₜ = 100((Yₜ/Yₜ₋₁)^s - 1)                        annualized (avoid for analysis)
```

Models:

```
Yₜ = μ + Σ_{j≥0} bⱼeₜ₋ⱼ,  b₀=1, Σbⱼ²<∞           Wold decomposition
Yₜ = α₀ + α₁Yₜ₋₁ + ... + α_pYₜ₋ₚ + eₜ            AR(p);  α(L)Yₜ = α₀ + eₜ
α(z) = 1 - α₁z - ... - α_p z^p                    autoregressive polynomial
Yₜ = μ + eₜ + θ₁eₜ₋₁ + ... + θ_q eₜ₋q            MA(q)
half-life = log(0.5)/log(α₁)                      AR(1) shock persistence
bⱼ = α₁bⱼ₋₁ + ... + α_p bⱼ₋ₚ,  b₀ = 1            impulse response recursion
IRFⱼ = σbⱼ                                        one-sd-shock scaling
Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + eₜ                   VAR
ΔYₜ = α(Yₜ₋₁ - βXₜ₋₁) + short-run terms + eₜ      error correction model
```

MA(1) moments: `var = (1+θ²)σ²`, `ρ(1) = θ/(1+θ²)`, `ρ(k) = 0` for `k ≥ 2`.
AR(1) moments: `E[Y] = α₀/(1-α₁)`, `var = σ²/(1-α₁²)`, `ρ(k) = α₁^k`.

**Stationarity**: all roots of `α(z)` outside the unit circle.
- AR(1): `|α₁| < 1`
- AR(2): `α₁ + α₂ < 1`, `α₂ - α₁ < 1`, `α₂ > -1` (a triangle); complex roots below `α₁² + 4α₂ = 0` give damped oscillations.

**Identification**: AR coefficients are identified; MA and ARMA generally are **not** — `θ` and `1/θ` give identical `ρ(1)`, and ARMA has cancelling roots. Restrict to invertible roots.

Estimation and inference:

```
α̂ = (ΣXₜXₜ')⁻¹(ΣXₜYₜ)                            OLS; consistent under stationarity + ergodicity
√n(α̂-α) →d N(0, Q⁻¹ΣQ⁻¹),  Σ = E[XₜXₜ'eₜ²]      MDS errors ⟹ ordinary robust SEs are valid
√n(α̂₁-α₁) →d N(0, 1-α₁²)                         AR(1), no intercept, homoskedastic
AIC(p) = n log σ̂²(p) + 2p                        keep the sample fixed across p
```

### Unit root

```
Yₜ = Yₜ₋₁ + eₜ                                    random walk
ΔYₜ = μ + δt + γYₜ₋₁ + Σ φⱼΔYₜ₋ⱼ + eₜ            ADF regression; test γ = 0
```

Dickey-Fuller critical values, not normal. `β̂` converges at rate `n`.

### Volatility

```
ARCH(q):    σₜ² = ω + Σ αᵢ eₜ₋ᵢ²
GARCH(1,1): σₜ² = ω + α eₜ₋₁² + β σₜ₋₁²
persistence = α + β                               typically .95-.99 daily
```

---

## 14. Nonparametrics, quantiles, RDD

See [[16 - Nonparametrics, Quantiles, and RDD]].

```
m̂(x) = Σ K((Xᵢ-x)/h) Yᵢ / Σ K((Xᵢ-x)/h)          Nadaraya-Watson (local constant)
local linear: min_{a,b} Σ K((Xᵢ-x)/h)(Yᵢ - a - b(Xᵢ-x))²      → m̂(x) = â
h_opt ∝ n^{-1/5}                                  MSE-optimal, local linear
rate = n^{-2/5}  (1 regressor);  n^{-2/(4+d)} in d dimensions
m(x) ≈ β₀ + β₁ψ₁(x) + ... + βₖψₖ(x)               series / sieve; K plays the role of h
```

### Quantile regression

```
Q_τ(Y|X) = X'β(τ)
β̂(τ) = argmin_b Σ ρ_τ(Yᵢ - Xᵢ'b)
ρ_τ(u) = u(τ - 1{u < 0})                          check function; τ=0.5 ⟹ median regression
```

### RDD

```
D = 1{X ≥ c}
θ̄ = m(c+) - m(c-)                                sharp RDD (needs m₀, m₁ continuous at c)
θ̄ = (m(c+) - m(c-)) / (p(c+) - p(c-))            fuzzy RDD (also needs D ⊥ θ near c)
θ̂ = [β̂₁(c)]₁ - [β̂₀(c)]₁                          local linear on each side
Y = β₀ + β₁X + β₃(X-c)D + θD + e   on |X-c| ≤ h  simple estimator (rectangular kernel)
```

Bias and variance:

```
bias[θ̂] = (h²σ²_{K*}/2)(m''(c+) - m''(c-))       zero to first order with a common bandwidth
var[θ̂]  = (R*_K/nh)(σ²(c+)/f(c+) + σ²(c-)/f(c-))
```

Fan-Gijbels rule-of-thumb bandwidth:

```
h_rot = 0.58 (σ̂²(ξ₂-ξ₁)/B̂)^{1/5} n^{-1/5},   B̂ = (1/n)Σ(m̂''(Xᵢ)/2)²1{ξ₁≤Xᵢ≤ξ₂}
```

Constant is 0.58 for a normalized kernel, 1.00 for unnormalized Rectangular, 1.42 for unnormalized Triangular. Compute for several polynomial orders `q`, cross-check with CV, then **reduce ~25%** to undersmooth.

Triangular kernel is efficient at the boundary; Rectangular costs ~3% root AMSE. Never global high-order polynomials. Plot the nonparametric estimate with confidence bands, **not binned means**.

---

## 15. Limited dependent variables

See [[17 - Limited Dependent Variables]].

```
P(x) = P[Y=1|X=x] = E[Y|X=x]                      response probability
var[e|X] = P(X)(1-P(X))                           inherently heteroskedastic
P(x) = G(x'β)                                     index model
  probit:  G = Φ,           σ = 1
  logit:   G = Λ(u) = (1+e^{-u})⁻¹,  σ = π/√3 ≈ 1.8
Y* = X'β + e,  Y = 1{Y* > 0}                      latent variable form
ℓₙ(β) = Σ log G(Zᵢ'β),  Z = X if Y=1 else -X      log-likelihood, globally concave
δ(x) = ∂P/∂x = β g(x'β)                           marginal effect ≠ coefficient
AME = β̂ (1/n) Σ g(Xᵢ'β̂)                           average marginal effect
odds ratio = exp(βⱼ)                              logit only
β* = β/σ                                          only the SCALED coefficient is identified
```

To compare: **multiply probit coefficients by 1.8, or divide logit by 1.8.** Identified quantities: scaled coefficients, ratios of coefficients, marginal effects. Coefficients are **not** comparable across specifications — the scale changes.

Hansen's preferred specification is the **probit series model** `P(x) = Φ(x_K'β_K)`: flexible *and* boundary-respecting. The LPM violates `[0,1]` in practice (an 80-year-old married with probability 113%); the linear series model can too (-27%). Choice of link barely matters; **specification of the index matters a lot**.

### Count data

```
E[Y|X] = exp(X'β)                                 Poisson; βⱼ = semi-elasticity
effect multiplier = exp(βⱼ)
```

Poisson QMLE + robust SEs is consistent whenever the conditional mean is right — no distributional assumption, handles zeros, works with fixed effects (no incidental-parameters problem).

### Censoring

```
Y* = X'β + e,  e|X ~ N(0,σ²),  Y = max(Y*, 0)     Tobit
m*(x) = X'β                                        uncensored mean
m(x)  = X'β Φ(X'β/σ) + σ φ(X'β/σ)                  censored mean
m#(x) = X'β + σ λ(X'β/σ)                           truncated mean
m*(x) ≤ m(x) ≤ m#(x)                               deleting zeros is WORSE than keeping them
β_BLP = β(1 - π),   π = P[Y=0]                     Greene's rule: OLS slope bias ≈ censoring %
```

Robust alternatives (Powell), using quantile equivariance `Q_τ[Y|X] = max(q_τ(X), 0)`:

```
CLAD: M̂ₙ(β) = (1/n)Σ |Yᵢ - max(Xᵢ'β, 0)|
CQR:  M̂ₙ(β;τ) = (1/n)Σ ρ_τ(Yᵢ - max(Xᵢ'β, 0))
```

No normality, no error independence needed. Not globally convex — watch for local minima. Hansen's preferred estimator under censoring.

### Sample selection

```
Y = X'β + e            observed only when S = 1
S = 1{Z'γ + u > 0}
E[Y | X, S=1] = X'β + ρ λ(X'γ)                    selection bias IS omitted variable bias
λ(u) = φ(u)/Φ(u)                                  inverse Mills ratio
E[Y | X, Z, S=1] = X'β + σ₂₁ λ(Z'γ)               Heckman's model
```

Bias arises **iff `ρ ≠ 0`**. If `X'γ` is constant, selection shifts only the intercept and marginal effects are unaffected. Unlike censoring (which always attenuates), selection can steepen *or* flatten.

Two-step: probit `S` on `Z` → `γ̂`; form `λ̂ᵢ = λ(Zᵢ'γ̂)`; OLS `Y` on `(X, λ̂)` on the selected sub-sample. `λ̂` is a generated regressor — correct the standard errors. The t-statistic on `σ̂₂₁` tests exogenous selection. Needs an exclusion restriction: a variable in `Z` not in `X`.

---

## 16. Model selection and ML

See [[18 - Model Selection and Machine Learning]].

```
AIC = -2 log L + 2k                               efficient for prediction
BIC = -2 log L + k log n                          consistent for a sparse true model
CV(K) = (1/K) Σ_k (prediction error on fold k)
LOO-CV for OLS = (1/n) Σ (êᵢ/(1-hᵢᵢ))² = σ̄²      closed form
```

### Penalized regression

```
ridge:  β̂ = argmin Σ(Yᵢ - Xᵢ'b)² + λΣbⱼ²  =  (X'X + λI)⁻¹X'Y
lasso:  β̂ = argmin Σ(Yᵢ - Xᵢ'b)² + λΣ|bⱼ|         exact zeros ⟹ selection
elastic net: λ₁Σ|bⱼ| + λ₂Σbⱼ²
```

Ridge beats OLS in MSE for `0 < λ < 2σ²/β'β` (Hansen Thm 29.2).

Lasso rate condition: `‖β‖₀ log p / √n → 0`. Lasso penalty scale: `λ ≈ C√(n log p)`.

### Post-selection inference

Setup:

```
Y = Dθ + X'β + e,   E[e|D,X] = 0                  structural
D = X'γ + V,        E[V|X] = 0                    auxiliary
Y = X'η + U,        η = β + γθ                    reduced form
```

**Double selection**: lasso (D on X) → `X₁`; lasso (Y on X) → `X₂`; OLS `Y` on `(D, X₁ ∪ X₂)`.

**Partialling-out (post-regularization)**:

```
Y - X'η = (D - X'γ)θ + e
V̂ = D - X'γ̂,   Û = Y - X'η̂,   θ̂_PR = (V̂'V̂)⁻¹V̂'Û
√n(θ̂_PR - θ) →d N( 0,  E[V²e²]/(E[V²])² )
```

Why it works — Neyman orthogonality:

```
naive moment:  m(θ,β) = E[D(Y - Dθ - X'β)],   ∂m/∂β = -E[DX'] ≠ 0     sensitive
PR moment:     m(θ,β) = E[(D-X'γ)(Y - Dθ - X'β)],  ∂m/∂β = -E[VX'] = 0  insensitive
```

**DML (cross-fitting)**: split into `K` folds; estimate `γ̂₋ₖ, η̂₋ₖ` off fold `k`; form `V̂ₖ, Ûₖ` on fold `k`; stack; `θ̂_DML = (V̂'V̂)⁻¹V̂'Û`. `K = 10`. Same asymptotic variance as `θ̂_PR`; error term improves from `Op(‖γ‖₀ log p/√n)` to `Op(√(‖γ‖₀ log p/n))`.

Rate conditions: `(‖β‖₀ + ‖γ‖₀) log p / √n = o(1)`; Lasso IV needs `‖Γ‖₀ log p/√n → 0`, and split-sample IV only `‖Γ‖₀ log p/n → 0`.

---

---

## 17. Systems, PCA, factor models

See [[20 - Multivariate Regression and Factor Models]].

```
Y = X̄'β + e,  E[ee'|X] = Σ                       multivariate regression
β̂_sur = (Σᵢ X̄ᵢ'Σ̂⁻¹X̄ᵢ)⁻¹ (Σᵢ X̄ᵢ'Σ̂⁻¹Yᵢ)           SUR (feasible GLS)
β̂_sur = β̂_ols   when regressors common, or Σ diagonal
V*_β = (E[X̄'Σ⁻¹X̄])⁻¹ ≤ V_β                        SUR efficiency gain
```

Reduced rank: `B = GA'` with rank `r`; `Ĝ` = generalized eigenvectors of `X̃'Ỹ(Ỹ'Ỹ)⁻¹Y'X̃` w.r.t. `X̃'X̃`, largest `r`. Maximized `ℓₙ` includes `-(n/2)Σⱼlog(1 - λ̂ⱼ)` — the Johansen statistic.

```
h_j = argmax_{h'h=1, h⊥h₁..h_{j-1}} var[h'X]      j-th principal component
Σ = HDH',  U = H'X,  var[U] = D                   PCA = eigendecomposition
share_j = d_j / tr(Σ)
```

Factor models:

```
X = ΛF + u,  E[FF'] = I_r,  Ψ = E[uu']
Σ = ΛΛ' + Ψ                                       communality + uniqueness
ℓₙ(Λ,Ψ) = -(nk/2)log2π - (n/2)logdet(ΛΛ'+Ψ) - (n/2)tr((ΛΛ'+Ψ)⁻¹Σ̂)
F̃ᵢ = (Λ̂'Ψ̂⁻¹Λ̂)⁻¹Λ̂'Ψ̂⁻¹Xᵢ                           Bartlett scoring
F̄ᵢ = Λ̂'Σ̂⁻¹Xᵢ                                     regression scoring (preferred)
```

Approximate factor model (principal-components estimator), `D̂`/`Ĥ` = first `r` eigenvalues/vectors of `Σ̂`:

```
Λ̂ = Ĥ D̂^{1/2},   F̂ᵢ = D̂^{-1/2} Ĥ' Xᵢ
```

Inconsistent for fixed `k`; consistent as `k → ∞`.

Factor-augmented regression `Y = F'β + Z'γ + e`, `X = ΛF + u`:

```
β̂ →p β* = (I_r + D⁻¹Λ'ΨΛD⁻¹)⁻¹ β                  inconsistent for fixed k
β* → β as k → ∞;  normality needs k²/n → ∞
```

---

## 18. Shrinkage and model averaging

See [[21 - Shrinkage and Model Averaging]].

### MSE of selection

```
θ̂_pms = θ̂·1{θ̂'θ̂ > c},   c = 2K (AIC), K log n (BIC), χ²_K(.95) (5% test)
mse[θ̂_pms] = K + (2λ - K)F_{K+2}(c,λ) - λF_{K+4}(c,λ),   λ = θ'θ
```

Lower MSE than `θ̂` for `λ < K`, higher for `λ > K`. BIC's MSE → ∞ as `n → ∞`.

### Shrinkage

```
θ̃ = (1-w)θ̂
bias = -wθ,  var = (1-w)²V,  wmse = K(1-w)² + w²λ,   λ = θ'V⁻¹θ
wmse[θ̃] < wmse[θ̂]  for 0 < w < 2K/(K+λ)
w₀ = K/(K+λ),   min wmse = Kλ/(K+λ)
θ̃ = (1 - c/(θ̂'V⁻¹θ̂))θ̂                             Stein rule (θ̂'V⁻¹θ̂ = Wald stat)
θ̃_JS = (1 - (K-2)/(θ̂'V̂⁻¹θ̂))θ̂                     James-Stein; dominates MLE for K > 2
θ̃⁺ = (1 - (K-2)/(θ̂'V̂⁻¹θ̂))₊ θ̂                     positive part; uniformly better
```

Toward restrictions `R'θ = r` with `q > 2`:

```
θ̂_R = θ̂ - V̂R(R'V̂R)⁻¹(R'θ̂ - r)
θ̃⁺ = θ̂ - ((q-2)/J)₁ (θ̂ - θ̂_R),   J = (θ̂-θ̂_R)'V̂⁻¹(θ̂-θ̂_R)
J = n(σ̂²_R - σ̂²)/s²                               convenient regression form
```

Group version: `θ̃_g = θ̂_g(1 - (K_g-2)/(θ̂_g'V̂_g⁻¹θ̂_g))₊`, needs every `K_g > 2`.

### Model averaging

```
θ̂(w) = Σ_m w_m θ̂_m,   w ∈ S (simplex: w_m ≥ 0, Σw_m = 1)
w_m = exp(-ΔBIC_m/2)/Σⱼexp(-ΔBICⱼ/2)               smoothed BIC weights
w_m = exp(-ΔAIC_m/2)/Σⱼexp(-ΔAICⱼ/2)               smoothed AIC weights
```

Use `Δ` (differences from the minimum), never levels — avoids overflow.

```
C(w) = ê(w)'ê(w) + 2σ̄²Σ_m w_m K_m = w'Ê'Êw + 2σ̄²K'w      Mallows (MMA)
CV(w) = w'Ẽ'Ẽw                                            jackknife (JMA / stacking)
ŵ = argmin_{w ∈ S} (·)                                    quadratic programming
```

For `M = 2` nested models, `ŵ_mma = (σ̄²(K₂-K₁)/(ê₁'ê₁ - ê₂'ê₂))₁` — a Stein-rule weight. MMA generalizes James-Stein to `M > 2`.

### Trees and forests

See [[18 - Model Selection and Machine Learning]].

```
Y = μ₁1{X_d ≤ γ} + μ₂1{X_d > γ} + e                regression sample split (NLLS over d, γ)
C = Σêᵢ² + αN                                      pruning criterion, N = leaves
m̂_bag(x) = (1/B)Σ_b m̂*_b(x)                        bagging
ẽᵢ = Yᵢ - m̂₋ᵢ(Xᵢ)                                  out-of-bag error (~37% left out)
V̂ₙ(x) = Σᵢ((1/B)Σ_b(N_ib - Nᵢ)(m̂*_b(x) - m̂_bag(x)))²   infinitesimal jackknife
(m̂_rf(x) - m(x))/√V̂ₙ(x) →d N(0,1)                  Wager-Athey, honest + subsampled
```

Random forest defaults: `N_min = 5`, `m = p/3` variables sampled per split, `0 < α ≤ 0.2`.

---

## 19. Bayesian methods

See [[22 - Bayesian Methods]].

```
f(x,θ) = f(x|θ)π(θ)                                joint
m(x) = ∫ Lₙ(x|θ)π(θ)dθ                             marginal likelihood
π(θ|X) = Lₙ(X|θ)π(θ)/m(X)                          posterior ∝ likelihood × prior
θ̂_Bayes = ∫θ π(θ|X)dθ                              posterior mean (quadratic loss)
θ̄_Bayes = median(θ|X)                              posterior median (absolute loss)
R(T|X) = ∫ℓ(T,θ)π(θ|X)dθ                           Bayes risk
```

Conjugate products:

```
beta(α₁,β₁)×beta(α₂,β₂) ∝ beta(α₁+α₂, β₁+β₂)
gamma(α₁,β₁)×gamma(α₂,β₂) ∝ gamma(α₁+α₂-1, β₁+β₂)
N(μ₁,1/ν₁)×N(μ₂,1/ν₂) ∝ N((ν₁μ₁+ν₂μ₂)/(ν₁+ν₂), 1/(ν₁+ν₂))
```

Standard posteriors (precision `ν = 1/σ²`):

```
Bernoulli + beta(α,β):    π(p|X) = beta(Sₙ+α, n-Sₙ+β),  p̂ = (Sₙ+α)/(n+α+β)
Normal mean, ν known:     μ̂ = (nνX̄ₙ + ν̄μ̄)/(nν + ν̄)
Normal precision:         σ̂²_Bayes = (nσ̃²_mle + 2ασ̄²)/(n + 2α)
Both unknown (NormalGamma): μ̂ = (nX̄ₙ + λμ̄)/(n+λ),  σ̂² = ((n-1)s² + 2ασ̄²)/(n-1+2α)
```

`λ` and `2α` are the prior's worth in pseudo-observations.

Credible sets and tests:

```
P[θ ∈ C | X] = ∫_C π(θ|X)dθ = 1-η                  credible interval
HPD: π(θ₁|X) ≥ π(θ₂|X) for all θ₁∈C, θ₂∉C          shortest such interval
πⱼ(X) = πⱼmⱼ(X)/Σᵢπᵢmᵢ(X)                          posterior model probability
Bayes Factor = m₂(X)/m₁(X);  select H₂ if prior odds × BF > 1
```

Frequentist properties in the normal model:

```
E[μ̂_Bayes] = (nμ + λμ̄)/(n+λ)
bias = λ(μ̄-μ)/(n+λ),   var = σ²/(n+λ)
```

Lower variance than `X̄ₙ`, biased unless the prior is centered on the truth.

---

## 20. Density estimation

See [[23 - Nonparametric Density Estimation]].

```
f̂(x) = nⱼ/(nw)                                    histogram, bin width w
f̂(x) = (1/nh) Σᵢ K((Xᵢ - x)/h)                     kernel density estimator
R_K = ∫K(u)²du                                     kernel roughness
R(f'') = ∫(f''(x))²dx                              density curvature roughness
```

Bias, variance, and the trade-off:

```
E[f̂(x)] = f(x) + (1/2)f''(x)h² + o(h²)             bias ∝ curvature
V_f̂ = f(x)R_K/(nh) + o(1/(nh))                     nh = effective sample size
AIMSE = (1/4)R(f'')h⁴ + R_K/(nh)
h₀ = (R_K/R(f''))^{1/5} n^{-1/5},   AIMSE ~ n^{-4/5},  rate n^{-2/5}
```

Bandwidth rules:

```
h_r = σ C_K n^{-1/5},   C_K = (8√π R_K/3)^{1/5}    reference (Silverman)
h_r = 1.06 s n^{-1/5}                              Gaussian optimal
h_r = 0.9 σ̃ n^{-1/5}                               Silverman rule of thumb
h = 0.776 / Ŝ₂(b̄₂(h))^{1/5} · n^{-1/5}             Sheather-Jones (Gaussian), solve numerically
```

`C_K` ranges only 1.049-1.064 across kernels, so **1.06 works for any unit-variance kernel**. Epanechnikov minimizes AIMSE; the Gaussian loses only **2%**. Bandwidth matters, kernel choice barely does.

## Quick reference card

| Situation | Formula / rule |
| --- | --- |
| Standard error of a mean | `s/√n` |
| Halve a standard error | 4× the data |
| 95% CI | `θ̂ ± 1.96 s(θ̂)` |
| 5% t-test | `\|T\| > 1.96` |
| 5% Wald, q restrictions | `W > 3.84, 5.99, 7.82` for `q = 1,2,3` |
| Default SEs | almost always wrong — use HC2/HC3 |
| Grouped data | cluster; effective `n` is `G` |
| Clustering inflation | `1 + ρ(N-1)` |
| Weak instruments | first-stage `F > 10`, or reduced-form `\|t\| > 3` |
| Many instruments | `ℓ/n ≥ 0.05` ⟹ prefer LIML |
| Monte Carlo replications | `B = 10,000`;  `s(P̂) ≈ 0.22/√B` |
| Bootstrap centering | at `θ̂`, never at `θ₀` |
| Nonlinear parameter CI | test inversion or bootstrap, not delta method |
| Multiple tests | Bonferroni: reject if `p < α/k` |
| Probit/logit | report marginal effects, not coefficients |
| Non-negative outcome | Poisson QMLE + robust SEs |
| Correctly specified AR(p) | ordinary robust SEs — HAC not needed |
| HAC bandwidth | `M = 1.4 n^{1/3}` as a benchmark |
| Censored outcome | check `π`; OLS slope bias ≈ `π`; prefer CLAD over Tobit |
| Never fix censoring by | dropping the zeros — truncation is worse |
| Binary outcome, want probabilities | probit series model, not LPM |
| Probit vs logit | irrelevant; the index specification is what matters |
| RDD plotting | nonparametric estimate + bands, never binned means |
| High-dim controls, want inference on θ | double selection / partialling-out / DML |
| Shrinkage always helps | some `0 < w < 2K/(K+λ)` beats no shrinkage |
| James-Stein needs | `K > 2`; always use the positive part |
| Model averaging default | jackknife (JMA / stacking), no homoskedasticity needed |
| Selecting a model | AIC or CV — not BIC |
| Density estimation | kernel choice ~2%, bandwidth is everything |
| Density bandwidth | Sheather-Jones; `n ≥ 100`, `≥ 50` distinct values |
| SUR is worth running only if | equations have different regressors |
| Factor loadings | never interpret individually — rotation is arbitrary |

Related:

- [[00 - Econometrics Hub]]
- [[21 - Shrinkage and Model Averaging]]
- [[22 - Bayesian Methods]]
- [[99 - Econometrics Glossary]]
