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
Ω = Σ_{k=-∞}^{∞} E[Xₜeₜ Xₜ₋ₖ'eₜ₋ₖ]              long-run variance
Ω̂ = Γ̂₀ + Σ_{k=1}^{M} w(k)(Γ̂ₖ + Γ̂ₖ')            Newey-West
w(k) = 1 - k/(M+1)                               Bartlett kernel
M ≈ 4(T/100)^{2/9}                               crude default bandwidth
```

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
Yₜ = α + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + eₜ               AR(p)
Yₜ = eₜ + θ₁eₜ₋₁ + ... + θ_q eₜ₋q                 MA(q)
half-life = log(0.5)/log(φ)                       AR(1) shock persistence
Yₜ = A₁Yₜ₋₁ + ... + AₚYₜ₋ₚ + eₜ                   VAR
ΔYₜ = α(Yₜ₋₁ - βXₜ₋₁) + short-run terms + eₜ      error correction model
```

Stationarity: AR(1) needs `|φ| < 1`; AR(p) needs all roots outside the unit circle.

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
τ = lim_{x↓c} E[Y|X=x] - lim_{x↑c} E[Y|X=x]       sharp RDD
τ_fuzzy = [ jump in Y ] / [ jump in P(D=1) ]      = IV with 1{X ≥ c} as instrument
```

Estimate with local linear on each side, triangular kernel, MSE-optimal bandwidth, bias-corrected robust CI (`rdrobust`). Never global high-order polynomials.

---

## 15. Limited dependent variables

See [[17 - Limited Dependent Variables]].

```
E[Y|X] = P(Y=1|X)                                 binary outcome
P(Y=1|X) = G(X'β)
  probit:  G = Φ
  logit:   G = Λ(u) = e^u/(1+e^u)
Y* = X'β + e,  Y = 1{Y* > 0}                      latent variable form
∂P/∂Xⱼ = g(X'β)·βⱼ                                marginal effect ≠ coefficient
AME = (1/n) Σ g(Xᵢ'β̂) β̂ⱼ                          average marginal effect
odds ratio = exp(βⱼ)                              logit only
β_logit ≈ 1.6-1.8 × β_probit
var[Y|X] = p(1-p)                                 LPM is inherently heteroskedastic
```

### Count data

```
E[Y|X] = exp(X'β)                                 Poisson; βⱼ = semi-elasticity
effect multiplier = exp(βⱼ)
```

Poisson QMLE + robust SEs is consistent whenever the conditional mean is right — no distributional assumption, handles zeros, works with fixed effects (no incidental-parameters problem).

### Sample selection

```
Y = X'β + e            observed only when S = 1
S = 1{Z'γ + u > 0}
λ̂ᵢ = φ(Zᵢ'γ̂)/Φ(Zᵢ'γ̂)                             inverse Mills ratio
Y = X'β + σ_eu λ̂ + v                              Heckman second step
```

Needs an exclusion restriction: a variable in `Z` not in `X`.

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
| High-dim controls, want inference on θ | double selection / partialling-out / DML |

Related:

- [[00 - Econometrics Hub]]
- [[99 - Econometrics Glossary]]
