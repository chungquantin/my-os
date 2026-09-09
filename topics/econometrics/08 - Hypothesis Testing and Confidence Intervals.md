---
title: Hypothesis Testing and Confidence Intervals
tags: [econometrics, testing, inference, p-values, multiple-testing, beginner]
---

# Hypothesis Testing and Confidence Intervals

Source: Hansen, *Econometrics*, chapter 9.

## The machinery

- **Null hypothesis** `H₀` — the restriction you are testing, usually "this coefficient is zero".
- **Alternative** `H₁` — what you conclude if you reject.
- **Test statistic** `T` — a number computed from data whose distribution under `H₀` you know.
- **Significance level** `α` — the probability of rejecting a true null that you are willing to accept. Conventionally 0.05.
- **Critical value** `c` — the threshold. Reject if `T > c`.
- **Type I error** — rejecting a true null. Occurs with probability `α` by construction.
- **Type II error** — failing to reject a false null.
- **Power** — 1 minus P(Type II). The probability of detecting a real effect.

## The t-test

For a single coefficient:
```
T = (β̂ⱼ - β⁰ⱼ) / s(β̂ⱼ)
```
Under `H₀` and asymptotic normality, `T →d N(0,1)`. Reject at 5% if `|T| > 1.96`.

If errors are normal and homoskedastic, the exact distribution is Student `t` with `n-k` degrees of freedom. For moderate `n` the difference is tiny; the `t` critical value is slightly larger and hence slightly conservative, which is why many packages default to it.

## The Wald test (multiple restrictions)

For `q` restrictions `r(β) = θ₀` at once:
```
W = (θ̂ - θ₀)' V̂θ⁻¹ (θ̂ - θ₀)   →d  χ²_q  under H₀
```
5% critical values: `q=1` → 3.84, `q=2` → 5.99, `q=3` → 7.82. Note `W = T²` when `q = 1`, so the Wald test generalizes the t-test.

Important detail: the limiting distribution depends only on `q`, the number of restrictions — not on `k`, the number of parameters estimated.

The **F version** is `F = W/q`, conventionally compared to `F_{q, n-k}` critical values. Under homoskedastic covariance estimation this is exactly the classic F statistic. With robust covariance there is no formal justification for the F distribution, but it is slightly more conservative and gives continuity with the exact theory, so it is common practice.

**Joint vs individual tests.** These can disagree, and the disagreement is informative. Two coefficients can each be individually insignificant while jointly significant (common under collinearity — the data pin down the sum but not each part). Conversely a joint test can reject while no individual test does. Report both when the group has meaning.

Hansen's example: union membership dummies for men and women. Individually, men's coefficient is significant (`t = 4.7`) and women's is not (`t = 1.2`). The joint Wald statistic is 23 with p-value 0.000 — union membership matters for *someone*, which is what the joint test tells you and neither individual test does.

## Criterion-based tests

An alternative family: compare the value of a minimized criterion with and without the restriction.

```
J = min_{β ∈ B₀} J(β) - min_{β ∈ B} J(β)
```

This is the **likelihood ratio** idea (for MLE), the **minimum distance** statistic (for least squares), and the **J statistic** (for GMM). All measure "how much worse does the fit get when I impose the restriction?" Under the null they also have `χ²_q` limits.

Wald and criterion-based tests are asymptotically equivalent but can differ in finite samples. Wald tests are notoriously sensitive to how you write a nonlinear restriction (testing `β₁ = β₂` versus `β₁/β₂ = 1` gives different statistics); criterion-based tests are not. When testing nonlinear restrictions, prefer the criterion-based or bootstrap version.

## P-values

The **p-value** is `p = 1 - G(T)`, the probability under `H₀` of a statistic at least as extreme as the one observed. Equivalently, the **marginal significance level** — the smallest `α` at which you would reject.

Under `H₀`, `p` is uniformly distributed on `[0,1]`. That is a useful fact for simulation checks: if your test is correctly sized, its p-values under the null should look uniform.

**What a p-value is not**: it is *not* the probability that the null hypothesis is true. This misinterpretation is extremely common and completely wrong. It is a statement about the data given the hypothesis, not about the hypothesis given the data.

**Report p-values, not stars.** Hansen is explicit: asterisk conventions (`*` for 10%, `**` for 5%, `***` for 1%) throw away information that the p-value carries at no extra cost. Report the p-value or, better, the confidence interval.

## "Accept" vs "do not reject"

Failing to reject is not evidence for the null. It means you had insufficient evidence against it — which could be because the effect is zero, or because your data are uninformative. Without a power analysis you cannot tell.

Writing "the regression finds that female union membership has no effect on wages" because `t = 1.2` is, in Hansen's words, an incorrect and most unfortunate interpretation. Correct: "we cannot reject the hypothesis of no effect; the 95% interval is `[-0.02, 0.06]`, which excludes effects larger than 6%."

## Statistical vs economic significance

This is the point Hansen hammers hardest.

With enough data, standard errors become tiny and *everything* becomes statistically significant, including effects far too small to matter. A wage effect of 1% with a standard error of 0.4% is significant and economically negligible.

Conversely, an economically large effect can be statistically insignificant in a small sample. That is a statement about your data, not about the world.

**The fix**: focus on the point estimate, its magnitude in interpretable units, and its confidence interval. "Married men earn 19-23% more" is informative. "The married-male coefficient is highly significant" is nearly content-free.

## t-ratios and the abuse of testing

Hansen's section 9.8 is worth quoting the substance of:

- Report **estimates and standard errors**, not t-ratios. Standard errors convey precision; t-ratios convey only significance, and the reader can compute the ratio if they want it.
- Papers that "describe which variables are significant and the signs of the coefficients" are doing very poor empirical work. Hansen calls it a recipe for banishment to lower-tier journals.
- Software prints a t-statistic and p-value for *every* coefficient by default. This unintentionally suggests that the entire list of hypotheses "this control equals zero" is interesting. It usually is not. Test the hypotheses you actually have a scientific reason to test.
- The default t-test is a test of `β = 0`. Discuss it when zero is an economically interesting value. Often it is not — for a control variable, "zero" is not a hypothesis anyone cared about.

## Multiple testing

If you run `k` independent 5% tests and all nulls are true, the chance that at least one rejects is roughly `1 - 0.95^k` — about 40% for `k = 10`, 92% for `k = 50`. Examining a large table of coefficients and reporting the significant ones is a machine for generating false positives.

**Bonferroni correction**: to control the **familywise error rate** (probability of *any* false rejection) at `α`, require each individual p-value to be below `α/k`. Equivalently, report the Bonferroni familywise p-value `k·min p`.

Derivation is Boole's inequality: `P[min pⱼ < α] ≤ Σ P[pⱼ < α] = kα`.

*Example.* Two coefficients with p-values 0.04 and 0.15. Individually the first "is significant". The Bonferroni familywise p-value is `2 × 0.04 = 0.08` — not significant at 5%. If instead the p-values were 0.01 and 0.15, familywise `p = 0.02`, which is significant.

Bonferroni is conservative (it ignores dependence between tests). Alternatives: Holm's step-down procedure, or false discovery rate control (Benjamini-Hochberg) when you have many tests and care about the *proportion* of false discoveries rather than avoiding any.

**Related discipline**: pre-registration of hypotheses, and reporting how many specifications you ran. Silent specification search is the biggest single source of non-replicable empirical results.

## Power and test consistency

**Test consistency**: for any fixed alternative, the rejection probability → 1 as `n → ∞`. t-tests and Wald tests are consistent, which is reassuring but not very informative — it says nothing about power at your actual sample size.

**Asymptotic local power**: the useful tool. Index the alternative as `θₙ = θ₀ + h/√n` — an alternative that shrinks as the sample grows, so the problem stays "hard". Then the Wald statistic has a **non-central chi-square** limit with non-centrality parameter depending on `h` and the variance. This gives usable power curves and is the basis for power calculations and minimum-detectable-effect computations.

**Do a power calculation before collecting data.** The question "what is the smallest effect I could detect with 80% probability at this sample size?" is far more useful than any post-hoc test.

## Confidence intervals by test inversion

Standard interval: `β̂ ± 1.96·s(β̂)`. This is exactly "the set of values not rejected by a two-sided t-test".

**Test inversion** generalizes this: for any test statistic `T(θ)` and critical value `c`, the set `{θ : T(θ) ≤ c}` is a confidence interval, and its coverage equals 1 minus the test's Type I error rate. So a well-sized test automatically gives a well-covering interval.

This matters most for **nonlinear parameters**. For `θ = β₁/β₂`, the naive delta-method interval can be far too narrow. The fix: rewrite the hypothesis linearly as `θβ₂ = β₁`, form
```
T(θ) = (β̂₁ - β̂₂θ) / (R'V̂R)^(1/2),    R = (1, -θ)'
```
and grid-search over `θ` for the values with `|T(θ)| ≤ 1.96`.

Hansen's wage-peak example: delta method gives `[29.8, 29.9]`, test inversion gives `[29.1, 30.6]`. The second is "greatly preferred".

## Practical checklist

1. State the hypothesis you care about before looking at the output.
2. Use robust or clustered standard errors appropriate to the data. See [[06 - Standard Errors and Clustering]].
3. Report estimate, standard error, confidence interval, and p-value. Skip the stars.
4. Interpret magnitude in economic units.
5. If you tested many things, adjust or disclose.
6. For nonlinear functions of coefficients, use test inversion or the bootstrap rather than the delta method.
7. Never write "no effect" when you mean "not statistically significant".

Related:

- [[03 - Statistical Inference Foundations]]
- [[06 - Standard Errors and Clustering]]
- [[07 - Asymptotic Theory]]
- [[09 - Bootstrap and Resampling]]
- [[19 - Applied Workflow and Common Mistakes]]
