---
title: Causality and Identification
tags: [econometrics, causal-inference, potential-outcomes, identification, beginner]
---

# Causality and Identification

Source: Hansen, *Econometrics*, chapter 2 (section 2.30) and chapters 12, 17-21.

## Potential outcomes

The Rubin causal model. For each unit there is an outcome function

```
Y = h(D, U)
```

where `D` is treatment and `U` collects everything unobserved about that unit. Write the two **potential outcomes**:

- `Y(0) = h(0, U)` — the outcome if untreated
- `Y(1) = h(1, U)` — the outcome if treated

The **individual causal effect** is `C = Y(1) - Y(0)`.

**The fundamental problem of causal inference**: you observe only one of the two.

```
Y = Y(0) if D = 0
Y = Y(1) if D = 1
```

The other is a counterfactual and is never in your data. This is a *missing data* problem, not an estimation problem, and it is why causal inference needs assumptions that no amount of data can test.

Two consequences follow immediately:

1. Causal effects are **individual-specific**. Hansen's example: Jennifer would earn $10/hr with high school and $20/hr with college (effect $10); George would earn $8 and $12 (effect $4). The "effect of college" is not one number.
2. Because they are individual-specific and unobserved, we settle for **averages** of them.

## The estimands

| Estimand | Definition | Reads as |
| --- | --- | --- |
| **ATE / ACE** | `E[Y(1) - Y(0)]` | Effect if everyone were treated vs everyone untreated |
| **ATT** | `E[Y(1) - Y(0) \| D = 1]` | Effect on those who actually got treated |
| **ATU** | `E[Y(1) - Y(0) \| D = 0]` | Effect on those who did not |
| **CATE** | `E[Y(1) - Y(0) \| X = x]` | Effect for a subgroup |
| **LATE** | `E[Y(1) - Y(0) \| compliers]` | Effect on those an instrument moves |

These are different numbers and different methods estimate different ones. Randomized experiments estimate the ATE. Instrumental variables estimate the LATE. Difference-in-differences estimates something like the ATT. Regression discontinuity estimates a CATE at the cutoff. **Always know which one your method delivers.**

## Selection bias, decomposed

The naive comparison is `E[Y | D=1] - E[Y | D=0]`. Decompose it:

```
E[Y|D=1] - E[Y|D=0]
  = E[Y(1) - Y(0) | D=1]              ← ATT, what you want
  + ( E[Y(0)|D=1] - E[Y(0)|D=0] )     ← selection bias
```

The second term asks: would treated units have differed from untreated units even without treatment? If people who go to college would have earned more anyway, the term is positive and the naive comparison overstates the effect.

There is a third term if effects are heterogeneous and correlated with selection ("differential selection into gains"), which is exactly Hansen's Jennifer/George example: Jennifers have both the bigger effect and the higher probability of attending. The naive comparison gives $8.25 when the true ATE is $7.00.

## The identification assumptions, from strongest to weakest

**1. Randomization.** `D` is independent of `(Y(0), Y(1))` by design. Then `E[Y|D=1] - E[Y|D=0] = ATE` exactly. Nothing else needed. This is why RCTs are the benchmark.

**2. Conditional Independence Assumption (CIA)** — also called selection on observables, unconfoundedness, or ignorability. Conditional on covariates `X`, `D` is independent of the unobservables `U`.

Under the CIA, Hansen's Theorem 2.12: the regression derivative with respect to `D` equals the conditional average causal effect,
```
m(1, x) - m(0, x) = ACE(x)
```
and averaging over the distribution of `X` gives the ATE.

Note the CIA is *weaker* than full independence of `U` from `(D, X)`. You only need `D` and `U` independent *after conditioning on `X`*. If `X` is rich enough this may be plausible. In the Jennifer/George example, the aptitude test score is exactly such an `X`: conditional on the test score, college attendance probability is the same (3/4 for high, 1/4 for low) for both types, so type and attendance become independent. Include the test score and the regression recovers the truth ($5.50 for low scorers, $8.50 for high scorers, average $7.00).

**3. Overlap / common support.** For the CIA to be usable, every covariate cell must contain both treated and untreated units — `0 < P(D=1 | X=x) < 1`. If some subgroup is always treated, you have no counterfactual for it and no amount of regression fixes that. Regression will silently extrapolate; matching methods at least tell you.

**4. Instrument exogeneity + relevance + monotonicity.** When the CIA is not credible, find a variable `Z` that shifts `D` but affects `Y` only through `D`. See [[11 - Instrumental Variables]].

**5. Parallel trends.** When you have before/after data on treated and control groups. See [[14 - Difference in Differences]].

**6. Continuity at a cutoff.** When treatment is assigned by a threshold rule. See [[16 - Nonparametrics, Quantiles, and RDD]].

**7. Strict exogeneity within unit.** When you observe units repeatedly and the confounder is time-invariant. See [[13 - Panel Data]].

## Identification, defined

A parameter is **identified** if it is uniquely determined by the distribution of the observable data. Identification is a property of the *population problem*, prior to and independent of estimation.

Three tiers of failure:

- **Not identified**: no amount of data determines the parameter. Example: the effect of a time-invariant variable in a fixed-effects model. More data does not help. Ever.
- **Weakly identified**: technically identified but the data contain very little information. Standard asymptotics break down. Example: weak instruments — the estimator is inconsistent, median-biased, and non-normal, and t-statistics can diverge while the true precision is terrible.
- **Point identified but with an unverifiable assumption**: the usual applied situation. The estimand equals a function of observables *if* the CIA (or parallel trends, or exclusion) holds. The assumption is the paper.

**Practical implication**: identification arguments cannot be tested by the data that use them. Testing "parallel pre-trends" is evidence about parallel trends, not proof. A well-written empirical paper spends more words on why the identifying assumption is plausible than on estimation.

## Directed acyclic graphs (a useful complement)

Hansen's books work in the potential outcomes tradition, but DAG reasoning answers "which controls?" cleanly:

- **Confounder** — causes both `D` and `Y`. *Control for it.* (Ability → schooling, ability → wages.)
- **Mediator** — caused by `D`, causes `Y`. *Do not control* if you want the total effect; controlling gives only the direct effect and can introduce bias.
- **Collider** — caused by both `D` and `Y`. *Never control.* Conditioning on it creates a spurious association where none existed.
- **Instrument** — causes `D`, no other path to `Y`. Use as an instrument; do not simply add as a control.
- **Post-treatment variable** — anything realized after treatment. Treat as suspect by default.

The single most common applied error is throwing every available variable into the regression as a "control". Bad controls actively create bias.

## Heterogeneous effects

If effects vary across units — they always do — then:

- The "treatment effect" reported is a weighted average, and the weights depend on the method. Two valid methods can produce different numbers without either being wrong.
- Different instruments identify different LATEs, because they move different compliers. Hansen makes this point explicitly: a distance-to-college instrument and a tuition-cost instrument affect overlapping but different sets of students, and if their causal effects differ, the LATEs differ. This is not a contradiction; it is a feature of the potential-outcomes framework.
- Extrapolating an estimate to a population the method never spoke about is unjustified. IV estimates using college proximity "should be interpreted as applying to the population of students who are incentivized to attend college by the presence of a college within their home county. The estimates should not be applied to other students."

Report heterogeneity when you can: interactions, subgroup estimates, quantile treatment effects.

## The checklist for any causal claim

1. **What is the estimand?** ATE, ATT, LATE, CATE at a point?
2. **What is the source of variation?** Randomization, policy change, cutoff, instrument, within-unit change?
3. **What must be true for that variation to be as-good-as-random?** State the assumption in words.
4. **What would violate it?** Name the most threatening story concretely.
5. **What evidence can you show that is consistent with the assumption?** Balance tests, pre-trends, placebo outcomes, falsification tests. None of these prove it.
6. **For whom does the estimate apply?** What population, what range of the treatment, what time period?
7. **How precise is it?** Confidence interval in economically meaningful units.

Related:

- [[04 - Conditional Expectation and Projection]]
- [[11 - Instrumental Variables]]
- [[13 - Panel Data]]
- [[14 - Difference in Differences]]
- [[16 - Nonparametrics, Quantiles, and RDD]]
