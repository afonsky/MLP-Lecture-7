---
title: Basis expansion
---

# Basis Functions

* Polynomials and piecewise constant functions generalize as **basis functions**, 
<br> $y | x = \beta_0 + \sum\limits_{j=1}^{K}\textcolor{red}{\beta_j} b_j(x) + \epsilon$
	* Disadvantage: $b_j$ are specified in advance (supplied by scientist), not learned
  * Advantage: least squares estimation (i.e. minimizing MSE) still works
    * So, we can still use $p$-values for coefficients, $\mathrm{F}$-test for model’s significance, residual evaluation, etc.
* Polynomial regression: $b_j(x) := x^j$
  * It’s a sum of monomial functions
* Piecewise constant regression: $b_j(x) = I(c_j \leq X \leq c_{j+1})$
* **Regression splines**: 
	* Bases are interactions of polynomials with indicator functions


---

# Basis Expansion

* We now introduce non-linear **basis functions**, $h_m: \R^p \rightarrow \R$, in our linear additive models
<br> $f (X_{\textcolor{grey}{p \times 1}}) := \sum\limits_{m=1}^{M} \beta_m h_m(x)$ is a linear basis expansion in $X$
* $h_m(X)$ can be:
  * **Global**, i.e. tuning the fit in one region has effect on all other regions
    * E.g. quadratic terms $X_j^2$, **interaction terms** $X_j X_k$, or $X_j^p$, $\log X_j$, $\sqrt{X_j}$, ...
* **Local**, allowing tuning fit in different regions. E.g.:
  * Indicators $I_{[L_m, U_m)} (X)$
  * Piecewise linear b/w **knots** $\xi_m$, $(X - \xi_m)_{+}$
  * Piecewise cubic $(X - \xi_m)_{+}^3$
  * Wavelets, harmonics
* Problem:
  * **Exponential growth** of complexity of the model (i.e. too many features)
    * $\mathcal{O}(p^d)$, where $p$ is # of original features, $d$ is degree of polynomial

---

# How to control the complexity of basis expansion?

Let $\mathcal{D}$ be the set/dictionary of all considered basis functions. $|\mathcal{D}|$ is cardinality of $\mathcal{D}$
* **Model Restriction**
  * Reduced $|\mathcal{D}|$ and use simpler models with high bias and low variance
    * E.g. Linear and logistic regressions, **piecewise linear/polynomial, splines, wavelets**
* **Feature Selection**
  * Original $|\mathcal{D}|$, but only “important” features included in the model
    * E.g. stepwise model selection, best subset selection, CART, MARS
* **Feature Regularization**
  * Original $|\mathcal{D}|$, but “unimportant” features have near zero coefficients (i.e. regularized)
    * E.g. Lasso, Ridge, Elastic Net
* **Dimension reduction** with **derived** features
  * Original $|\mathcal{D}|$. Transform high dimensional feature space to a smaller dimensional space.
    * All features from  are used in the model, but features are “grouped” into a smaller set of higher-level features
    * E.g. [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis), [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization)