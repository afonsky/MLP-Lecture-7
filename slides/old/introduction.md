---
title: Linearity in Linear Regression
---

# Linearity in Linear Regression

* Consider observation $X_{p \times 1}$ and estimated parameters $\hat{\beta}_{p \times 1}$
	* We'll ignore the intercept $\hat{\beta}_0$ to simplify notation for now 
* We assume a **linear** conditional expectation, $\mathbb{E}[Y|X]$. What does it mean?
	* $\hat{Y}(X|\hat{\beta}) := \sum_{j=1}^p \hat{\beta}_j X_j = X \hat{\beta}$ is linear in its **parameters** $\hat{\beta}_j$ and linear in its **arguments** $X_j$
	* $\hat{Y}(X|\hat{\beta}) := \sum_{j=1}^p \hat{\beta}_j \textcolor{red}{f_j}(X)$ is linear in its **parameters**, but **may not** be linear in arguments $X_j$
		* Ex. $\hat{Y} := \hat{\beta}_1 X_1 + \hat{\beta}_2 X_2^2 + \hat{\beta}_3 X_1 X_2$ has nonlinear quadratic and interaction terms
* If linearity fails, we can still find **least-squares estimation (LSE)** line by minimizing residual sum of squares, $\mathrm{RSS}(b) = \sum_{i=1}^N (y_i - \hat{y}_i)^2$
	* Linearity is not binary (i.e. exists or not). $\beta$'s express the **degree** and **certainty** of linearity
	* LSE is a computational method and makes **no statistical assumptions**, except linear independence of features. I.e. its only requirement is invertibility of $X^\prime X$
<div class="grid grid-cols-[1fr,7fr,2fr]">
<div>
</div>
<div>

* If $X^\prime X$ is singular, identify and drop perfectly multicollinear columns
* In computer vision, mono-color background pixels (in corners) may need to be regularized
</div>
<div>
  <figure>
    <img src="/ESL_figure_1.2.png" style="width: 130px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=161">ESL Fig. 1.2</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Orthogonal Projection & Least Squares Estimates

* The solution of minimization of $\mathrm{RSS}(b_{p \times 1} | X_{N \times p}, y_{N \times 1}) = ||y - Xb||_2^2$ is<br>
$\hat{\beta}(X, y) := \underset{b}{\mathrm{argmin}} ~\mathrm{RSS}(b | X, y) = (X^\prime X)^{-1} X^\prime y$<br>
computed by setting $0 = \nabla_b \mathrm{RSS} = -2 X^\prime (y - Xb)$
	* Setting $0 = X^\prime (y - Xb)$ finds $b$ such that $X^\prime$ and $y - Xb$ are **orthogonal**
	* This yields $\hat{y}$ as **orthogonal projection** of $y_{N \times 1}$ on **column space** of $X$, which **spans** $\R^N$
	* The difference $y - \hat{y}$ is a **residual** vector $\hat{\varepsilon}_{N \times 1}$
<div class="grid grid-cols-[4fr,2fr]">
<div><br>

* The orthogonal projection is captured by<br> **projection matrix** (a.k.a. "**hat**" matrix), $H$
	* $H$ "puts a hat" on $y$: $~~~\hat{y} := \textcolor{red}{X} \hat{\beta} = \color{grey}\underbrace{\color{#006} \textcolor{red}{X}(X^\prime X)^{-1}X^\prime}_{H_{N \times N}} \color{red} y \color{#006} = H y$
* Note: [sample covariance](https://en.wikipedia.org/wiki/Sample_mean_and_covariance) of $X_{N \times p}$ is $\hat\Sigma_X := \frac{1}{N} X_C^\prime X_C$
	* where $X_C$ is a matrix with centered columns
	* So, $\hat\Sigma_X \neq X^\prime X$, unless $X = X_C/\sqrt{N}$
</div>
<div>
  <figure>
    <img src="/Le1iu.png" style="width: 290px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://stats.stackexchange.com/questions/108591/what-is-the-importance-of-hat-matrix-h-xx-top-x-1-x-top-in-linear-reg">https://stats.stackexchange.com/questions/108591/what-is-the-importance-of-hat-matrix-h-xx-top-x-1-x-top-in-linear-reg</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Properties of Least Squares Estimates, $\beta_j$

* The following assumptions are needed for statistical inference, not to compute LSE
	* <font color='red'>If</font> we assume **uncorrelated** $y_i$, **constant** $\sigma^2 := \sigma_\varepsilon^2$, and **fixed** $x_i$, we can estimate:
		* $\mathbb{V}\hat\beta = (X^\prime X)^{-1} \sigma^2$, where $\sigma^2$ is estimated as $\hat\sigma^2 := \frac{\mathrm{RSS}}{N - p - 1}$
			* The degree of greedom (DF) $N - p - 1$ yields an **unbiased** estimator, i.e. $\mathbb{E}\hat\sigma^2 = \sigma^2$
* Also, <font color='red'>if</font> we assume **linearity** $y = \mathbb{E}[Y | X] + \varepsilon = X\beta + \varepsilon$ and $\varepsilon \sim N(0, \sigma^2)$, then we have
<div class="grid grid-cols-[3fr,2fr]">
<div><br>

$\hat\beta_{p \times 1} \sim N(\beta, \sigma_{\hat\beta} := (X^\prime X)^{-1} \hat\sigma^2)$<br>
$(N - p - 1) \hat\sigma^2 \sim \sigma^2 \chi_{N - p - 1}^2$,<br>
a **chi-squared** distribution with<br> $\mathrm{DF} = N - p - 1$, $~\hat\beta \perp \hat\sigma^2$
* Thus, we can estimate uncertainty and confidence bands around our estimates: $\hat\beta \pm 2 \cdot \mathrm{SE}(\beta)$
</div>
<div>

<font size="2pt">Higher degree collinearity of features yields greater uncertainty about </font> $\beta$
```r {all} {maxHeight:'50px'}
solve(matrix(c(1,1,2,2), nrow=2, byrow=T))
> Error in solve.default(matrix(c(1, 1, 2, 2), nrow = 2, byrow = T)):
> Lapack routine dgesv: system is exactly singular: U[2,2] = 0

solve(matrix(c(1,1,2,1), nrow=2, byrow=T))
> A matrix: 2 × 2 of type dbl
> -1 1
> 2 -1

solve(matrix(c(1,1,2,2.00000001), nrow=2, byrow=T))
```
</div>
</div>

---

# Testing for Linearity

* With previous assumptions, we can formally test if $\beta_j = 0$
	* i.e. whether $X_j$ feature is linearly related to $y$
	* Define **Z-score**: $z_j := \hat\beta / \sigma_{\hat\beta_j}$, where $\sigma_{\hat\beta_j}$ is a $j$th diagonal element of $\sigma_{\hat\beta}$
	* $z_j \sim t_{N - p - 1}$ and we can compute the $p$-value to decide on linearity hypothesis at, say, significance level $\alpha = 0.05$
		* We use Student's t-distribution because we estimated $\sigma$ of the Gaussian error $\varepsilon$ as $\hat\sigma$
		* Since t-distribution is similar to $N(0, 1)$, especially with large DF, we often make $H_0$ decision using Gaussian assumption on $z_j$
* To test all $\beta_j = 0$ simultaneously (Eg. at each step of stepwise regression), we use:
$$\mathrm{F_{stat}} := \frac{(\mathrm{RSS}_0 - \mathrm{RSS}_1) / (p_1 - p_0)}{\mathrm{RSS}_1 / (N - p - 1)} \sim \color{grey}\underbrace{\color{#006}{\mathcal{F}_{p_1 - p_0, N - p - 1}}}_{\mathrm{F-distribution}}$$

where $\mathrm{RSS}_1$, $\mathrm{RSS}_0$ are for larger and smaller models, respectively, i.e. $p_1 > p_0$

---

# Ex. Linearity: Prostate Cancer, [ESL p. 49](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=68)

* Variables:
	* Response $Y$:
		* `lpsa`: log of prostate-specific antigen (PSA)
	* Features $X_j$:
		* `lcavol`: log cancer volume
		* `lweight`: log prostate weight
		* `age`
		* `lbph`: log of the amount of benign prostatic hyperplasia
		* `svi`: seminal vesicle invasion
		* `lcp`: log of capsular penetration
		* `gleason`: Gleason score
		* `pgg45`: % Gleason scores 4 or 5

---

# Ex. Linearity: Prostate Cancer, [ESL p. 49](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=68)

<div class="grid grid-cols-[2fr,3fr]">
<div>

* Our **key concern** is correlation between $X_j$ and the response **lpsa**
* **Strong** $\rho_{Y, X_J}$ favors **direct** application of linear regression
* **Weak** $\rho_{Y, X_J}$ implies weak linear dependence, but **non-linear** dependence may still exist
* Correlations among predictors are of **secondary** concern
* Resolve **perfect** collinearity before inverting $X^\prime X$
* **Strong** collinearity of $X_j, X_{j^\ast}$ yields highly variable $\hat\beta_j, \hat\beta_{j^\ast}$, but you can still make good predictions with this model
</div>
<div>
  <figure>
    <img src="/lpsa.png" style="width: 490px !important">
  </figure>
```r {all}
library(PerformanceAnalytics)
df = read.table('prostate.data')
chart.Correlation(df[df$train==T, 1:9])
```
</div>
</div>

---

# Gauss-Markov Theorem (GMT)

* *GMT*:<br>
Among all **linear unbiased** estimators of $\beta_{p \times 1}$, the least squares estimator, $\hat\beta$, is “best” (i.e. **least-squares**).
	* i.e. $\mathbb{V} \color{grey}\underbrace{\color{#006}{a^\prime \hat\beta}}_{\hat\theta} \color{#006} \leq \color{grey}\underbrace{\color{#006}{\mathbb{V} c^\prime y}}_{\tilde\theta}$ for any $a, c$ such that $\mathbb{E} \hat\theta = \mathbb{E} \tilde\theta = a^\prime \beta =: \theta$

* GMT guarantees that your $\mathrm{MSE}(\hat\theta) = \mathbb{V}\hat\theta + \color{grey}\underbrace{\color{#006}{[\mathbb{E}\hat\theta -\theta]^2}}_{\mathrm{Bias^2}\hat\theta = 0} \color{#006} = \mathbb{V}\hat\theta$ is lowest in its “group”
	* Yet, you can still find a **lower MSE estimator** outside of linear unbiased estimators!

---

# Multiple Regression from Simple Linear Regression

* For $X_{N \times 1}$ and model $Y_{N \times 1} = X\beta + \varepsilon$ we estimate the slope as $\hat\beta := \frac{x^\prime y}{x^\prime x} = \frac{\sum_i x_i y}{\sum_i x_i^2}$<br>
We extend this univariate model to multivariate with **orthogonal** (uncorrelated) features
* For $X_{N \times p} = \bm{x}_{1:p}$, $\bm{x}_j^\prime \bm{x}_{k \neq j} = 0$, model $Y_{N \times 1} = X\beta + \varepsilon$ we estimate $\hat\beta := \frac{\bm{x}_j^\prime y}{\bm{x}_j^\prime \bm{x}_j}$
	* i.e. orthogonal inputs result in diagonal covariance matrix of $\hat\beta$ (no relations b/w $\hat\beta_j$ & $\hat\beta_{k \neq j}$)
* We further extend it to **non-orthogonal** (i.e. correlated) features via change of basis from $\bm{x}_{1:p}$ to the orthogonal basis $Z_{N \times p} = \bm{z}_{1:p}$ in the same column space as $\bm{x}_{1:p}$
	* [Gram-Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) (GSP) iteratively computes $\bm{z}_j$, which is orthogonal to $\bm{z}_{1:j-1}$

<div class="grid grid-cols-[1fr,27fr,10fr]">
<div>
</div>
<div>

* Each $\bm{x}_j$ is a linear combination of $\bm{z}_{1:p}$, i.e. $X = Z \Gamma$
	* where $\Gamma_{p \times p}$ is a matrix of regression coefficients from GSP
* Using **QR decomposition** of $X$, we deduce $\hat\beta = (D \Gamma)^{-1} (Z D^{-1})^\prime y$
	* $D_{p \times p} = \mathrm{diag}\{||\bm{z}_j||, \forall j\}$, see [ESL p.55](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=74)
</div>
<div>
  <figure>
    <img src="/ESL_figure_3.4.png" style="width: 280px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=73">ESL Fig. 3.4</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Gram-Schmidt Process

* We used the same method to compute orthogonal principal components (PCs)
* *Algorithm*
	1. Initialize $\bm{z}_0 = \bm{x}_0 = 1$
	2. For $j = 1, 2, ..., p$<br>
			Regress $\bm{x}_j$ on $\bm{z}_0, \bm{z}_1, ..., \bm{z}_{j-1}$<br> to produce coefficients $\hat\gamma_{\ell j} = \frac{<\bm{z}_\ell, \bm{x}_j>}{<\bm{z}_\ell, \bm{z}_\ell>}$, $\ell = 0, ..., j - 1$<br> and residual vector $\bm{z}_j = \bm{x}_j - \sum\limits_{k=0}^{j-1} \hat\gamma_{kj}\bm{z}_k$
	3. Regress $y$ on the residual $\bm{z}_p$ to give the estimate $\hat\beta_p$.

---

# Multiple Outputs, $Y_{N \times K}$
<div class="grid grid-cols-[4fr,2fr]">
<div>

* Ex. Given telemetry feed (e.g. camera, lidar, radar) from a self-driving car, estimate speed, acceleration, distance to nearest object on each side
	* There might be correlation between outputs
		* Ex. Slower driving cars can drive closer to one another
</div>
<div>
  <figure>
    <img src="/When-will-we-see-driverless-cars-on-British-roads.jpg" style="width: 220px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://www.businessleader.co.uk/when-will-we-see-driverless-cars-on-british-roads/79454/">https://www.businessleader.co.uk/when-will-we-see-driverless-cars-on-british-roads/79454/</a>
    </figcaption>
  </figure>
</div>
</div>

* For **uncorrelated** $Y_{1:k}$, the model is<br>
$Y_{N \times K} = X_{N \times p} B_{p \times K} + E_{N \times K}$ with a loss function<br>
$\mathrm{RSS}(B) := \sum\limits_{k=1}^K \sum\limits_{i=1}^N (y_{ik} - \hat{y}_{ik})^2 = \mathrm{trace}[(Y - XB)^\prime (Y - XB)]$, where $\hat{B} := (X^\prime X)^{-1} X^\prime Y$

* For **correlated** $Y_{1:k}$ , i.e. $\Sigma_{K \times K} := \mathbb{V}E$ is the covariance matrix of $E$ error terms, we have<br>
$\mathrm{RSS} (B | \Sigma) := \sum\limits_{i=1}^N (y_i - \hat{y}_i)^\prime \Sigma^{-1}(y_i - \hat{y}_i)$, where $y_i \in \R^K$<br> and again, the estimator is $\hat{B} := (X^\prime X)^{-1} X^\prime Y$

---

# Subset Selection
* Too many features in a model diminishes interpretability and prediction accuracy$^\ast$
	* $^\ast$ this is because large $p$ attracts highly correlated features and yields unstable $\beta_j$
* **Best Subset (BS) selection**: tries all possible combinations of features
	* Likely to overfit. Computationally intractable for $\gtrsim 40$ features (on a single CPU)
		* Note: we can embarrassingly parallelize this model selection (no dependency among models)
	* Use theoretical metrics (not cross validation (CV)) for performance reasons: $R^2, \mathrm{AIC}, ...$
* **Forward Stepwise (Fwd) selection**: iteratively/greedily add the next “best” feature until all features are used up to build models $\mathcal{M}_{0:p}$. Then choose the best model.
* **Backward Stepwise selection**: similar to Fwd, but starts with a full set of features
	* Can fail for in case of $p > n$, if model can’t handle **wide** data matrices
* **Hybrid (mixed) Stepwise selection**: a dance between Forward and Backward
* If $\mathcal{X}_k^\ast =$ features in model $^\ast$ with $k$ features:
	* There is no guarantee that $\mathcal{X}_k^\mathrm{BS} \subset \mathcal{X}_{k+1}^\mathrm{BS}$
	* We always have $\mathcal{X}_k^\mathrm{Fwd} \subset \mathcal{X}_{k+1}^\mathrm{Fwd}$

---

# Forward Stagewise Regression
* Similar to Forward Stepwise, but more restricted and slower to converge
	* However, it tends to work better with high dimensions
* *Algorithm*
	1. Initialize: Center all predictors as $\bm{x}_j \leftarrow \bm{x}_j - \bar{\bm{x}}, \beta := 0$, 
	2. Step $s = 0$: Start with no features and $\hat{y} := \bar{y}$
	3. Compute model residual $r_s$ of the step $s$
	4. Find $\bm{x}_j$ most correlated with $r_s$ and compute $\hat\beta_j$ by via SLR $r_s$ on $\bm{x}_j$
		* Note: previous coefficients are not adjusted

---

# Shrinkage Methods

* We introduce a global penalty on model parameters, which forces them to zero
* **Ridge** penalized RSS and the solution for the **centered** matrix $X_{N \times p}$<br>
$\hat\beta^{\mathrm{ridge}} := \underset{b}{\mathrm{argmin}} \{\mathrm{RSS}(b) - \textcolor{red}{\lambda} ||b||_2^2\} = (X^\prime X + \textcolor{red}{\lambda} I_{p \times p})^{-1} X^\prime y$
	* Dual formulation: $\hat\beta^{\mathrm{ridge}} := \underset{b}{\mathrm{argmin}} \{\mathrm{RSS}(b) |~||b||_2^2 \leq \textcolor{red}{t}\}$
		* Higher penalty $\textcolor{red}{\lambda}$ (or lower $\textcolor{red}{t}$) shifts focus from minimizing RSS to minimizing magnitude of $b$
	* Effective degrees of freedom is $\mathrm{EDF}(\textcolor{red}{\lambda}) = \mathrm{trace}[X(X^\prime X + \textcolor{red}{\lambda}I)^{-1} X^\prime] \leq p$
* Lasso penalized RSS: $\hat\beta^{\mathrm{lasso}} := \underset{b}{\mathrm{argmin}} \{\frac{1}{2}\mathrm{RSS}(b) - \textcolor{red}{\lambda} ||b||_1\}$
	* Dual formulation: $\hat\beta^{\mathrm{lasso}} := \underset{b}{\mathrm{argmin}} \{\mathrm{RSS}(b) |~||b||_1 \leq \textcolor{red}{t}\}$
	* Coefficients are estimated with quadratic programming
* $\textcolor{red}{\lambda}$ increases: all coefficients decrease,  $\hat\beta^{\mathrm{lasso}}$ snap to zero, but  $\hat\beta^{\mathrm{ridge}}$ may not
	* For “small enough” $\textcolor{red}{\lambda}: ~\hat\beta^{\mathrm{lasso}} = \hat\beta^{\mathrm{ridge}} = \hat\beta^{\mathrm{LSE}}$. With lasso we just need to have $\textcolor{red}{t} > ||\hat\beta^{\mathrm{LSE}}||_1$
	* For “large enough” $\textcolor{red}{\lambda}: ~\hat\beta^{\mathrm{lasso}} = \hat\beta^{\mathrm{ridge}} = 0$
	* See [ESL p.64](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=83) for more discussion of singular value decomposition (SVD)

---

# Ex. Prostate Cancer with $p = 8$

* Observe the shrinkage of coefficients as $\lambda$ grows (effective $\mathrm{DF}(\lambda)$ and $s$ drop):
	* $\hat\beta^{\mathrm{lasso}}$ decrease monotonically, but $\hat\beta^{\mathrm{ridge}}$ do not (and even flip sign). Why?
	* Shrinkage factor $s := \frac{t}{||\hat\beta^{\mathrm{LSE}}||_1}$
* We estimate $\lambda$ via CV
<div class="grid grid-cols-[2fr,2fr]">
<div>
  <figure>
    <img src="/ESL_figure_3.8.png" style="width: 260px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=84">ESL Fig. 3.8</a>
    </figcaption>
  </figure>
</div>
<div>
  <figure>
    <img src="/ESL_figure_3.10.png" style="width: 259px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=89">ESL Fig. 3.10</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Generalized Penalties

* Generally, we can have any $q$-norm ($L_q$ penalty) penalized RSS:<br>
$\hat\beta^{L_q} := \underset{b}{\mathrm{argmin}} \{\mathrm{RSS}(b) - \textcolor{red}{\lambda} ||b||_q^q\}$
* We can also use Elastic Net, a weighted average of Lasso & Ridge penalties<br>
$\hat\beta^{elnet} :=\underset{b}{\mathrm{argmin}} \{\mathrm{RSS}(b) - \textcolor{red}{\lambda} (\textcolor{red}{\alpha} ||b||_1^1 + (1 - \textcolor{red}{\alpha}) ||b||_2^2) \}, ~\alpha \in [0, 1]$
	* It offers a compromise between Lasso and Ridge<br>

<br>

<div class="grid grid-cols-[5fr,2fr]">
<div>
  <figure>
    <img src="/ESL_figure_3.12.png" style="width: 560px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=91">ESL Fig. 3.12</a>
    </figcaption>
  </figure>
</div>
<div>
  <figure>
    <img src="/ESL_figure_3.13.png" style="width: 200px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=92">ESL Fig. 3.13</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Least Angle Regression (LAR)

* Similar to Forward Stepwise regression
	* Iteratively, we look for the next highly correlated feature 
	* Then identify “excess” correlation in the current model
	* All currently added coefficients are gradually increased towards LSE
* *Algorithm*
	1. Standardize the predictors to have mean zero and unit norm. Start with the residual<br> $\bm{r} = \bm{y} - \bar{\bm{y}}, ~\beta_1, \beta_2, ..., \beta_p = 0$
	2. Find the predictor $\bm{x}_j$ most correlated with $\bm{r}$
	3. Move $\beta_j$ from 0 towards its least-squares coefficient $<\bm{x}_j, \bm{r}>$, until some other competitor $\bm{x}_k$ has as much correlation with the current residual as does $\bm{x}_j$
	4. Move $\beta_j$ and $\beta_k$ in the direction defined by their joint least squares coefficient of the current residual on $(\bm{x}_j, \bm{x}_k)$, until some other competitor $\bm{x}_l$ has as much correlation with the current residual
	5. Continue in this way until all $p$ predictors have been entered. After $\mathrm{min}(N − 1, p)$ steps, we arrive at the full least-squares solution.

---

# Derived Input Models

* **Principal Component Regression (PCR)**
	* PCA on inputs [``%>%``](https://stackoverflow.com/questions/27125672/what-does-function-mean-in-r) linear regression
		* New inputs $\bm{z}_{1:k}$ are derived as linear combinations of the original $\bm{x}_{1:p}$ features, $k \leq p$
		* Feature space compression is done **without** considering the response $y$

* **Partial Least Squares (PLS)**
	* Also shrinks input space, but uses $y$ to do so

* Remember to standardize features in both methods
	* They use distance metric, which favors features with larger values

---

# Dantzig Selector
<div class="grid grid-cols-[5fr,2fr,2fr]">
<div>

* Authors: [Emmanuel Candes](https://profiles.stanford.edu/emmanuel-candes) and [Terence Tao](https://en.wikipedia.org/wiki/Terence_Tao) ([2007](https://arxiv.org/abs/math/0506081))
* Uses $L_\infty$ [norm](https://en.wikipedia.org/wiki/Lp_space#Definition) = max absolute value of elements,<br> i.e. $||\bm{u}||_\infty = \underset{i}{max} \{|u_i| ~|~\forall i\}$
</div>
<div>
  <figure>
    <img src="/Candes.png" style="width: 150px !important">
  </figure>
</div>
<div>
  <figure>
    <img src="/Tao.png" style="width: 150px !important">
  </figure>
</div>
</div>

* Formulation (either):<br>
$\underset{b}{\mathrm{argmin}} ||b||_1~~~~~~~~~~~~~~~~~~~~~~~~$	subject to $~~||X^\prime (y - X^\prime b)||_\infty \leq \textcolor{red}{s}$<br>
$\underset{b}{\mathrm{argmin}} ||X^\prime (y - X^\prime b)||_\infty~~~$	subject to $~~||b||_1 \leq \textcolor{red}{t}$<br>

These can be solved via linear programming
* **Good**: recovers sparse coefficient (like Lasso does)
* **Bad**: it can include features with lower correlations with response. Hence,...
	* poorer predictive accuracy and 
	* unstable coefficient paths (when selecting a regularizer)

---

# Grouped Lasso
* Predictors may belong to some pre-defined groups or clusters
	* We can use prior knowledge or clustering methods to identify these
	* We may want to shrink grouped features in a similar way
* Ex. Groups of genes, cells or patients in the problem of predicting some disease
* We have $L$ groups, each with $p_\ell$ features captured by $X_\ell$ data matrix. We solve<br>
$\underset{b}{\mathrm{argmin}} \bigg\{||y - \sum\limits_{\ell=1}^L X_\ell b_\ell||_2^2 + \textcolor{red}{\lambda}\sum\limits_{\ell=1}^L \sqrt{p_\ell} ||b_\ell||_2\bigg\}$
	* where $b_\ell \in \R^{p \ell}$
		* As $\lambda$ increases entire groups of parameters are zeroed out

---

# Further Reading in ESL Ch.3
* Investigate:
	* Relaxed lasso
	* Adaptive lasso
	* SCAD
	* etc.

---

# Gauss-Markov Theorem Proof ([Ex. 3.3](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=113))

* Prove the Gauss–Markov theorem: the least squares estimate of a parameter $a^T \beta$ has variance no bigger than that of any other linear unbiased estimate of $a^T \beta$.
<hr>

Let $b$ be a column vector of length $N$, and let $\mathbb{E}(b^T y) = \alpha^T \beta$. Here $b$ is fixed, and the
equality is supposed true for all values of $\beta$. A further assumption is that $X$ is not random.
Since $\mathbb{E}(b^T y) = b^T X \beta$, we have $b^T X = \alpha^T$. We have $\mathbb{V}(\alpha^T \hat\beta) = \alpha^T (X^T X)^{-1} \alpha = b^T X (X^T X)^{-1} X^T b$, and $\mathbb{V}(b^T y) = b^T b$.<br>

So we need to prove $X (X^T X)^{-1} X^T \preceq I_N$.<br>

To see this, write $X = QR$ where $Q$ has orthonormal columns and is $N \times p$, and $R$ is $p \times p$ upper triangular with strictly positive entries on the diagonal.

Then $X^T X = R^T Q^T Q R = R^T R$. Therefore $X(X^T X)^{-1} X^T = QR(R^T R)^{-1} R^T Q^T = QQ^T$.<br>
Let $[QQ_1]$ be an orthogonal $N \times N$ matrix.<br>
Therefore $I_N = [Q Q_1] \cdot \begin{bmatrix} Q^T\\ Q_1^T \end{bmatrix} = Q Q^T + Q_1 Q_1^T$<br>

Since $Q_1 Q_1^T$ is positive semidefinite, the result follows.