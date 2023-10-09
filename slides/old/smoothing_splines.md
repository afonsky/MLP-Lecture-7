# Smoothing Splines

* Each point is a knot, complexity is regularized with smoothing parameter, $\lambda > 0$ <br>
$\mathrm{RSS}(f, \lambda) :=
\color{grey}\underbrace{\color{#006} \sum\limits_{i=1}^N (y_i - f(x_i))^2}_{\mathrm{raw ~RSS}(f)}
\color{#006}{+}
\color{red} \lambda
\color{#006}{\cdot}
\color{grey}\underbrace{\color{#006} \int (f^{\prime\prime}(t))^2 dt}_{\mathrm{curvature ~penalty}}
$
* For least squares line we have $f(x) = x\theta$, $f^{\prime}(x) = \theta$, $f^{\prime\prime}(x) = 0$
  * So, when $\lambda = 0$, $f$  is unregularized and can be any rough function
  * When $\lambda = \infty$, optimizer effectively chooses a linear approximation (wipes out infinite penalty)
* $N$ natural cubic splines uniquely minimize the RSS: $\mathrm{RSS}(\theta, \lambda) = ||\bm{y}_{\textcolor{grey}{N \times 1}} - \bm{N}\theta||_2^2 + \lambda \theta^T \bm{\Omega}_N\theta$,
where 
  * $f(x) = \sum_{j=1}^N N_j(x)\theta_j = \bm{N}_{\textcolor{grey}{N \times N}} \theta_{\textcolor{grey}{N \times 1}}$, <br> and $N_j(x)$ are $N$-dim set of basis functions representing natural splines
  * $\{\bm{\Omega}_N\}_{jk} := \int N_j^{\prime\prime}(t) N_k^{\prime\prime}(t) dt$
* Penalty is a quadratic term $\theta^T \bm{\Omega}_N\theta$ with a closed form solution from generalized ridge: <br>
$\hat{\theta}(\lambda) := (\bm{N}^T \bm{N} + \lambda \bm{\Omega}_N)^{-1} \bm{N}^T \bm{y}$
  * FYI: inverting a dense $N \times N$ matrix may be excitingly challenging 😉 

---

# Ex. Spinal Bone Mineral Density (BMD)

* We are modeling the **change** in BMD over 1 year period, not the BMD itself
<div class="grid grid-cols-[5fr,4fr]">
<div>

* Smoothing splines demonstrate great flexibility and fit
* Note the $\lambda = 0.00022$ is near 0, allowing more complex shape of the natural cubic spline
<br>

* How to find such small $\lambda$ with CV?

* How many parameters would you fit?
  * $1, 2, ..., \mathrm{million}$?

</div>
<div>
  <figure>
    <img src="/ESL_figure_5.6.png" style="width: 530px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=171">ESL Fig. 5.6</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Find $\lambda$: Brute Force

* Try exponentially increasing values:
  * $10^{-3}$, $10^{-2}$, $10^{-1}$, $10^{0}$, $10^{1}$, $10^{2}$, $10^{3}$, ...
* Then do a more granular search of $\lambda$

* Alternatively, we can simply specify the **effective degrees of freedom** (DF)
  * Which translates to the specific $\lambda$ from a trace of a familiar **smoother matrix**, $\bm{S}_\lambda$

---

# Smoother Matrix, $\bm{S}_\lambda$

* Consider $N$-vector of estimates, $\hat{\bm{f}}$ <br>
$
\color{grey}\underbrace{\color{#006} \hat{\bm{f}}(\lambda)}_{N \times 1}
\color{#006}{~:=}
\color{grey}\underbrace{\color{#006} ~N_~}_{N \times N}
\color{grey}\underbrace{\color{#006} \hat{\theta}(\lambda)}_{N \times 1}
\color{#006}{~:= \bm{N}(\bm{N}^T \bm{N} + \lambda \bm{\Omega}_N)^{-1} \bm{N}^T \bm{y} = \bm{S}_\lambda \bm{y}}
$
* $\bm{S}_\lambda$ is similar ot the "hat matrix", $\bm{H} :=  \bm{X}_{\textcolor{grey}{N \times p}} (\bm{X}^T \bm{X})^{-1} \bm{X}^T$
  * Both are symmetric, positive semi-definite matrices, i.e. $\bm{v}^T \bm{Hv} \geq 0$, $\forall \bm{v}_{\textcolor{grey}{N \times 1}}$
  * $\bm{H}_{\textcolor{grey}{N \times p}}$ is idempotent, $\bm{HH} = \bm{H}$, but $\bm{S}_{\lambda, \textcolor{grey}{N \times N}}$ is shrinking, $\bm{S}_\lambda \bm{S}_\lambda \preccurlyeq \bm{S}_\lambda$
  * $\mathrm{trace}(\bm{H})$, $\mathrm{trace}(\bm{S}_\lambda)$ give us the effective degrees of freedom (DF)
* So, we can compute $\lambda$ as a function of DF:<br>
$\lambda (\mathrm{DF}) = \{\lambda | \mathrm{trace}(\bm{S}_\lambda) = \mathrm{DF}\}$

---

# Nonparametric Logistic Regression

* The penalized log likelihood in this case is <br>
$
\ell(f | \lambda) = \sum\limits_{i=1}^N \bigg[ y_i f(x_i) - \log (1 + e^{f(x_i)}) \bigg] - \frac{1}{2} \lambda \int \{ f^{\prime\prime}(t) \}^2 dt
$
  * where $f(x) = \log p(x) = \log \mathbb{P}[Y = 1 | X = x] = \sum\limits_{j=1}^N N_j(x) \theta_j = \bm{N}\theta$ <br> and $N_j$ are natural cubic splines
* We compute
  * $\ell^{\prime} (\theta) = \bm{N}_{\textcolor{grey}{N \times N}}^T (\bm{y}_{\textcolor{grey}{N \times N}} - \bm{p}_{\textcolor{grey}{N \times 1}}) - \lambda \bm{\Omega}\theta$, where $\bm{p} := [p(x_i)]$
  * $\ell^{\prime\prime} (\theta) = -\bm{N}^T \bm{WN} - \lambda \bm{\Omega}$, where $\bm{W} = \mathrm{diag} [p(x_i) (1 - p(x_i))]_{i=1:N}$
    * where $\{ \bm{\Omega}_N\}_{jk} := \int N_j^{\prime\prime}(t) N_k^{\prime\prime}(t) dt$

* To find roots of  we use iterative Newton Raphson algorithm (see [ESL, p.162](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=181)) <br>
$\theta_{\textcolor{grey}{N \times 1}}^{\mathrm{new}} := (\bm{N}^T \bm{WN})^{-1} \bm{N}^T \bm{W} (\bm{N} \theta^{\mathrm{old}} + \bm{W}^{-1} (\bm{y} - \bm{p}))$

---

# Multidimensional Splines

* Let’s extend the 1D feature space to 2D, i.e. $X_{\textcolor{grey}{2 \times 1}}^T = [X_1, X_2]$ is an observation
<div class="grid grid-cols-[1fr,1fr]">
<div>

* Define **tensor product basis** as <br> $g_{jk} (X) = h_{1j}(X_1) h_{2j}(X_2)$
  * where $h_1(X_1) \in \R^{M_1}, h_2(X_2) \in \R^{M_2}$ <br> are the **basis** functions
* The estimator is
$g(X) := \sum\limits_{j=1}^{M_1} \sum\limits_{k=1}^{M_2} \theta_{jk} g_{jk} (X)$
  * And $\theta$ can be estimated by least squares
* Similar extension to higher dimensions add 
exponentially many interaction terms

* See B-splines in [Appendix of the Chapter 5, p. 186](https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=205)

</div>
<div>
  <figure>
    <img src="/ESL_figure_5.10.png" style="width: 360px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=182">ESL Fig. 5.10</a>
    </figcaption>
  </figure>
</div>
</div>