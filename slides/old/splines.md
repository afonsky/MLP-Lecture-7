# Piecewise Polynomials and Splines

<div class="grid grid-cols-[3fr,2fr]">
<div>
<br>

* Consider $f(X) := \sum_m \beta_m h_m (X)$ and <br>
$R_m := \{i | x_i \in [\xi_{m-1}, \xi_m) \}$ indices in region <br>
$m = 1,2,3$ with $\xi_0 := -\infty$, $\xi_3 = \infty$
* Type of basis functions:
  1. $h_m(x) := I_{(x \in R_m)}$ are indicators b/w knots
    * Regression coefficients are averages on each region: <br>
    $\hat{\beta}_m := \bar{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i$
  2. $h_m$ are linear regression fit b/w knots
  3. $f(X)$ - piecewise linear, continuous
    * We force $f(\xi_m^{-}) = f(\xi_m^{+})$
      * i.e. left and right limits at $\xi_m$ should be equal
    * Alt., we can use a basis $1, X, (X - \xi_m)_{+}, m = 1, 2$
* Effective DF = # estimates - # constraints

</div>
<div>
  <figure>
    <img src="/ESL_figure_5.1.png" style="width: 330px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=161">ESL Fig. 5.1</a>
    </figcaption>
  </figure>
</div>
</div>

<!-- DF=3
DF=6
DF=6-2=4
DF=0 -->

---

# Piecewise CubicPolynomials and Splines

<div class="grid grid-cols-[3fr,2fr]">
<div>
<br>

1. $f(X)$ is piecewice cubic
2. $f(X)$ is also continuous
3. $f(X)$ is also continuous in the $1^{\mathrm{st}}$ derivative
4. $f(X)$ is also continuous in the $2^{\mathrm{nd}}$ derivative
  * This is a **cubic spline**
  * Connections at knots are not visible to the eye
  * One **representation** is with the basis: <br>
$1, X, X^2, X^3, (X - \xi_1)_{+}^3, (X - \xi_2)_{+}^3$
  * There are many other bases to represent $f$

</div>
<div>
  <figure>
    <img src="/ESL_figure_5.2.png" style="width: 330px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=162">ESL Fig. 5.2</a>
    </figcaption>
  </figure>
</div>
</div>

<!-- Degrees of freedom
1. DF=4+4+4=12
2. DF=12-2=10
3. DF=10-2=8
4. DF=8-2=6 or 4+K=6 -->

---

# Natural Cubic Spline (NCS)

* Polynomials and cubic splines are **unstable** at boundaries, where observations are sparce
* NCS forces **linearity** at the boundaries
  * Stabilizes extrapolation and frees **4 DF** (i.e. less uncertainty)

<br>
<br>

<div class="grid grid-cols-[1fr,1fr]">
<div>

* Basis for $K$ knots has $K$ basis functions:
* $N_1 (X) := 1$
* $N_2 (X) := X$
* $N_{k+2} (X) := d_k (X) - d_{K-1} (X)$
  * where $d_k (X) := \frac{(X - \xi_k)_{+}^3 - (X - \xi_K)_{+}^3}{\xi_K - \xi_k}$

* $N_k^{\prime\prime} (X) =  N_k^{\prime\prime\prime} (X) = 0$ for $X \geq \xi_k$
</div>

<div>
  <figure>
    <img src="/ISLRv2_figure_7.4.png" style="width: 400px">
    <figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
      <a href="https://hastie.su.domains/ISLR2/ISLRv2_website.pdf#page=306">ISLR Fig. 7.4</a>
    </figcaption>
  </figure>
</div>
</div>