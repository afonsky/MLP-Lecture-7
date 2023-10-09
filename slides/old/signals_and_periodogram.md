# Periodogram and Spectral Density

* **Periodogram** is an estimate of (true) spectral density, which is power (frequency) function
  * Basically, for a (time) series with $N$ points we fit to $N/2$ harmonics (or sine functions)
  * If model is appropriate, then smoothing or thresholding periodogram is effective
    * in noise reduction and signal extraction
* **Fourier transform** is often used to convert between time and frequency domains
  * **Fast Fourier Transform** ([FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform)) – among the biggest breakthroughs in science!

<figure>
  <img src="/spectral_density.png" style="width: 740px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Images sources:
    <a href="https://en.wikipedia.org/wiki/Spectral_density">https://en.wikipedia.org/wiki/Spectral_density</a>, <a href="https://en.wikipedia.org/wiki/Harmonic">https://en.wikipedia.org/wiki/Harmonic</a>
  </figcaption>
</figure>
---

# Ex. Spectra of Financial Series

* To find a signal in stock price time series, we 
  * Compute returns, i.e. make it weakly stationary, i.e. derive a series with a constant mean and variance
    * This is often done by differencing, $y_t := x_t - x_{t-1}$
    * or scaled differencing, $y_t := (x_t - x_{t-1}) / x_t$
    * or log differencing, $y_t := \log x_t - \log x_{t-1}$

<div class="grid grid-cols-[5fr,4fr]">
<div>
<br>

* Decompose $y_t$ into sinusoidals, i.e. compute its **spectra**
* This can be done via **FFT** or via **regression** on sinusoidals
* Squared returns, $y_t^2$, estimate volatility of $y_t$
* These can also be decomposed into spectra

* Financial time series are more difficult to analyze <br> because of "**high noise to signal ratio**"
</div>
<div>

  <figure>
    <img src="/financial_series.png" style="width: 450px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://link.springer.com/article/10.1007/s11634-019-00365-8">https://link.springer.com/article/10.1007/s11634-019-00365-8</a>
    </figcaption>
  </figure>
</div>
</div>

---

# Ex. Classify Phonemes: "aa" and "ao"

<div class="grid grid-cols-[5fr,3fr]">
<div>

* We convert from time to frequency domain
  * Each point at a frequency $f$ is a sine function
  * Then smooth coefficients with a natural cubic spline <br> with $256$ knots, $12$ basis functions
    * Test error: ``0.255`` (raw), ``0.158`` (regularized)
  * Logistic regression classifier with $N = 1000$: <br>
  $G(X_{\textcolor{grey}{1000 \times 256}}) := \log \frac{\mathbb{P}[aa | X]}{\mathbb{P}[ao | X]} \approx X\beta$
    * A row of $X$ is a log-periodogram in $\R^{256}$
* Express $\beta$ as averaged splines, $h_m$: $\beta(f) = \sum\limits_{m=1}^M h_m (f) \textcolor{red}{\theta_m}$
  * Or, $\beta = H_{\textcolor{grey}{p \times M}} \textcolor{red}{\theta}, p = 256, M = 12$ 
  * So, we can fit $\textcolor{red}{\theta}$ via log. reg.: $G(X) \approx X H_{\textcolor{grey}{p \times M}} \textcolor{red}{\theta} = X^\ast \textcolor{red}{\theta}$
* Alternatively, we can fit the classifier to <br> transformed inputs $X^\ast$ to estimate $\textcolor{red}{\theta_{\textcolor{grey}{12 \times 1}}}$ directly
</div>
<div>

  <figure>
    <img src="/ESL_figure_5.5.png" style="width: 490px !important;">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=168">ESL Fig. 5.5</a>
    </figcaption>
  </figure>
</div>
</div>