---
background: /mountain.jpg
layout: cover
hideInToc: true
---

# Vapnik-Chervonenkis Dimension

---

# Vapnik-Chervonenkis (VC) Dimension

* VC theory gives a general measure of model complexity and bounds on optimism
* **VC dimension** of binary classifiers $\mathcal{F} := \{f_{\alpha}\}$ is largest number of points (in some configuration) that can be shattered by members of $\mathcal{F}$
	* **Shattering** points means perfectly separating their any label assignment ([see more](https://en.wikipedia.org/wiki/Shattered_set))
	* $\mathcal{F}$ needs to shutter all labeling assignments, not all the point arrangements
		* VC dim is $\geq p$, if **there exists some** combination of $p$ points that can be shattered for any labeling arrangement
		* VC dim is $< p$, if for any combination of $p$ points, **there exists some** inseparable label assignment

* VC dim gives an intuition of what to expect from classifier families
	* We should not expect a line to perfectly classify 4 points all the time
* Definition can be extended to more complex outputs (3+ levels, quantitative, etc.)

---

# VC Example
<div class="grid grid-cols-[5fr,4fr]">
<div>

* Which function is more complex (higher VC dim)?

* $\ell_{\vec{\alpha}} := I(\alpha_0 + \alpha_1^\prime x > 0)$ with $p + 1$ params
	* VC dim = 3 in $\mathbb{R}^p$
	* In general:
		* $p$-dimensional hyperplane has VC dim of $p + 1$
			* E.g. perceptron. See [proof](http://www.cs.columbia.edu/~jebara/4771/hw2_2015_solutions.pdf)

* $s_{\alpha} := I(\sin \alpha x > 0)$ with 1 param
	* VC dim = $\infty$ in $\mathbb{R}$
</div>
<div>
<br>
  <figure>
    <img src="/ESL_figure_7.5.png" style="width: 350px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
      <a href="https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf#page=256">ESL Fig. 7.5</a>
    </figcaption>
  </figure>
<br>
<br>
  <figure>
    <img src="/VC_classifier.png" style="width: 390px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:
	  <a href="https://datascience.stackexchange.com/questions/32557/what-is-the-exact-definition-of-vc-dimension">https://datascience.stackexchange.com/questions/32557/what-is-the-exact-definition-of-vc-dimension</a>
    </figcaption>
  </figure>
</div>
</div>

---

# VC Example: Rectangular Classifiers in $\mathbb{R}^2$
<div class="grid grid-cols-[5fr,4fr]">
<div>

* A rectangle with edges parallel to axes
	* Some arrangement of $4$ points, which can be shattered, i.e. separable regardless of label assignment
	* Any arrangement of $5$ points cannot be shattered
		* Rectangle needs to cover $4$ points only, but every arrangement of points will have a **bounding** rectangle with the fifth point in it or on edge. This is the case, when the $5^{\mathrm{th}}$ point cannot have a distinct label
			* We ignore trivial cases where 3,4,5 points are on the same horizontal/vertical line
* VC dim is 4
</div>
<div>
<br>
  <figure>
    <img src="/VC_rectangle_shatter.png" style="width: 350px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image sources:
      <a href="https://www.cs.cornell.edu/courses/cs683/2008sp/lecture%20notes/683notes_0428.pdf">[1]</a>,
      <a href="http://me-ramesh.blogspot.com/p/machine-learning.html">[2]</a>
    </figcaption>
  </figure>
</div>
</div>

---

# VC Example: Rotatable Rectangle in $\mathbb{R}^2$
<div class="grid grid-cols-[1fr,1fr]">
<div>

* Shattering of 3 and 4 points in shown
* Left as an exercise:
	* Shattering of 0,1,2, 6,7
	* Non-shattering of 8 points
* VC dim is 7
  <figure>
    <img src="/VC_rotatable_rectangle_shatter_2.png" style="width: 300px !important">
  </figure>
</div>

<div>
  <figure>
    <img src="/VC_rotatable_rectangle_shatter_1.png" style="width: 300px !important">
      <figcaption style="color:#b3b3b3ff; font-size: 11px;">Images sources:<br>
      <a href="https://www.cs.cornell.edu/courses/cs683/2008sp/lecture%20notes/683notes_0428.pdf">https://www.cs.cornell.edu/courses/cs683/2008sp/lecture%20notes/683notes_0428.pdf</a>,<br><br>
      <a href="http://me-ramesh.blogspot.com/p/machine-learning.html">http://me-ramesh.blogspot.com/p/machine-learning.html</a>
    </figcaption>
  </figure>
</div>
</div>

---

# VC Example: Square Classifiers in $\mathbb{R}^2$
<div class="grid grid-cols-[1fr,1fr]">
<div>

* A square with edges parallel to axes
* Left as an exercise
* VC dim is 3
</div>

<div>
  <figure>
    <img src="/VC_square_shatter.png" style="width: 300px !important">
      <figcaption style="color:#b3b3b3ff; font-size: 11px;">Images sources:<br>
      <a href="https://www.cs.cornell.edu/courses/cs683/2008sp/lecture%20notes/683notes_0428.pdf">https://www.cs.cornell.edu/courses/cs683/2008sp/lecture%20notes/683notes_0428.pdf</a>,<br><br>
      <a href="http://me-ramesh.blogspot.com/p/machine-learning.html">http://me-ramesh.blogspot.com/p/machine-learning.html</a>
    </figcaption>
  </figure>
</div>
</div>

---

# VC Example: Circle Classifiers in $\mathbb{R}^2$
<br>
<div class="grid grid-cols-[1fr,1fr]">
<div>

* At origin: VC dim 2
* In any location: VC dim 3
</div>

<div>
	<figure>
	<img src="VC_Circles.png" style="width: 420px !important">
    </figure>
    <figure>
    <img src="/VC_Circle_Classifiers.png" style="width: 400px !important">
      <figcaption style="color:#b3b3b3ff; font-size: 11px;">Images sources:<br>
      <a href="https://slideplayer.com/slide/10807865/">https://slideplayer.com/slide/10807865/</a>,<br><br>
      <a href="https://datascience.stackexchange.com/questions/21693">https://datascience.stackexchange.com/questions/21693</a>
    </figcaption>
  </figure>
</div>
</div>

---

# VC Dimension of Neural Network (NN)
<br>
<div class="grid grid-cols-[3fr,1fr]">
<div>

* NN can be described by **directed acyclic graph** with 
	* $V$ vertices, $E$ edges with weights 
	* single **sink** node, single **source** node
	* internal nodes can have activation function $f$ (sigmoid, sign, ...)
* $f$ is a sign function: 		VC dim $= \mathcal{O}(|E|, \cdot \log|E|)$
* $f$ is a sigmoid function: 		VC dim $\in [\mathcal{O}(|E|^2), \mathcal{O}(|E|^2 |V|^2))$
* If weights from a finite range: 	VC dim $= \mathcal{O}(|E|)$

What NN would you build to completely separate any labeling of ?
</div>

<div>
    <figure>
    <img src="/Tred-G.svg.png" style="width: 200px !important">
      <figcaption style="color:#b3b3b3ff; font-size: 11px;">Image source:<br>
      <a href="https://en.wikipedia.org/wiki/Directed_acyclic_graph">https://en.wikipedia.org/wiki/Directed_acyclic_graph</a>
    </figcaption>
  </figure>
</div>
</div>
<br>
<br>

Further reading: [https://igi-web.tugraz.at/PDF/139.pdf](https://igi-web.tugraz.at/PDF/139.pdf)