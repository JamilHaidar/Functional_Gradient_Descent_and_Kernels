# Functional Gradient Descent
I explore the theory behind calculus of variations, functionals, divergence theorem, functional differentiation, and functional gradient descent. I implement functional gradient descent using both analytical and numerical methods.

---

# What is a Functional?
Multivariate calculus concerns itself with infitesimal changes of numerical functions – that is, functions that accept a vector of real-numbers and output a real number:
```math
f : \mathbb{R}^n \rightarrow \mathbb{R}
```
We can take this concept one level further, and inspect functions of functions, called **functionals**. Given a set of functions, $\mathcal{F}$, a functional is a mapping between $\mathcal{F}$ and the real-numbers:
```math
F : \mathcal{F} \rightarrow \mathbb{R}
```
In other words, a functional is a function defined over functions, that maps functions to the field of real (or complex) numbers. 

Some examples of functionals include:
  - The evaluation functional: $E_x(f) = f(x)$
  - The sum functional: 
```math
  S_{{x_1, \ldots, x_n}}(f) = \sum_{i=1}^n f(x_i) dx$
```
  - The integration functional: $I_{[a, b]}(f) = \int_a^b f(x) dx$

It follows that the composition of a function $g: \mathbb{R} \to \mathbb{R}$ with a functional is also a functional. 

This is where it gets interesting. Assuming we can create a loss function functional, we can create a measure to how two functions are similar. For example,
```math
L(f) = \sum_{i=1}^n(y_i - f(x_i))^2 + \lambda\lVert f \rVert ^2
```
could be considered a loss function, where the first term (the $L^2$ term), measures how close **$f(x)$** is to **$y$**, while the regularization term limits the complexity of the learned function **$f$**. 

The previous loss function maps a hypothesized function **$f$** to data points $y_i$. This can be very useful in machine learning and statistical inference. Furthermore, utilizing gradient descent where we update **$f$** in the form: **$f \rightarrow f - \eta \nabla L(f)$**, has been studied and optimized. Utilizing a Reproducing Kernel Hilbert Space (RKHS) where we fix the RBF Kernel **$K$** for example:
```math
K(x_i, x_j) = \exp\left(-\frac{\lVert x_i - x_j \rVert^2}{2\sigma^2}\right)
```
provides a space of functions $f \in H_K$ ( $H_K$ is the associated RKHS with **$K$**). Furthemore, $H_K$ is closed under the derivative, which allows the utilization of the Kernel trick to compute the functional derivative for all $f \in H_K$ efficiently. This has been studied heavily and applied to a more general sets of functions, where moving along the gradient is not always possible.

This is good for when data points $y_i$ are available to "train" on, provided the loss function $L(f)$. However, when all we are given is the loss function, we need to define a "gradient" on the functional, to allow us to do "gradient descent", minimizing the loss function.

---

# Functional Derivative
From wikipedia:

> In the calculus of variations, a field of mathematical analysis, the functional derivative (or variational derivative) relates a change in a functional (a functional in this sense is a function that acts on functions) to a change in a function on which the functional depends.
> 
> In the calculus of variations, functionals are usually expressed in terms of an integral of functions, their arguments, and their derivatives. In an integral **$L$** of a functional, if a function f is varied by adding to it another function δf that is arbitrarily small, and the resulting integrand is expanded in powers of $\delta f$, the coefficient of $\delta f$ in the first order term is called the functional derivative.

Given a function $f \in F$, the functional derivative of $F$ at $f$, denoted $\frac{\delta F}{\delta f}$, is defined to be the function for which:
```math
\begin{align*}
\int \frac{\partial F}{\partial f}(x) \eta(x) \ dx &= \lim_{h \rightarrow 0}\frac{F(f + h \eta) - F(f)}{h} \\ &= \frac{d F(f + h\eta)}{dh}\bigg\rvert_{h=0}
\end{align*}
```

---

# Analytical Solution
Consider the functional:
```math
F[\rho] = \int f(r,\rho(r),\nabla \rho(r)) dr
```
The functional derivative of $F[\rho]$, denoted $\delta F/\delta \rho$ is defined as:
```math
{\displaystyle {\begin{aligned}\int {\frac {\delta F}{\delta \rho ({\boldsymbol {r}})}}\,\phi ({\boldsymbol {r}})\,d{\boldsymbol {r}}&=\left[{\frac {d}{d\varepsilon }}\int f({\boldsymbol {r}},\rho +\varepsilon \phi ,\nabla \rho +\varepsilon \nabla \phi )\,d{\boldsymbol {r}}\right]_{\varepsilon =0}\\&=\int \left({\frac {\partial f}{\partial \rho }}\,\phi +{\frac {\partial f}{\partial \nabla \rho }}\cdot \nabla \phi \right)d{\boldsymbol {r}}\\&=\int \left[{\frac {\partial f}{\partial \rho }}\,\phi +\nabla \cdot \left({\frac {\partial f}{\partial \nabla \rho }}\,\phi \right)-\left(\nabla \cdot {\frac {\partial f}{\partial \nabla \rho }}\right)\phi \right]d{\boldsymbol {r}}\\&=\int \left[{\frac {\partial f}{\partial \rho }}\,\phi -\left(\nabla \cdot {\frac {\partial f}{\partial \nabla \rho }}\right)\phi \right]d{\boldsymbol {r}}\\&=\int \left({\frac {\partial f}{\partial \rho }}-\nabla \cdot {\frac {\partial f}{\partial \nabla \rho }}\right)\phi ({\boldsymbol {r}})\ d{\boldsymbol {r}}\,.\end{aligned}}}
```
The derivation can be found on wikipedia, where each line uses very interesting math that had me go through deep rabbit holes to actually understand and grasp. Concepts used are the [total derivative](https://en.wikipedia.org/wiki/Functional_derivative), [product rule for divergence](https://en.wikipedia.org/wiki/Divergence#Properties), [divergence theorem](https://en.wikipedia.org/wiki/Divergence_theorem), and the [fundamental lemma of calculus of variations](https://en.wikipedia.org/wiki/Fundamental_lemma_of_calculus_of_variations).

Therefore, the functional derivative is
```math
\frac{\delta F}{\delta \rho(r)} = \frac{\partial f}{\partial \rho} - \nabla \cdot \frac{\partial f}{\partial \nabla \rho}
```

## Analytical Solution Examples
I will now attempt to apply this derivation and solve multiple examples.

### Information Entropy
The entropy of a discrete random variable is a functional of the probability mass function.
```math
H(p_X) := \sum_{x \in \mathcal{X}} - p_X(x) \log p_X(x)
```
We can utilize the equation that defines the functional derivative:
```math
\begin{align*} \sum_{x \in \mathcal{X}} \frac{\partial H}{\partial p_X}(x) \eta(x) &= \frac{d H(p_X + h\eta)}{dh}\bigg\rvert_{h=0} \\ &= \frac{d}{dh} \sum_{x \in \mathcal{X}} -(p_X(x) + h\eta(x))\log(p_X(x) + h\eta(x))\bigg\rvert_{h=0} \\ &= \sum_{x \in \mathcal{X}} - \eta(x)\log(p_X(x) + h\eta(x)) + \eta(x)\bigg\rvert_{h=0} \\ &= \sum_{x \ in \mathcal{X}} (-1 - \log p_X(x))\eta(x)\end{align*}
```
Thus, $\frac{\delta H}{\delta p(x)} = -1-\log p(x)$

### Composite $L^2$ cost function for exponential

I created the functional:
```math
F[r] = \int{f(r,r(t),\nabla r(t))} dt
```
where $f = (r(t) + \frac{\partial r(t)}{\partial t})^2$, $r(0) = 1$.
This functional acts like a cost function, where we want to minimize $f$, which should converge to the result $r(t) = e^{-t}$.

#### Attempt 1
First, I will use the definition of functional derivative.

```math
{\displaystyle {\begin{aligned}\int {\frac {\delta F}{\delta r (t)}}\,\phi (t)\,dt&=\left[{\frac {d}{d\varepsilon }}\int \left(r +\varepsilon \phi + \nabla r +\varepsilon \nabla \phi \right)^2 \,dt\right]_{\varepsilon =0}\\&=\left[\int 2\cdot \left( \phi + \nabla \phi \right) \left(r +\varepsilon \phi + \nabla r +\varepsilon \nabla \phi \right) dt\right]_{\varepsilon =0}\\&=\int 2\cdot \left( \phi + \nabla \phi \right) \left(r + \nabla r\right) dt\\&=2\int \phi \cdot \left(r + \nabla r\right) + \nabla \phi \cdot \left(r + \nabla r\right) dt
\end{aligned}}}
```

If we didn't know how to expand $\nabla \phi$, we would be stuck at this point. This is where the generic rule for the product rule for divergence comes in to play. Therefore, I will stop persuing this, and just apply the generic formula directly.

#### Attempt 2
We will use the generic formula:
```math
\begin{aligned}
\frac{\delta F[r]}{\delta r(t)} &= \frac{\partial f}{\partial r} - \nabla \cdot \frac{\partial f}{\partial \nabla r}\\&=2\left(r(t) + \frac{\partial r(t)}{\partial t}\right) - \frac{\partial}{\partial t}\left(\frac{\partial f}{\partial r}\right)\\&=2\left(r(t) + \frac{\partial r(t)}{\partial t}\right) - \frac{\partial}{\partial t}\left(2\left(r(t) + \frac{\partial r(t)}{\partial t}\right)\right)\\&= 2r(t) + 2\frac{\partial r(t)}{dt} -2 \frac{\partial r(t)}{dt} -2 \frac{\partial^2 r(t)}{\partial t^2}\\&= 2r(t) -2 \frac{\partial^2 r(t)}{\partial t^2}
\end{aligned}
```
Finally, we found $\frac{\delta F[r]}{\delta r(t)} = 2r(t) -2 \frac{\partial^2 r(t)}{\partial t^2}$, which is the correct solution! We can verify this using MATLAB's ``functionalDerivative`` method:

> ![image](https://user-images.githubusercontent.com/60647115/216766594-6f408203-91e7-4b75-8907-acc61616d93b.png)

### More degrees!
Let $f$ be $f = \left(r(t) + \frac{\partial r(t)}{\partial t} + \frac{\partial^2 r(t)}{\partial t^2}\right)^2$. I will apply the generic functional derivative formula:
```math
\begin{aligned}
\frac{\delta F[r]}{\delta r(t)} &= \frac{\partial f}{\partial r(t)} - \nabla \cdot \frac{\partial f}{\partial \nabla r(t)} + \nabla^2 \cdot \frac{\partial f}{\partial \nabla^2 r(t)}\\&=2\left(r(t) + \frac{\partial r(t)}{\partial t} + \frac{\partial^2 r(t)}{\partial t^2}\right)\\&-2\left(\frac{\partial r(t)}{\partial t} + \frac{\partial^2 r(t)}{\partial t^2}+\frac{\partial^3 r(t)}{\partial t^3}\right)\\&+2\left(\frac{\partial^2 r(t)}{\partial t^2}+\frac{\partial^3 r(t)}{\partial t^3}+\frac{\partial^4 r(t)}{\partial t^4}\right)\\&=2r(t) + \frac{\partial^2 r(t)}{\partial t^2} + 2\frac{\partial^4 r(t)}{\partial t^4}
\end{aligned}
```
This can be verified using MATLAB's ``functionalDerivative`` method:

> ![image](https://user-images.githubusercontent.com/60647115/216766663-54ac40d6-1931-4b22-8519-53dfe8ef50f3.png)

# Numerical Solution
Will be updated soon.
