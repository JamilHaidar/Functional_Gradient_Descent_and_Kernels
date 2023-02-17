# Functional Gradient Descent and Kernels
I explore the theory behind calculus of variations, functionals, divergence theorem, functional differentiation, calculus of variations, and functional gradient descent. I implement functional gradient descent using both analytical and numerical methods. Furthermore, I explore kernels, reproducing kernel hilbert spaces, support vector machines, and so much more.

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

# Kernel Trick and RKHS

## Theory

A kernel $K: X \times X \to \mathbb{R}$ is a function that generalizes dot products where:
```math
K(x_i,x_j) = {\langle \phi(x_i),\phi(x_j) \rangle}_{H_K}
```
where $H_K$ is the Hilbert space corresponding with **$K$**.

Basically, it is a dot product between "features" of x ( $\phi(x)$ ). So why do we care about representing x in a different space? A reproducing kernel Hilbert space (RKHS), is a space of functions where every function **$f$** is some linear combination of the kernel **$K$** evaluated at some "centers" $x_{C_f}$:

```math
f = \sum_{i=1}^{n} \alpha_{f_i} K(\cdot,x_{C_{f_{i}}})
```

This means, if there is a function **$f \in H_K$**, it can be determined completely by **$\alpha_f$** and **$x_{C_f}$**!

Assume we want to estimate do functional gradient descent where our loss function is $L(f)$ as defined above. We can represent our estimate $f(x)$ in multiple ways (in the discrete calculus of variations sense) as a combination of functions, which are considered the eigenfunctions of the space our kernel represents.

```math
f(x) = \sum_{l=1}^{\infty} \hat{f}_{l} e_l(x)
```

In the case of the Fourier Series, we have
```math
f(x) = \sum_{l=-\infty}^{\infty} f_l e^{i2{\pi}lx}
```
We can see clearly that we can represent a function $f(x)$ as a linear combination of the eigenfunctions $\phi(x) = e^{i2{\pi}lx}$ with some coefficients. However, assuming we have the kernel, but not the eigenfunctions of our space, this could get tricky. We can use Mercer's theorem which states that for the Gram matrix **$\mathcal{K}$** where
```math
(\mathcal{K})_{ij} = k(x_i,x_j)
```
we can compute an eigenvector decomposition as
```math
\mathcal{K} = \mathbf{U}^T \Lambda \mathbf{U}
```
where  

```math
(\mathcal{K})_{ij} = \phi(x_i)^T \phi(x_j)
```
```math
\phi(x_i) = \Lambda^{1/2} \mathbf{U}_{:,i}
```
Instead of computing $\phi(x)$ which may be difficult, we can use the kernel trick instead. The kernel trick is defined as:

> Kernel methods owe their name to the use of kernel functions, which enable them to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space. This operation is often computationally cheaper than the explicit computation of the coordinates.This approach is called the **"kernel trick"**

Here's the idea in an example. For the RBF kernel:
```math
k(x_i, x_j) = \exp\left(-\frac{\lVert x_i - x_j \rVert^2}{2\sigma^2}\right)
```
For this kernel, the dimension of the feature space defined by $\phi(\cdot)$ is $D=\infty$. The kernel trick will allow us to avoid explicitly computing $\phi(\cdot)$. We can easily compute the $n \times n$ Gram matrix using the kernel, even though we have implicitly projected our objects to an infinite dimensional feature space.

The whole idea boils down to this. We want to represent the estimated function f(x) as linear combination of basis functions. Instead of doing this computation, we can instead express it as a linear combination of a kernel evaluated at specific "centers".

## Application
For
```math
L(f) = \sum_{i=1}^n(y_i - f(x_i))^2 + \lambda\lVert f \rVert ^2
```
we have the functional derivative $DL(f)$ given by
```math
DL(f) = \sum_{i=1}^n -2(y_i-f(x_i)) \cdot K(x_i,\cdot) + 2\lambda f
```
where $f_k$ at iteration $k$ can be calculated by:
```math
f_k = \sum_{i=1}^{n} \alpha_{f_ki} K(\cdot,x_i)
```
Finally, we can choose to represent $f_k$ implicitly by $\alpha_k$, making the final update as:
```math
\alpha_{f_{k+1}} = 2 \eta (y-f_k(x)) + (1-2\lambda \eta)\alpha_{f_k}
```
## Algorithm
Finally, we define our algorithm and implement it.
1. Get dataset $x,y:=f(x)$
2. Create the Gram kernel $K(i,j) = k(x_i,x_j)$, where $k$ is the selected kernel (e.g. RBF kernel)
3. Initialize $\alpha$ vector (random values close to 0 perhaps)
4. Perform descent:
   - Calculate function: $f_k(x) = K \cdot \alpha_k$
   - Update function: $\alpha_{k+1} = 2 \eta (y-f_k(x)) + (1-2\eta \lambda) \alpha_k$

### RBF Kernel

Using the RBF Kernel defined earlier, we get the following results:


https://user-images.githubusercontent.com/60647115/219679722-ad381867-f537-4130-b685-78c7064fe4f3.mp4


After 200 iterations using the RBF kernel with $\eta = 0.01, \lambda = 0.01, \gamma := \frac{1}{2\sigma^2} = 4$. However, let's zoom out a little and see what's going on:


https://user-images.githubusercontent.com/60647115/219680575-cfdded7c-47a1-41fd-8db4-8c06d7441a3f.mp4


It looks like there is a region where the function becomes flat, and the rbf function with $\gamma = 4$ can no longer fit better.

### Fourier series Kernel

I will now attempt to implement a fourier series kernel. The kernel will be defined on $\mathbb{R}^D$ as:
```math
\begin{align*}
k(x_i,x_j) &= \int_{\mathbb{R}^D} e^{-i(x_i-x_j)^T\omega}p(\omega) d\omega \\ &\approx \frac{1}{C} \sum_{c=1}^{C} e^{-i(x_i-x_j)^T\omega_c}\\&=\langle \hat{\phi(x_i)},\hat{\phi(x_j)}\rangle \\
\therefore k(x_i,x_j) &\approx \frac{1}{C} \sum_{c=1}^{C} \left[ cos(\omega^T_c (x_i-x_j)) \right]
\end{align*}
```

where $C$ is the number of spectral samples from the density $p$. This is is in fact a *Monte Carlo* (MC) approximation to the integral. Through the standard trigonometric identity $cos(u − v) = cos(u) cos(v) + sin(u) sin(v)$ we arrive at the $2C$-dimensional mapping $\phi : \mathcal{X} \rightarrow \mathbb{R}^{2C}$the final representation:

```math
\begin{align*}
\hat{\phi(x)} = \frac{1}{\sqrt{C}} \Bigl[ & cos(x^T\omega_1), \dots, cos(x^T\omega_C),\\ & sin(x^T\omega_1), \dots, sin(x^T\omega_C) \Bigr]
\end{align*}
```

I specified the kernel to be
```math
k(x_i,x_j) = \sum_{\omega_c=0}^{99} cos(\omega_c (x_i-x_j))
```

https://user-images.githubusercontent.com/60647115/219693430-340d9d31-36a5-4a6e-851d-98a4c5e4a566.mp4


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

# Functional Gradient Descent: Numerical Solution

Finally, after understanding how the functional derivative works, we will attempt functional gradient descent. Unlike the Kernel approach, we do not have the functional $L(f)$. Instead, we assume there's simply a black box that takes as input the function $f(x)$ and returns a value. I will be relying on this definition:

```math
\begin{align*}
\int \frac{\partial F}{\partial f}(x) \eta(x) \ dx &= \lim_{h \rightarrow 0}\frac{F(f + h \eta) - F(f)}{h} \\ &= \frac{d F(f + h\eta)}{dh}\bigg\rvert_{h=0}
\end{align*}
```

This means that to get the gradient along a direction, we can perturb the function $f$ with some other function $\eta$, and simply get the gradient by making this perturbation small enough to approach the limit.

The approach looks like this 
1. Generate range of independent variable x
3. Generate a perturbation function $\phi(x)$
4. Initialize learning rate $\eta$ and epsilon $\epsilon$
2. Generate initial estimate $\rho(x)$
3. Set any initial conditions $\rho(0) = c$
4. Start iterating until max iterations reached or the Functional reaches a threshold cost
   - Up until a specific degree $d$, generate $\frac{\partial^d \rho}{\partial x^d}, \frac{\partial^d \phi}{\partial x^d}$
   - Calculate the gradient $Df = \frac{f(x,\rho + \epsilon \phi,\partial \rho + \epsilon \partial \phi,\dots) - f(x,\rho,\partial \rho,\dots)}{\epsilon}$
   - Update the estimate $\rho_{k+1}(x) = \rho_k(x) - \eta Df$

Important things to note:
  - The learning rate (step size) must be tuned correctly or else we risk the solution diverging. 
  - When an initial condition exists, we don't update that specific point. This means that if $\rho{x_0} = cst$, then the neighboring points $\rho{x_k}$ will be affected by the first degree derivative after $k$ iterations. This is seen only if the functional is affected by degrees of $\rho(x)$. The example $L(\rho) = \int (\rho(x)+\frac{\partial \rho}{\partial x}(x))^2 dx$ solves to $\rho(x) = c_0e^{-x}$. This example is illustrated below.
  - When calculating the gradient numerically, I used central difference for points not at the edges. For points at the edges I used forward difference and backward difference.

## Sin function

I'll start with a simple functional: $L(\rho) = \int f(x,\rho(x)) dx$ where $f(x,\rho(x)) = \frac{1}{2}(\rho(x)-sin(x))^2 dx$. This is a simple function that is minimized when $\rho(x) = sin(x)$. I initialize $\rho(x) = 1$ (arbitrary). This is how it went:


https://user-images.githubusercontent.com/60647115/219733330-b7883b96-339c-406f-b53c-939cda98344e.mp4

## Decaying Exponential function

Now, I'll get to a differential equation: $L(\rho) = \int f(x,\rho(x), \partial \rho(x)) dx$ where $f(x,\rho(x),\partial \rho(x)) = \frac{1}{2}(\rho(x)-\frac{\partial \rho}{\partial x}(x))^2 dx$ with $\rho(0) = 1$. The solution should converge to $\rho(x) = e^{-x}$. This required more iterations as I needed to decrease the learning rate and wanted to make it more fine (increased number of points):


https://user-images.githubusercontent.com/60647115/219735230-c973d981-8e07-4847-a606-7ae34cee695f.mp4


# Conclusion

This turned out to be a huge project as I had not planned to dive as deep as I did. However, I learned so much about calculus of variations, divergence theorem, functionals, functional differentiation, SVMs, Kernels, Hilbert spaces, RKHS, Mercer's theorem, Gram kernels, and so much more interesting math that I had to go over grad lectures from multiple universities, wikipedia, books, and published papers to be enough to scratch the surface of this field and make something that applies this theory that actually works.

# References
- [Wikipedia Functional Derivative](https://en.wikipedia.org/wiki/Functional_derivative)
- [Lecture 1](http://www.cs.cmu.edu/~16831-f12/notes/F12/16831_lecture21_danielsm.pdf),[Lecture 2](http://www.cs.cmu.edu/~16831-f14/notes/F10/16831_lecture24_varunnr/16831_lectureNov11.vramakri.pdf)
- [Gradient Boosting blog](https://nicolas-hug.com/blog/gradient_boosting_descent)
- Tompkins, A., & Ramos, F. (2018, April). Fourier feature approximations for periodic kernels in time-series modelling. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1). ([link](https://ojs.aaai.org/index.php/AAAI/article/view/11696/11555))
- Great reference for Functional Gradient Descent with Kernels: [Simple Complexities Blog](https://simple-complexities.github.io/optimization/functional/gradient/descent/2020/03/04/functional-gradient-descent.html)
- Summary reference for [Reproducing Kernel Hilbert Spaces](https://people.eecs.berkeley.edu/~bartlett/courses/281b-sp08/7.pdf)
- Great simple to understand reference to RKHS and kernels [lecture](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf)
- Kernels and Kernel Methods [lecture](https://www.cs.princeton.edu/~bee/courses/scribe/lec_10_09_2013.pdf)
- [Wikipedia Kernel Method](https://en.wikipedia.org/wiki/Kernel_method)
- [Wikipedia Divergence](https://en.wikipedia.org/wiki/Divergence) and [Wolfram Divergence](https://mathworld.wolfram.com/Divergence.html)
- [Wikipedia Divergence Theorem](https://en.wikipedia.org/wiki/Divergence_theorem) and [Wolfram Divergence Theorem](https://mathworld.wolfram.com/DivergenceTheorem.html)
- [SVM and Kernel SVM](https://towardsdatascience.com/svm-and-kernel-svm-fed02bef1200)
