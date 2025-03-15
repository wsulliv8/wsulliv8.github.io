---
layout: post
title: "Descent Methods: From Theory to Practice"
image: /assets/images/gradient/optimization.png
excerpt: In this blog, we explore the various types of gradient and use-cases of gradient descent with specific emphasis on relation to linear algebra.
---
---
---

Optimization problems are everywhere in data science, machine learning, and scientific computing. At the heart of many of these problems lie descent methods. Descent methods are algorithms that iteratively move toward a solution by following paths of decreasing function values. The concept of gradient-based optimization was first proposed in the early 1800s by Augustin-Louis Cauchy, but it wasn't until the 1980s that it gained traction in machine learning due to its use in training neural networks. Let's explore the rich theory and practical applications of descent methods with a particular focus on the Conjugate Gradient Method and preconditioning.

## Overview of Gradient Descent

Gradient descent is the cornerstone of optimization algorithms. The fundamental idea is elegantly simple: to minimize a differentiable function, one should repeatedly step in the direction of steepest descent (the negative gradient).
- You may recall from Calculus that the gradient is a vector that gives the **direction** and **magnitude** of the greatest rate of increase of a function. For a scalar function which takes multiple inputs, the gradient is defined as:

$$
\nabla f(\chi_{1}, \chi_{2},\dots,\chi_{n}) = \left( \frac{\partial f}{\partial \chi_{1}},  \frac{\partial f}{\partial \chi_{2}},\dots,\frac{\partial f}{\partial \chi_{n}} \right)
$$

The function we use in gradient descent is called the **cost function** (also known as the loss function or objective function). Common cost functions include:
- Mean Squared Error (for regression)

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}
$$

- Cross-Entropy Loss (for binary classification)

$$
CEL = -\frac{1}{n}\sum_{i=1}^{n}\Big( y_{i}\log(\hat{y}_{i})+(1-y_{i})\log(1-\hat{y}_{i})\Big)
$$

- Log-Likelihood (for binary classification)

$$
\log L(\theta)=\sum_{i=1}^{n}\Big(y_{i}\log(\sigma(z_{i}))+(1-y_{i})\log(1-\sigma(z_{i}))\Big)
$$
- There are many more cost functions available that are tailored to solve particular problems such as regression, classification, ranking, and reinforcement.

The iterative update rule for gradient descent is:

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

Where:

- $x_k$ is our current position
- $\nabla f(x_k)$ is the gradient at that position
- $\alpha_k$ is the step size or learning rate

The figures below depict gradient descent in 2D and 3D. We take iterative steps in the direction of steepest descent until we reach the minimum of the cost curve (plane).

<div class="image-container"><img src="/assets/images/gradient/2d-gradient.png" style="width:800px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-1: 2D Gradient Descent</em></p>

<div class="image-container"><img src="/assets/images/gradient/3d-gradient.png" style="width:800px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-2: 3D Gradient Descent</em></p>

*Fig-1* and *Fig-2* each depict an initial starting guess $x_{0}$ and the iterative steps the algorithm takes to reach the minimum. 

What makes gradient descent particularly powerful is its generality since it requires only that the function be differentiable. The convergence rate depends on several key factors:
- Learning rate
	- Too large: May overshoot or diverge from the minimum
	- Too small: Converges slowly or gets stuck in plateaus
- Cost function characteristics 
	- For convex functions with Lipschitz-continuous gradients, convergence to a global minimum is guaranteed
	- For strongly convex functions, the convergence rate is linear
	- For poorly conditioned problems, convergence can be painfully slow
- Starting Point
	- Proximity to local minima will affect which minimum is found
- Gradient Properties
	- Vanishing or exploding gradients where updates become negligible or extremely large, respectively. 
	- Certain gradient direction and curvature such as a saddle point could stall the algorithm.
	- Problems with sparse features may be exploited for computational efficiency.
- Termination Criteria
	- Gradient magnitude threshold and max number of iterations
- Employment of Optimization Methods
	- Methods such as momentum and adaptive learning rates may make convergence more consistent 

Below are six varieties of cost functions that illustrate possible optimization challenges.


<div class="image-container"><img src="/assets/images/gradient/normal-gradients.png" style="width:1200px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-3: Basic Cost Surfaces</em></p>


<div class="image-container"><img src="/assets/images/gradient/optimization.png" style="width:1200px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-4: Optimization Challenges</em></p>

## Famous Examples of Gradient Descent


### 1. Neural Network Training

The back-propagation algorithm in neural networks is fundamentally gradient descent applied through the chain rule.

For a neural network with weights $W$, input $x$, target output $y$, and loss function $L$, gradient descent updates each weight as:

$$W_{ij} \leftarrow W_{ij} - \alpha \frac{\partial L}{\partial W_{ij}}$$

Deep learning frameworks like TensorFlow and PyTorch implement automatic differentiation to compute these gradients efficiently across complex architectures with millions of parameters. The success of deep learning in computer vision, natural language processing, and reinforcement learning all stem from these gradient-based optimization techniques.

### 2. Linear Regression

In linear regression, we minimize the sum of squared residuals:

$$\min_\beta \lvert \lvert X\beta-y \rvert  \rvert ^{2}_{2}$$

While the closed-form solution $\beta = (X^TX)^{-1}X^Ty$ exists, gradient descent provides a scalable alternative when $X$ is large. The closed-form solution is also known as the "normal equation". There are far more efficient direct methods such as $QR$ or SVD decomposition. The gradient of the cost function with respect to $\beta$ is:

$$\nabla_\beta J(\beta) = X^T(X\beta - y)$$

Leading to the update rule:

$$\beta_{k+1} = \beta_k - \alpha X^T(X\beta_k - y)$$

This approach scales to massive datasets through mini-batch variants, making it central to large-scale machine learning.

### 3. Support Vector Machines

SVMs find optimal separating hyperplanes between classes, maximizing the margin between them. The primal optimization problem is:

$$\min_{w,b} \frac{1}{2}|w|^2 \text{ subject to } y_i(w^Tx_i + b) \geq 1 \text{ for all } i$$

The dual formulation transforms this into a convex quadratic programming problem:

$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$

$$\text{subject to } \alpha_i \geq 0 \text{ and } \sum_{i=1}^n \alpha_i y_i = 0$$

Sequential Minimal Optimization (SMO) efficiently solves this through a series of gradient descent-like steps on carefully chosen pairs of dual variables.

### 4. Computer Vision and Graphics

From image denoising to 3D reconstruction, gradient descent helps minimize energy functionals that encode our priors about the visual world.

For instance, in total variation denoising, we minimize:

$$
\min_{u} \frac{1}{2}\lvert u-f \rvert ^{2}+\lambda \lvert \nabla u \rvert _{1}
$$

When $f$ is the noisy image, $u$ is the denoised output, and $\lvert \nabla u \rvert_{1}$ encourages piecewise smoothness while preserving edges.

In computer graphics, gradient descent optimizes parameters for physical simulations, rendering equations, and geometric modeling. It powers physics engines in video games, cloth simulation in animated films, and computational fluid dynamics in special effects.

### 5. Reinforcement Learning

In deep reinforcement learning, policy gradient methods use gradient descent to directly optimize the expected return:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[R]$$

Where $\pi_\theta$ is the policy parameterized by $\theta$ and $R$ is the cumulative reward. The REINFORCE algorithm computes an estimate of $\nabla_\theta J(\theta)$ using sampled trajectories, then performs gradient ascent.

This approach has led to groundbreaking results like AlphaGo's victory over the world champion in Go, OpenAI's Dota 2 agent beating professional teams, and Boston Dynamics' robots learning complex locomotion skills.

## Roots in Linear Algebra

The deep connection between descent methods and linear algebra becomes especially clear when we examine quadratic optimization problems:

$$f(x) = \frac{1}{2}x^TAx - b^Tx + c$$

Where $A$ is a symmetric positive-definite matrix, $b$ is a vector, and $c$ is a constant. For these problems:

- The gradient is simply $\nabla f(x) = Ax - b$
- The Hessian (matrix of second derivatives) is just $A$ itself
- The minimum occurs exactly where $Ax = b$

This gives us a fundamental insight: **gradient descent for quadratic functions is iteratively solving a linear system**. The update becomes:

$$x_{k+1} = x_k - \alpha_k(Ax_k - b)$$

The convergence properties are determined by the eigenvalues of $A$:

- The condition number $\kappa(A) = \lambda_{max}/\lambda_{min}$ determines how quickly the method converges
- The optimal fixed step size is $\alpha = 2/(\lambda_{max} + \lambda_{min})$
- The convergence rate is approximately $((\kappa-1)/(\kappa+1))^2$ per iteration

## Steepest Descent vs. Conjugate Gradient Method

### Method of Steepest Descent

The method of steepest descent (also called gradient descent) iteratively minimizes a quadratic function by following the direction of the negative gradient (or equivalently, the residual vector for linear systems).

For solving $Ax = b$ where $A$ is symmetric positive definite, we define the residual as $r_k = b - Ax_k$, which is the negative gradient of the quadratic function $f(x) = \frac{1}{2}x^TAx - b^Tx$.

Each iteration computes:

1. The residual/steepest descent direction: $r_k = b - Ax_k$
2. The optimal step size: $\alpha_k = \frac{r_k^T r_k}{r_k^T A r_k}$
3. The new solution: $x_{k+1} = x_k + \alpha_k r_k$

The optimal step size is derived by minimizing $f(x_k + \alpha r_k)$ with respect to $\alpha$. Taking the derivative and setting it to zero:

$$\frac{d}{d\alpha}f(x_k + \alpha r_k) = r_k^T A (x_k + \alpha r_k) - r_k^T b = 0$$

Simplifying and solving for $\alpha$:

$$r_k^T A x_k + \alpha r_k^T A r_k - r_k^T b = 0$$ $$\alpha r_k^T A r_k = r_k^T (b - A x_k) = r_k^T r_k$$ $$\alpha_k = \frac{r_k^T r_k}{r_k^T A r_k}$$

This ensures we take the optimal step in the chosen direction.

While intuitive, steepest descent has a critical flaw: successive directions tend to oscillate, particularly in narrow valleys of the function landscape. After minimizing in one direction, the next step often partially undoes previous progress.

The error in steepest descent decreases at best by a factor of:

$$\frac{\kappa(A) - 1}{\kappa(A) + 1}$$

per iteration, where $\kappa(A)$ is the condition number of $A$. For ill-conditioned problems where $\kappa(A)$ is large, convergence becomes extremely slow.

### Conjugate Gradient Method

The conjugate gradient (CG) method resolves the inefficiency of steepest descent by choosing search directions that are $A$-conjugate:

$$p_i^T A p_j = 0 \text{ for } i \neq j$$

This property ensures that minimizing along a new direction preserves the progress made in previous directions. Unlike steepest descent, which may revisit the same subspaces repeatedly, conjugate gradient explores a new subspace with each iteration.

The key insight is that by constructing $A$-conjugate directions, CG effectively diagonalizes the problem in the search space, eliminating the zigzagging behavior of steepest descent.

The conjugate gradient method has remarkable theoretical properties:

- In exact arithmetic, it converges in at most $n$ iterations (where $n$ is the dimension)
- Each new search direction is built as a linear combination of the current residual and previous direction
- The residuals are orthogonal: $r_i^T r_j = 0$ for $i \neq j$
- The search directions are $A$-conjugate: $p_i^T A p_j = 0$ for $i \neq j$

In practice, CG often converges much faster than the theoretical $n$ iterations because:

1. The matrix $A$ may have clustered eigenvalues, effectively reducing the dimensionality
2. We often don't need full precision, so early termination is possible
3. Preconditioning can dramatically accelerate convergence

The theoretical convergence rate of conjugate gradient depends on the eigenvalue distribution of $A$. If $A$ has only $m$ distinct eigenvalues, CG converges in at most $m$ iterations. More generally, after $k$ iterations, the error is bounded by:

$$\frac{\|x_k - x^*\|_A}{\|x_0 - x^*\|_A} \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$

This is much better than steepest descent, especially for ill-conditioned problems.

## Preconditioning

For ill-conditioned problems (where $\kappa(A)$ is large), both steepest descent and conjugate gradient converge slowly. Preconditioning addresses this by transforming the original system into an equivalent one with better conditioning.

Instead of solving $Ax = b$, we solve:

$$M^{-1}Ax = M^{-1}b$$

Where $M$ is the preconditioner—ideally, a matrix that approximates $A$ but is easy to invert.

### Why Preconditioning Improves Performance

Preconditioning works by transforming the eigenvalue distribution of the system matrix. To understand this, consider the preconditioned system:

$$M^{-1}Ax = M^{-1}b$$

If $M$ approximates $A$ well, then $M^{-1}A$ approximates the identity matrix, which has perfect conditioning (all eigenvalues equal to 1).

Mathematically, the effectiveness of preconditioning comes from:

1. Eigenvalue Clustering: If $M^{-1}A$ has eigenvalues clustered around 1, convergence will be rapid. For conjugate gradient, convergence in $k$ steps is guaranteed if there are only $k$ distinct eigenvalues.
    
2. Condition Number Reduction: The condition number $\kappa(M^{-1}A)$ is typically much smaller than $\kappa(A)$. Since the convergence rate depends on: $$\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$ reducing $\kappa$ from 10,000 to 10 can make the difference between thousands of iterations and just a few.
    
3. Spectral Transformation: Preconditioning transforms the spectrum (eigenvalues) of the operator. For example, if $A$ has eigenvalues ranging from 10^-6 to 10^2, a good preconditioner might compress this range to 0.5 to 2.0.
    

The theoretical foundation was established by Axelsson and Lindskog (1986), who showed that for symmetric positive definite systems, the optimal preconditioner in the Frobenius norm is:

$$M_{opt} = (diag(A))^{1/2} \cdot A \cdot (diag(A))^{-1/2}$$

In practice, we use more computationally feasible approximations.

### Common Preconditioners:

1. **Diagonal (Jacobi) Preconditioner**: $M = \text{diag}(A)$
    
    - Simply uses the diagonal elements of $A$
    - Easy to compute and apply: $M^{-1} = \text{diag}(1/a_{11}, 1/a_{22}, ..., 1/a_{nn})$
    - Effective when off-diagonal elements are small relative to diagonal ones
    - Memory overhead is minimal: O(n)
2. **Incomplete Cholesky Factorization (IC)**: $M \approx LL^T$
    
    - Computes an approximate Cholesky factorization by discarding fill-in
    - IC(0) maintains the sparsity pattern of A
    - Higher levels (IC(k)) allow more fill-in for better approximation
    - Benzi et al. (2002) demonstrated 5-10x speedup for elliptic PDEs
3. **Sparse Approximate Inverse (SPAI)**:  $M\approx A^{-1}$
    
    - Directly approximates the inverse by minimizing $\lvert AM-I \rvert_{F}$
    - Preserves sparsity by limiting the pattern of M
    - Highly parallelizable application in each iteration
    - Particularly effective for highly irregular problems
4. **Algebraic Multigrid (AMG)**:
    
    - Constructs a hierarchy of problems at different resolutions
    - Smooths errors at each level using simple iterative methods
    - Transfers information between levels via restriction and prolongation
    - Nearly optimal complexity for elliptic PDEs: O(n log n)
    - Demonstrated by Stüben (2001) to achieve mesh-independent convergence
5. **Domain Decomposition**:
    
    - Splits the domain into subdomains with some overlap
    - Solves local problems in parallel
    - Combines solutions via weighted averaging
    - Scales well on parallel architectures

## Practical Algorithms and Python Code

Let's implement both gradient descent and conjugate gradient for solving the linear system $Ax = b$:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(A, b, x0, alpha=None, max_iter=1000, tol=1e-6):
    x = x0.copy()
    r = b - A @ x
    iterations = [x.copy()]
    residuals = [np.linalg.norm(r)]
    
    # If no step size provided, use optimal for quadratic
    if alpha is None:
        eigvals = np.linalg.eigvalsh(A)
        alpha = 2 / (max(eigvals) + min(eigvals))
    
    for i in range(max_iter):
        if residuals[-1] < tol:
            break
            
        x = x + alpha * r
        r = b - A @ x
        
        iterations.append(x.copy())
        residuals.append(np.linalg.norm(r))
    
    return x, residuals, iterations

def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-6):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    
    iterations = [x.copy()]
    residuals = [np.linalg.norm(r)]
    
    for i in range(max_iter):
        if residuals[-1] < tol:
            break
            
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        
        iterations.append(x.copy())
        residuals.append(np.linalg.norm(r))
    
    return x, residuals, iterations

# Example 
def create_test_problem(n=100, condition_number=100):
    """Create a test problem with specified condition number."""
    # Create random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    
    # Create diagonal matrix with desired condition number
    diag = np.linspace(1, condition_number, n)
    A = Q.T @ np.diag(diag) @ Q
    
    # Create random solution and right-hand side
    x_true = np.random.randn(n)
    b = A @ x_true
    
    return A, b, x_true

# Visualization 2D 
def visualize_2d_comparison(A, b, x_gd, x_cg, gd_iterations, cg_iterations):
    """Create visualization comparing GD and CG convergence in 2D."""
    # Create meshgrid for contour plot
    delta = 0.1
    x1 = np.arange(-3, 3, delta)
    x2 = np.arange(-3, 3, delta)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)

    # Compute function values
    for i in range(len(x1)):
        for j in range(len(x2)):
            x = np.array([X1[j, i], X2[j, i]])
            Z[j, i] = 0.5 * x.dot(A).dot(x) - b.dot(x)

    # Plot contours and convergence paths
    plt.figure(figsize=(12, 10))
    plt.contour(X1, X2, Z, 20, cmap='RdBu')
    plt.plot([x[0] for x in gd_iterations], [x[1] for x in gd_iterations], 'o-', label='Gradient Descent')
    plt.plot([x[0] for x in cg_iterations], [x[1] for x in cg_iterations], 's-', label='Conjugate Gradient')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.title('Convergence Paths')
    plt.grid(True)
    
    return plt

# Run a simple example
A = np.array([[5, 2], [2, 3]])  # Symmetric positive definite
b = np.array([1, 2])
x0 = np.zeros(2)

x_gd, gd_residuals, gd_iterations = gradient_descent(A, b, x0)
x_cg, cg_residuals, cg_iterations = conjugate_gradient(A, b, x0)

# Plotting residuals
plt.figure(figsize=(10, 6))
plt.semilogy(gd_residuals, 'o-', label='Gradient Descent')
plt.semilogy(cg_residuals, 's-', label='Conjugate Gradient')
plt.xlabel('Iteration')
plt.ylabel('Residual Norm (log scale)')
plt.legend()
plt.title('Convergence Rate Comparison')
plt.grid(True)
plt.show()

# Visualize paths in 2D
vis_plt = visualize_2d_comparison(A, b, x_gd, x_cg, gd_iterations, cg_iterations)
vis_plt.show()
```


<div class="image-container"><img src="/assets/images/gradient/race1.png" style="width:1000px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-5: Convergence Rate By Iteration</em></p>


<div class="image-container"><img src="/assets/images/gradient/race.png" style="width:1000px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Fig-6: Convergence Path on Contour Overlay</em></p>


## Variations and Use-Cases

### Variations of Descent Methods

1. **Stochastic Gradient Descent (SGD)**
    
    - Uses random subsets of data (mini-batches)
    - Essential for large-scale machine learning
    - Trade-offs: noise for computational efficiency
2. **Momentum Methods**
    
    - Add a velocity component to "coast" through valleys
    - Help overcome small local minima and saddle points
    - Examples: Classical momentum, Nesterov accelerated gradient
3. **Adaptive Learning Rate Methods**
    
    - Adjust step sizes automatically for each parameter
    - Examples: AdaGrad, RMSProp, Adam
    - Critical for training deep neural networks
4. **Limited-Memory BFGS**
    
    - Approximates second-order information (Hessian)
    - Combines the rapid convergence of Newton's method with the memory efficiency of gradient descent
    - Popular for general nonlinear optimization
5. **Preconditioned Conjugate Gradient**
    
    - Combines preconditioning with conjugate gradient
    - State-of-the-art for large sparse systems
    - Used in scientific simulations, finite element methods, etc.

## Conclusion

From the simplicity of gradient descent to the efficiency of preconditioned conjugate gradient, descent algorithms demonstrate how deeply linear algebra underpins modern computational methods.

As we've seen, the key insights come from understanding:

- How the geometry of the problem affects convergence
- How to choose directions and step sizes intelligently
- How preconditioning transforms ill-conditioned problems into well-conditioned ones

---

**References:**

1. _Advanced Linear Algebra: Foundations to Frontiers_, Chapter 8
2. Nocedal, J., & Wright, S. (2006). _Numerical Optimization_
3. Shewchuk, J. R. (1994). _An Introduction to the Conjugate Gradient Method Without the Agonizing Pain_