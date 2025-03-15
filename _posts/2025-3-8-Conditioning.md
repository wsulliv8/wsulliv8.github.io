---
layout: post
title: "When Numbers Betray: A Practical Guide to Numerical Stability and Conditioning"
image: /assets/images/conditioning/stability.png
excerpt: Numerical stability vs conditioning. Two important metrics when solving linear systems.
---

In computational mathematics, an algorithm's numerical stability and a problem's conditioning play a crucial role in ensuring accurate and reliable results. They independently amplify error when solving linear systems which means if either the algorithm is numerically unstable and/or the problem is ill-conditioned, solution accuracy will be drastically reduced. Recognizing when errors may be introduced by stability or conditioning and understanding methods to improve solution error is essential for computation scientists and machine learning engineers.

# Numerical Stability

Numerical stability refers to how errors propagate through an algorithm. An algorithm is considered stable if small changes in input (including round-off errors from floating-point arithmetic) result in proportionally small changes in output. Conversely, an unstable algorithm amplifies these errors, leading to unreliable results.

### Floating-Point Arithmetic and Error Propagation
Computers represent real numbers using finite precision, which introduces small errors. These errors accumulate in iterative or large-scale computations. For instance, summing a series of floating-point numbers in a different order can lead to different results due to rounding errors.

In general, floating-point representation adheres to:

$$
fl(\chi \, op \, \psi) = (\chi \, op \, \psi)(1 + \epsilon), \quad \text{where} \quad |\epsilon| \leq \epsilon_{mach}
$$

-  $\epsilon_{mach}$ is machine epsilon which is the smallest positive floating point number that, when added to $1$, results in a value different than $1$. 

$$
\epsilon_{mach}=min\{\chi>0 \mid 1.0 +\chi \neq 1.0  \}
$$

- For IEEE (Institute of Electrical and Electronics Engineers) single-precision (32-bit): $\epsilon_{mach}\approx1.19\times 10^{-7}$
- For IEEE double-precision (64-bit): $\epsilon_{mach}\approx2.22\times 10^{-16}$

<aside> <strong>Python to Compute Machine Epsilon</strong><br>
{% highlight python %} 
import numpy as np 
eps = 1.0 
while 1.0 + eps != 1.0: 
  eps /= 2 eps *= 2 # Last value before breaking loop 
print(eps)
{% endhighlight %} 
</aside>

- $op$ represents arithmetic operations such as addition or multiplication.

### Example: Gaussian Elimination and Stability
Gaussian elimination is a fundamental method for solving linear systems. However, *without* partial pivoting (swapping the row with the largest pivot element in each iteration thereby reducing the chance of dividing by a small number), it can be numerically unstable, leading to large errors. Partial pivoting improves stability by reducing round-off errors, ensuring a more accurate solution.

Take the matrix:
$$
A=\begin{pmatrix}
10^{-20}  & 1 \\
1 & 1
\end{pmatrix}
$$
Without pivoting:
- First pivot $=10^{-20}$
- Multiplier $=\frac{1}{10^{-20}}$ 

This large multiplier creates enormous coefficients in subsequent calculations. Partial pivoting acts as a numerical damping mechanism and makes Gaussian elimination more stable.
#### **Theorem: Backward Stability**
An algorithm solving $Ax = b$ is backward stable if it computes an approximate solution $\hat{x}$ such that:

$$
(A + \Delta A)\hat{x} = b + \Delta b, \quad \text{where} \quad \frac{\|\Delta A\|}{\|A\|} + \frac{\|\Delta b\|}{\|b\|} = O(\epsilon_{mach})
$$

The above  indicates that the computed approximate solution, $\hat{x}$, is the exact solution to a slightly perturbed problem and the peturbation is small relative to machine precision.

### Python Demonstration of Numerical Stability

To illustrate the impact of unstable algorithms, let's analyze the effect of catastrophic cancellation in a numerically unstable algorithm ($\frac{1-\cos (x)}{x^{2}}$). A separate, numerically stable, algorithm that uses the half-angle identity ($\frac{\left( \frac{\sin\left( \frac{x}{2} \right)}{\frac{x}{2}}  \right)^{2}}{2}$) is compared against the unstable algorithm. 

<aside> 
{% highlight python %} 
import numpy as np
import matplotlib.pyplot as plt

def unstable_calculation(x):
	return (1 - np.cos(x)) / (x**2)

def stable_calculation(x):
	y = x / 2
	return (np.sin(y) / y)**2 / 2

# Generate points to demonstrate, with more points near zero where issues occur
x_values = np.concatenate([
	np.linspace(1e-16, 1e-15, 10),
	np.linspace(1e-15, 1e-14, 10),
	np.linspace(1e-14, 1e-13, 10),
	np.linspace(1e-13, 1e-12, 10),
	np.linspace(1e-12, 1e-10, 10),
	np.linspace(1e-10, 1e-8, 10),
	np.linspace(1e-8, 1e-6, 10),
	np.linspace(1e-6, 1e-4, 10),
	np.linspace(1e-4, 1e-2, 10),
	np.linspace(1e-2, 0.1, 10)
])

# The true value should be close to 0.5 for small x
true_value = 0.5

# Use stable and unstable methods
unstable_results = [unstable_calculation(x) for x in x_values]
stable_results = [stable_calculation(x) for x in x_values]

# Relative errors
unstable_errors = [abs(result - true_value) / true_value for result in unstable_results]
stable_errors = [abs(result - true_value) / true_value for result in stable_results]

print("For small x values, we expect (1-cos(x))/x² to be close to 0.5")
print("\nExample calculations:")
for i in range(0, len(x_values), len(x_values) // 5):
	x = x_values[i]
	print(f"x = {x:.1e}")
	print(f" Unstable formula: {unstable_results[i]:.10f} (error: {unstable_errors[i]:.10f})")
	print(f" Stable formula: {stable_results[i]:.10f} (error: {stable_errors[i]:.10f})")
 

plt.figure(figsize=(10, 6))
plt.loglog(x_values, unstable_errors, 'r.-', label='Unstable Formula')
plt.loglog(x_values, stable_errors, 'g.-', label='Stable Formula')
plt.axhline(y=1e-16, color='k', linestyle='--', label='Machine Epsilon (~1e-16)')
plt.xlabel('x value (log scale)')
plt.ylabel('Relative Error (log scale)')
plt.title('Numerical Stability Comparison')
plt.legend()
plt.grid(True)
plt.show()

OUTPUT:
For small x values, we expect (1-cos(x))/x² to be close to 0.5 
Example calculations: 
x = 1.0e-16 Unstable formula: 0.0000000000 (error: 1.0000000000) Stable formula: 0.5000000000 (error: 0.0000000000) 
x = 1.0e-14 Unstable formula: 0.0000000000 (error: 1.0000000000) Stable formula: 0.5000000000 (error: 0.0000000000) 
x = 1.0e-12 Unstable formula: 0.0000000000 (error: 1.0000000000) Stable formula: 0.5000000000 (error: 0.0000000000) 
x = 1.0e-08 Unstable formula: 0.0000000000 (error: 1.0000000000) Stable formula: 0.5000000000 (error: 0.0000000000)
x = 1.0e-04 Unstable formula: 0.4999999970 (error: 0.0000000061) Stable formula: 0.4999999996 (error: 0.0000000008)
{% endhighlight %}
</aside>

<div class="image-container"><img src="/assets/images/conditioning/stability.png" style="width:1000px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Stable vs. Unstable Algorithm</em></p>
As the figure depicts, the unstable variant suffers dramatic precision loss for small $x$ values. This is due to the fact that when $x$ is small, $\cos(x)$ is near $1$ which means $(1-\cos (x))$ subtracts two nearly identical numbers, causing catastrophic cancellation. Additionally, this error-dominated result is then divided by a very small number ($x^{2}$) which amplifies this error dramatically. The alternate equation avoids these issues and is far more numerically stable.

### Common Stable vs. Unstable Algorithms

Below is a comparison of several common linear algebra algorithms and their stability properties:

| Algorithm                                  | Stability            | Details                                                                                      |
| ------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------- |
| QR Factorization with Householder          | Stable               | Provides excellent numerical stability for solving linear systems and least squares problems |
| Modified Gram-Schmidt                      | Stable               | Ensures orthogonality is maintained to high precision                                        |
| Classical Gram-Schmidt                     | Unstable             | Loses orthogonality rapidly in finite precision                                              |
| LU with Partial Pivoting                   | Stable               | Standard method in most software libraries                                                   |
| LU without Pivoting                        | Potentially Unstable | Can fail catastrophically for certain matrices                                               |
| SVD (Singular Value Decomposition)         | Highly Stable        | Works well even for ill-conditioned or rank-deficient problems                               |
| Explicit Normal Equations $(A^TA)x = A^Tb$ | Unstable             | Squares the condition number, amplifying errors                                              |
| Householder Transformations                | Stable               | Orthogonal transformations preserve conditioning                                             |
| Gauss-Jordan Elimination                   | Less Stable          | More prone to roundoff error than LU factorization                                           |
| Power Method with Normalization            | Stable               | Convergence depends on eigenvalue separation                                                 |
| Power Method without Normalization         | Unstable             | Can lead to overflow/underflow                                                               |


# Conditioning of a Problem

Conditioning measures how sensitive a problem is to small changes in input. A problem is well-conditioned if small perturbations in input lead to small changes in output. It is ill-conditioned if small changes in input cause large changes in output.

### Condition Number of a Matrix
For a square matrix $A$, the condition number $\kappa(A)$ is defined as:

$$
\kappa(A) = \|A\| \cdot \|A^{-1}\|
$$

A large condition number indicates that the system is ill-conditioned. 

If the two-norm is used in the above definition, it becomes:

$$
\kappa(A)=\frac{\sigma_{max}}{\sigma_{min}}
$$

- $\sigma_{max}$ and $\sigma_{min}$ are the largest and smallest singular values of $A$ (see [SVD Blog](https://wsulliv8.github.io/SVD/))
- A matrix is singular if $\sigma_{min}=0$ (it has no inverse), but if $\sigma_{min}$ is very close to $0$ (the matrix is very close to being singular), $k(A)$ becomes very large.
- Simple Example:

$$
A=\begin{pmatrix}
1 & 0  \\
0 & 0.0001
\end{pmatrix}
$$

$$
\kappa(A)=\frac{1}{0.0001}=10^{4}
$$

#### **Theorem: Sensitivity of Linear Systems**
Relating the condition number to the linear system $Ax=b$ , we said that conditioning is a measure of how small perturbations in the input ($b$) are amplified into perturbations in the output ($x$). The equation below depicts this:

$$
\frac{\|\Delta x\|}{\|x\|} \leq \kappa(A)\frac{\|\Delta b\|}{\|b\|}
$$

This shows that a large condition number amplifies small input errors into large output errors.

Similar results may be shown for other problems linear algebra such as the Linear Least Squares (LLS) problem which aims to find the $\hat{x}$ that satisfies:


$$
\lvert \lvert b-A\hat{x} \rvert  \rvert _{2}= \min_{x}\lvert \lvert b-Ax \rvert  \rvert _{2}
$$

The error equation becomes:

$$
\frac{\|\Delta x\|}{\|x\|} \leq \frac{1}{\cos(\theta)}\kappa(A)\frac{\|\Delta b\|}{\|b\|}
$$

This could spell trouble if one uses the Method of Normal Equations to solve the LLS problem which would turn the above equation into:


$$
\frac{\|\Delta x\|}{\|x\|} \leq \kappa^{2}(A)\frac{\|\Delta b\|}{\|b\|}
$$

This is bad! Squaring the condition number could result in a much larger error in the result. Luckily solving the LLS problem via other methods such as $QR$ factorization can improve conditioning.

### Python Demonstration of Conditioning 

<aside> 
{% highlight python %} 
import numpy as np
# Create an ill-conditioned matrix
A = np.array([[1, 1], [1, 1.0001]])
b = np.array([2, 2.0001])

# Condition number of A
print("Condition number of A:", np.linalg.cond(A))

# Normal equations approach
AtA = A.T @ A
print("Condition number of A^T A:", np.linalg.cond(AtA)) # Much worse

# QR approach (better stability)
Q, R = np.linalg.qr(A)
print("Condition number of R (QR approach):", np.linalg.cond(R)) # Similar to A
{% endhighlight %} 
Output: <br>
Condition number of A: 40002.00007491187 <br>
Condition number of A^T A: 1600159720.6211963 <br>
Condition number of R (QR approach): 40002.00007491187
</aside>
### Visualization of Condition Number Effects

The following figures illustrates how the condition number affects linear transformations and sensitivity of solutions:


<div class="image-container"><img src="/assets/images/conditioning/condition.png" style="width:1400px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Transformation Visualization</em></p>
*Figure 1: Depicts transformation of unit circle by matrices with different condition numbers. Blue circle is input space, red ellipse depicts transformation, and green/purple arrows show how error is amplified*
<div class="image-container"><img src="/assets/images/conditioning/LLS.png" style="width:1400px;"></div>
<p style="font-size: 0.8rem; text-align: center;"><em>Solution Space Visualization</em></p>

*Figure 2: Depicts how condition number affects sensitivity of solutions. Color contours are residual norm $\lvert \lvert Ax-b \rvert \rvert$, red start is true solution, green dot shows how small perturbation in $b$ causes shift in solution*

# Warning Signs and Practical Mitigation Strategies

### Warning Signs

1. **Inconsistent results** - Different runs or slight input variations produce dramatically different outputs
2. **Loss of precision** - Results have fewer significant digits than expected
3. **Failure to converge** - Iterative methods fail to reach desired tolerance
4. **Solutions don't satisfy original constraints** - Forward error checks show large residuals

### Mitigation Strategies

1. **Use stable algorithms** - Use known numerically stable algorithms (ex. QR instead of Classical Gram-Schmidt)
2. **Apply preconditioning** - Transform ill-conditioned problems to improve their condition number (see [Gradient Descent Blog](https://wsulliv8.github.io/gradient-descent/))
3. **Implement scaling** - Normalize inputs to avoid large magnitude differences
4. **Increase precision** - Use double or extended precision
5. **Reorder computations** - Changing the order of operations may improve stability (ex. Kahan summation)
6. **Verify results** - Calculate residuals ($\lvert \lvert Ax-b \rvert \rvert$) to check solution quality
7. **Use orthogonal transformations** - Methods based on orthogonal matrices (like QR) preserve conditioning

# Real-World Applications

### Machine Learning

In deep learning, training neural networks involves optimizing highly non-convex functions with millions of parameters. Ill-conditioning appears as:
- **Vanishing/exploding gradients** - An ill-conditioned Hessian causes gradients to either vanish or explode during backpropagation
- **Slow convergence** - Optimization methods struggle with ill-conditioned loss landscapes
- **Sensitivity to initialization** - Small changes in initial weights lead to completely different models

**Mitigation strategies:**
- Batch normalization to improve conditioning of the optimization problem
- Residual connections to provide better gradient flow
- Adaptive optimizers like Adam that implicitly handle ill-conditioning

## Conclusion

Understanding numerical stability and problem conditioning is essential for designing robust numerical algorithms. A well-conditioned problem paired with a stable algorithm ensures accurate results, whereas an ill-conditioned problem or unstable algorithm can lead to significant computational errors. 

In real-world applications from machine learning to engineering, these concepts are not merely theoretical, but have practical implications for the reliability and efficiency of computational methods. By analyzing condition numbers, employing stable methods like pivoting, and implementing appropriate preconditioning strategies, these issues may be mitigated by improving the reliability of numerical computations.

As datasets grow larger and mathematical models become more complex, the importance of numerically sound algorithms only increases. Being able to recognize the warning signs of stability and conditioning issues and knowing how to address them is a critical skill for anyone working in computational science and engineering.