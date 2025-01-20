
## 2.5.1.1

$U\in \mathbb{C}^{m \times m}$ is unitary if and only if $(Ux)^H(Uy)=x^Hy$ for all $x,y \in \mathbb{C}^m$.

___

$\Rightarrow$ Assume $U$ is unitary.
$$(Ux)^H(Uy)$$
<$(Az)^H=z^HA^H$>
$$=x^HU^HUy$$
<$U^HU=I$ if $U$ is unitary.>
$$=x^Hy$$
_______________________________________________

$\Leftarrow$ Assume $(Ux)^H(Uy)=x^Hy$ for all $x,y \in \mathbb{C}^m$
$$(Ux)^H(Uy)=x^Hy$$
<$(Az)^H=z^HA^H$>
$$=x^HU^HUy=x^Hy$$
For the above equality to hold, $U^HU=I$ must be true. Also, $U$ is a square matrix. Therefore $U$ is a unitary matrix.

_______________________________________________
$\therefore$ $U\in \mathbb{C}^{m \times m}$ is unitary if and only if $(Ux)^H(Uy)=x^Hy$ for all $x,y \in \mathbb{C}^m$.

****
____

## 2.5.1.2

Let $A,B \in \mathbb{C}^{m \times n}$. Furthermore, let $U \in \mathbb{C}^{m \times m}$ and $V \in \mathbb{C}^{n \times n}$ be unitary.

TRUE/FALSE: $UAV^H=B \iff U^HBV=A$.
____
$\Longrightarrow$
$$U^HBV$$
<$UAV^H=B$>
$$=U^HUAV^HV$$
<$C^HC=I$ if $C$ is unitary>
$$=IAI$$
<$CI=C$>
$$=A$$
_____
$\Longleftarrow$
$$UAV^H$$
$<U^HBV=A>$
$$=UU^HBVV^H$$
$<C^HC=I$ if $C$ is unitary$>$
$$=IBI$$
$<CI=C>$
$$=B$$
___
TRUE
___
___

## 2.5.1.3

Prove that nonsingular $A \in \mathbb{C}^{n \times n}$ has a condition number $\kappa _{2} (A)=1$ if and only if $A= \sigma Q$ where $Q$ is unitary and $\sigma \in \mathbb{R}$ is positive.
___
$\Longleftarrow$ If $A=\sigma Q$ then,
$$\kappa _{2}(A)$$
$<$definition of condition number$>$
$$
=\lvert \lvert A \rvert  \rvert_{2} \; \lvert \lvert A^{-1} \rvert \rvert _{2}  
$$
$<A=\sigma Q>$
$$
=\lvert \lvert \sigma Q \rvert  \rvert _{2} \; \lvert \lvert (\sigma Q)^{-1} \rvert  \rvert _{2}
$$
$<{(cA)}^{-1}=\frac{1}{c}A^{-1}>$
$$
=\lvert \lvert \sigma Q \rvert  \rvert _{2} \; \left\lvert  \left\lvert  \frac{1}{\sigma}Q^{-1}  \right\rvert   \right\rvert _{2}
	$$
$<$homogeneity$>$
$$
\lvert \sigma \rvert \; \lvert \lvert Q \rvert  \rvert _{2} \left\lvert  \frac{1}{\sigma}  \right\rvert \; \lvert \lvert Q^{-1} \rvert  \rvert _{2}
$$
$<$cancel terms$>$
$$
\lvert \lvert Q \rvert  \rvert _{2} \; \lvert \lvert Q^{-1} \rvert  \rvert _{2}
	$$
$<Q$ and $Q^{-1}$ are both unitary; $\lvert \lvert A \rvert \rvert_{2}=1$ if $A$ is unitary $>$
$$
=1
$$
____
$\Longrightarrow$ If $\kappa_{2}(A)=1$ then 
$$
A
$$
$<$ Replace with SVD where $U$ and $V$ are unitary and $\Sigma$   is a diagonal matrix with the singular values of $A>$
$$
=U\Sigma V^{H}

$$
$<$ If $\kappa_{2}(A)=1$ then $\sigma_{0}=\sigma_{1}=\dots=\sigma_{n-1}>0>$
$$
=U(\sigma I)V^{H}
$$
$<aI=a$; commutativity of scalar multiplication$>$
$$
=\sigma UV^{H}
$$
$<$ Change of terms; $U_{0},U_{1}\in \mathbb{C}^{n\times n}$ and unitary then $U_{0}U_{1}$ are unitary $>$
$$
=\sigma Q
$$

___
___

## 2.5.1.4

Let $U\in \mathbb{C}^{m\times m}$ and $V\in \mathbb{C}^{n\times n}$ be unitary.
ALWAYS/SOMETIMES/NEVER: The matrix $\begin{pmatrix} U & 0 \\ 0 & V \end{pmatrix}$ is unitary.
____
For some $A\in \mathbb{C}^{m\times m}, A$ is unitary iff $A^{H}A=I$ and $A$ is square. Two square matrices partitioned together as above yields a square matrix that is $\mathbb{C}^{(m+n)\times(m+n)}$

Let $Q\in \mathbb{C}^{(m+n)\times(m+n)}= \begin{pmatrix} U & 0 \\ 0 & V \end{pmatrix}$
$$
Q^{H}Q
$$
< as defined >
$$
=\begin{pmatrix} U & 0 \\ 0 & V \end{pmatrix}^{H}\begin{pmatrix} U & 0 \\ 0 & V \end{pmatrix}
$$
< hermitian transpose of partitioned matrix >

$$
=\begin{pmatrix} U^{H} & 0 \\ 0 & V^{H} \end{pmatrix}\begin{pmatrix} U & 0 \\ 0 & V \end{pmatrix}
$$
< partitioned matrix-matrix multiplication >
$$
\begin{pmatrix} U^{H}U & 0 \\ 0 & V^{H}V \end{pmatrix}
$$
$<A^{H}A=I$ if $A$ is unitary; $U$ and $V$ are unitary >  
$$
\begin{pmatrix} I & 0 \\ 0 & I \end{pmatrix}
$$
< combine partitioned matrices >
$$
=I
$$
___
ALWAYS

____
____

## 2.5.1.5

Matrix $A\in \mathbb{R}^{m\times m}$ is a stochastic matrix $\iff$ it is nonnegative (all its entries are nonnegative) and the entries in its columns sum to one: $\sum_{i=0}^{m-1}\alpha_{i,j}=1$. Show that a matrix $A$ is both unitary and stochastic $\iff$ it is a permutation matrix.
___
Let $A=P\in \mathbb{R}^{m\times m}=P(p)=\begin{pmatrix}e^{T}_{k_{0}} \\ e^{T}_{k_{1}} \\ \vdots  \\ e^{T}_{k_{m-1}} \end{pmatrix}$ be a permutation matrix.
$\Longleftarrow$

1) $$
P
$$
< as defined >
$$
=\begin{pmatrix}e^{T}_{k_{0}} \\ e^{T}_{k_{1}} \\ \vdots  \\ e^{T}_{k_{m-1}} \end{pmatrix}
$$
< since $P$ is an $I$ with its rows (columns) switched, there can only be one "1" in each row and column; transpose $P$ >
$$
=\begin{pmatrix}e_{k_{0}}  &  e_{k_{1}}  &  \dots   &  e_{k_{m-1}} \end{pmatrix}^{T}
$$
All entries of $P$ either 0 or 1, so $P$ is nonnegative. Also, $e_{j}$ is a vector of 0's with a 1 in the jth position. Since all columns of P are unit basis vectors, $\sum_{i=0}^{m-1}\alpha_{i,j}=1$ for all columns.
$\therefore P$ is stochastic.

2) Also, $U\in \mathbb{C}^{m\times m}$ is unitary $\iff$ $U^{H}U=I$ 
$$
P^{H}P
$$
< definition; $P\in \mathbb{R}^{m\times m}$, so $P^{H}=P^{T}$ >
$$
=\begin{pmatrix}e^{T}_{k_{0}} \\ e^{T}_{k_{1}} \\ \vdots  \\ e^{T}_{k_{m-1}} \end{pmatrix}^{T}\begin{pmatrix}e^{T}_{k_{0}} \\ e^{T}_{k_{1}} \\ \vdots  \\ e^{T}_{k_{m-1}} \end{pmatrix}
$$
< partitioned matrix-matrix multiplication >
$$
=e_{k_{0}}e_{k_{0}}^{T}+e_{k_{1}}e_{k_{1}}^{T}+\dots+e_{k_{m-1}}e_{k_{m-1}}^{T}
$$
< $e_{j}e_{j}^{k}$ yields a matrix where all elements $\epsilon=0$ except $\epsilon_{j,j}=1$ >
$$
=I
$$
Another way of looking at this is since $P$ is a permutation matrix, $P^{T}=P^{-1}$ because permutations are invertible by nature. So, $P^{H}P=P^{T}P=P^{-1}P=I$.

Also, $P$ is square.
$\therefore P$ is unitary.


___

$\Longrightarrow$ Let $A$ be unitary and stochastic.
$$
AA^{H}
$$
< $Q^{-1}=Q^{H}$ for unitary matrices >
$$
=AA^{-1}
$$
$$
=I
$$

Multiplying a permutation matrix by its inverse will yield the identity.

Also, since A is unitary, the columns of $A$ are mutually orthogonal vectors. Since $A$ is stochastic, $\alpha_{i,j}\ge 0$ for all $i,j$ and each column sums to 1. Combining the two, each column of $A$ must contain exactly one 1 with the rest of the entries equal to 0. $$
A=\begin{pmatrix}
a_{0} & a_{1} & \dots & a_{m-1}
\end{pmatrix}^{T}
					$$ $\lvert \lvert a_{j} \rvert \rvert_{2}=1$ and $\sum_{i=0}^{m-1}\alpha_{i,j}=1$ then $a_{j}=e_{j}$ (the jth unit basis vector). Since every column of $A$ has one entry equal to 1 and the rest equal to 0, $A$ is a permutation of $I$. Since $A^{H}A=I$ the rows of $A$ must behave in the same manner.

___
___

## 2.5.1.6

Show that if $\lvert \lvert \dots \rvert \rvert$ is a norm and $A$ is nonsingular, then $\lvert \lvert \dots \rvert \rvert_{A^{-1}}$ defined by $\lvert \lvert x \rvert \rvert_{A^{-1}}=\lvert \lvert A^{-1}x \rvert \rvert$ is a norm. Interpret this result in terms of the change of basis of a vector.

1) For $x\ne 0 \Rightarrow \lvert \lvert x \rvert \rvert_{A^{-1}}>0$ (positive definite)
$$
\lvert \lvert x \rvert  \rvert _{A^{-1}}
$$
< as defined >
$$
=\lvert \lvert A^{-1}x \rvert  \rvert
$$
< change of variables; $y=A^{-1}x>0$ since $A$ is nonsingular implies $A^{-1}$ $\Rightarrow$ $A^{-1}x=0$ only if $x=0$ >
$$
=\lvert \lvert y \rvert  \rvert
$$
< $\lvert \lvert \dots \rvert \rvert$ is a norm $\Rightarrow$ $\lvert \lvert \dots \rvert \rvert$ is positive definite >
$$
>0
$$
2) $\lvert \lvert \alpha x \rvert \rvert_{A^{-1}}=\lvert \alpha \rvert\lvert \lvert x \rvert \rvert_{A^{-1}}$ (homogeneous)
$$
\lvert \lvert \alpha x \rvert  \rvert _{A^{-1}}
$$
< as defined >
$$
=\lvert \lvert \alpha A^{-1}x \rvert  \rvert
	$$
< $\lvert \lvert \dots \rvert \rvert$ is a norm $\Rightarrow$ homogeneous >
$$
=\lvert \alpha \rvert \lvert \lvert A^{-1}x \rvert  \rvert
$$
< as defined >
$$
=\lvert \alpha \rvert \lvert \lvert x \rvert  \rvert _{A^{-1}}
$$
3) $\lvert \lvert x+y \rvert \rvert_{A^{-1}}\le \lvert \lvert x \rvert \rvert_{A^{-1}}+\lvert \lvert y \rvert \rvert_{A^{-1}}$ (obeys triangle inequality)
$$
\lvert \lvert x+y \rvert  \rvert _{A^{-1}}
$$
< as defined >
$$
=\lvert \lvert A^{-1}(x+y) \rvert  \rvert
$$
< matrix-vector multiplication is distributive >
$$
=\lvert \lvert A^{-1}x+A^{-1}y \rvert  \rvert
$$
<$\lvert \lvert \dots \rvert \rvert$ is a norm $\Rightarrow$ $\lvert \lvert \dots \rvert \rvert$ obeys the triangle inequality >
$$
\le \lvert \lvert A^{-1}x \rvert  \rvert +\lvert \lvert A^{-1}y \rvert  \rvert
$$
< as defined >
$$
=\lvert \lvert x \rvert  \rvert _{A^{-1}}+\lvert \lvert y \rvert  \rvert _{A^{-1}}
$$
____
Hence $\lvert \lvert x \rvert \rvert_{A^{-1}}$ is norm. 
Interpretation: When a vector changes basis, its norm may also change. In this instance, the norm changes from $\lvert \lvert \dots \rvert \rvert$ to $\lvert \lvert \dots  \rvert \rvert_{A^{-1}}$. If $A$ is unitary then $A^{-1}=A^{H}$ and $x$ can be written as a linear combination with $a_{i}^{H}x$ as the scalars.

___
___

## 2.5.1.7

Let $A\in \mathbb{C}^{m\times m}$ be nonsingular and $A=U\Sigma V^{H}$ be its SVD. The condition number of $A$ is given by:

1) $\kappa_{2}(A)=\lvert \lvert A \rvert \rvert_{2} \; \lvert \lvert A^{-1} \rvert \rvert_{2}$  < this is the definition of condition number >
2) $\kappa_{2}(A)=\frac{\sigma_{0}}{\sigma_{m-1}}$   < for this to hold, $A$ must be nonsingular which it is>
$$
\kappa_{2}(A)
$$
< definition >
$$
=\lvert \lvert A \rvert  \rvert _{2} \; \lvert \lvert A^{-1} \rvert  \rvert _{2}
$$
< definition of 2-norm; $A=U\Sigma V^{H}$; and $\lvert \lvert B^{-1} \rvert \rvert=\frac{1}{min_{\lvert \lvert x \rvert \rvert_{2}=1}\lvert \lvert Bx \rvert \rvert_{2}}$>
$$
=(\max_{\lvert \lvert x \rvert  \rvert _{2}=1}\lvert \lvert U\Sigma V^{H}x \rvert  \rvert _{2})\left( \frac{1}{\min _{\lvert \lvert x \rvert  \rvert _{2}=1}\lvert \lvert U\Sigma V^{H}x \rvert  \rvert _{2}} \right)
$$
< $\Sigma$ is a diagonal matrix and the $\lvert \lvert \dots \rvert \rvert_{2}$ of a diagonal matrix is $\max_{0 \le i<m}\lvert \delta_{i} \rvert$. Th largest diagonal value of $A^{-1}$ is the smallest of $A$. >
$$
=(\sigma_{0})\left( \frac{1}{\sigma_{m-1}} \right)
$$
$$
=\frac{\sigma_{0}}{\sigma_{m-1}}
$$
3. $$
\Sigma=U^{H}AV
$$
< rearrange SVD; partition matrices >
$$
=\begin{pmatrix}
u_{0}^{H} \\
\vdots \\
u_{m-1}^{H}
\end{pmatrix}(Av_{0}\dots Av_{m-1})
$$
< partitioned matrix matrix multiplication >
$$
=\begin{pmatrix}
u_{0}^{H}Av_{0} & \dots & 0 \\
0 & \ddots &  0 \\
0 & \dots & u_{m-1}^{H}Av_{m-1}
\end{pmatrix}
$$
< using $\kappa_{2}(A)=\frac{\sigma_{0}}{\sigma_{m-1}}$ from 2. ; select terms >
$$
=\frac{{u_{0}^{H}Av_{0}}}{u_{m-1}^{H}Av_{m-1}}
$$
$$
=\frac{\sigma_{0}}{\sigma_{m-1}}
$$
4. $$
(\max_{\lvert \lvert x \rvert  \rvert _{2}=1}\lvert \lvert U\Sigma V^{H}x \rvert  \rvert _{2})\left( \frac{1}{\min _{\lvert \lvert x \rvert  \rvert _{2}=1}\lvert \lvert U\Sigma V^{H}x \rvert  \rvert _{2}} \right)
$$
< $\Sigma$ is a diagonal matrix and the $\lvert \lvert \dots \rvert \rvert_{2}$ of a diagonal matrix is $\max_{0 \le i<m}\lvert \delta_{i} \rvert$. Th largest diagonal value of $A^{-1}$ is the smallest of $A$. >
$$
=(\sigma_{0})\left( \frac{1}{\sigma_{m-1}} \right)
$$
$$
=\frac{\sigma_{0}}{\sigma_{m-1}}
$$
		Equivalent to sub steps of 2.
___
Hence, all four give the condition number of $A$.

___
____

## 2.5.1.8

If $A\in \mathbb{C}^{m\times m}$ preserves length ($\lvert \lvert Ax \rvert \rvert_{2}=\lvert \lvert x \rvert \rvert_{2}$ for all $x \in \mathbb{C}^{m\times m}$) then $A$ is unitary. Give an alternative proof using the SVD.
____
Using SVD: A matrix $B\in \mathbb{C}^{m\times m}$ is unitary iff $B^{H}B=I$.

$$
\lvert \lvert Ax \rvert  \rvert _{2}^{2}
$$
< definition of 2-norm >
$$
=(Ax)^{H}(Ax)
$$
< distribute hermitian >
$$
=x^{H}A^{H}Ax
$$
< for A to preserve length, $A^{H}A$ must be $I$ >
$$
=x^{H}Ix
$$
< $Iy=y$ >
$$
=x^{H}x
$$
< definition of 2-norm >
$$
=\lvert \lvert x \rvert  \rvert _{2}^{2}
$$
	Square root beginning and end.
___
Since the above implies $A^{H}A=I$ for $A$ to preserve length:
$$
A^{H}A
$$
< split $A$ into SVD >
$$
=(U\Sigma V^{H})^{H}(U\Sigma V^{H})
$$
< distribute hermition >
$$
=V\Sigma^{H}U^{H}U\Sigma V^{H}
$$
<$B^{H}B=I$ if $B$ is unitary; $U$ is unitary >
$$
=V\Sigma^{H}\Sigma V^{H}
$$
< $B^{H}B=I$ if $B$ is unitary; $V$ is unitary; $\Sigma$ must equal $I$ >
$$
=I
$$
___
For $A^{H}A$ to equal $I$, $\Sigma=I$ (all singular values are 1)
So,
$$
A
$$
< split into SVD >
$$
=U\Sigma V^{H}
$$
<  $\Sigma=I$ >
$$
=UIV^{H}
$$
< $AI=A$ >
$$
=UV^{H}
$$
	Since $U$ and $V$ are unitary, their product ($A$) is unitary and $A$ is square. 
Hence, $A$ is unitary.

____
____

## 2.5.1.9

Prove $\lvert \lvert A \rvert \rvert_{2}\le \lvert \lvert A \rvert \rvert_{F}$ given $A\in \mathbb{C}^{m\times n}$ using SVD.
___
Using SVD of $A$:
$$
\lvert \lvert A \rvert  \rvert _{2}
$$
< split into SVD >
$$
=\lvert \lvert U\Sigma V^{H} \rvert  \rvert _{2}
$$
< $U$ and $V$ are unitary $\Rightarrow$ preserve length>
$$
=\lvert \lvert \Sigma \rvert  \rvert _{2}
$$
< $\lvert \lvert D \rvert \rvert_{2}=\max_{0 \le i<min(m,n)}\lvert \delta_{i} \rvert$ where $D$ is a diagonal matrix>
$$
=\sigma_{0}
$$
and
$$
\lvert \lvert A \rvert  \rvert _{F}
$$
< split into SVD >
$$
=\lvert \lvert U\Sigma V^{H} \rvert  \rvert_{F}
$$
< $U$ and $V$ are unitary $\Rightarrow$ preserve length >
$$
=\lvert \lvert \Sigma \rvert  \rvert _{F}
$$
< definition frobenius norm >
$$
=\sqrt{ \sigma_{0}^{2}+\dots+\sigma_{min(m,n)}^{2} }
$$
___
$$
\sigma_{0} \le \sqrt{ \sigma_{0}^{2}+\dots+\sigma_{min(m,n)}^{2} }
$$
Hence $\lvert \lvert A \rvert \rvert_{2}\le \lvert \lvert A \rvert \rvert_{F}$


____
____

## 2.5.1.10

Given $A\in \mathbb{C}^{m\times n}$, prove $\lvert \lvert A \rvert \rvert_{F} \le \sqrt{ r }\lvert \lvert A \rvert \rvert_{2}$, where r is the rank of matrix $A$.
___
From previous question 2.5.1.9,
$$
\lvert \lvert A \rvert  \rvert _{F}
$$
< split into SVD >
$$
=\lvert \lvert U\Sigma V^{H} \rvert  \rvert _{F}
$$
< unitary matrices preserve length >
$$
=\lvert \lvert \Sigma \rvert  \rvert _{F}
$$
< definition of Frobenius norm >
$$
=\sqrt{ \sigma_{0}^{2}+\dots+\sigma_{r-1}^{2} }
$$
< $\Sigma$ defined as $\sigma_{0} \ge \dots \ge \sigma_{r-1} >0$>
$$
\le \sqrt{ \sigma_{0}^{2}+\dots+\sigma_{0}^{2} }
$$
< algebra >
$$
=\sqrt{ r\sigma_{0}^{2} }
$$
< algebra >
$$
=\sqrt{ r }\sigma_{0}
$$
< $\lvert \lvert A \rvert \rvert_{2}=\sigma_{0}$>
$$
=\sqrt{ r }\lvert \lvert A \rvert  \rvert _{2}
$$
Hence, $\lvert \lvert A \rvert \rvert_{F} \le \sqrt{ r }\lvert \lvert A \rvert \rvert_{2}$
