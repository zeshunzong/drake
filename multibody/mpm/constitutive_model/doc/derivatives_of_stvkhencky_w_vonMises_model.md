How return mapping should be performed has already been documented. Here we derive the derivative relationships.

Below we denote $F$ as the deformation gradient *after* return mapping, i,e, the *.
​
Write 
$$F = U \Sigma V^T,$$
with $\Sigma = \operatorname{diag}(\sigma_0, \sigma_1, \sigma_2).$
​
Define the energy density function $\psi$ to be
$$
\begin{aligned}
\psi(F) &= \hat{\psi}(\sigma_0, \sigma_1, \sigma_2) \\
& =\mu\left[\left(\log \sigma_0\right)^2+\left(\log \sigma_1\right)^2+\left(\log \sigma_2\right)^2\right] +\frac{1}{2} \lambda\left[\left(\log \sigma_0\right)+\left(\log \sigma_1\right)+\left(\log \sigma_2\right)\right]^2.
\end{aligned}
$$
First Piola Stress is $\frac{d\psi}{dF}.$
$$P =\frac{d\psi}{dF}= U \frac{d\psi}{d\Sigma} V^T.$$
$$
\begin{aligned}
\frac{\partial \psi}{\partial \sigma_i} & =\frac{1}{\sigma_i}\left[2 \mu\left(\log \sigma_i\right)+\lambda\left(\log \sigma_0+\log \sigma_1+\log \sigma_2\right)\right] \\
& =\Sigma^{-1}[2 \mu(\log \Sigma)+\operatorname{trace}(\log \Sigma)].
\end{aligned}
$$
Next we compute $\frac{dP}{dF}.$
We follow the steps in [https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf] and [http://alexey.stomakhin.com/research/sca2012_tech_report.pdf].
Essentially this is computed as 
$$
\left(\frac{\partial {P}}{\partial {F}}({F})\right)_{i j r s}=\left(\frac{\partial P}{\partial F}(\Sigma)\right)_{k l m n} {U}_{i k} {V}_{j l}{U}_{r m} {V}_{s n}.
$$
Denote 
$$M_{klmn} = \left(\frac{\partial P}{\partial F}(\Sigma)\right)_{k l m n}.$$
This 9by9 or 3by3by3by3 matrix has 21 non-zero entries. See [http://alexey.stomakhin.com/research/sca2012_tech_report.pdf] for details. The results are the following
$$
M_{klmn}=\left\{\begin{array}{cl}
\frac{\partial^2 \hat{\psi}}{\partial \sigma_k \partial \sigma_m}, & \text { if } k=l, m=n, \text{ i.e. } M_{0000,0011,0022,1100,1111,1122,2200,2211,2222} \\
M_{ijij}, & \text { if }k=m=i, l=n=j, i\neq j, \text{ i.e. } M_{0101,0202,1010,1212,2020,2121}\\
M_{ijji}, & \text { if }k=n=i, l=m=j, i\neq j, \text{ i.e. } M_{0110,0220,1001,1221,2002,2112}\\
0, & \text{ ow,}
\end{array}\right.
$$
where
$$
M_{ijij} = \frac{\sigma_i \frac{\partial \hat{\psi}}{\partial \sigma_i} - \sigma_j \frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i^2 - \sigma_j^2} = \frac{1}{2}\cdot \left( \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j} +  \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}+\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i + \sigma_j}\right),\\
M_{ijji} = \frac{\sigma_j \frac{\partial \hat{\psi}}{\partial \sigma_i} - \sigma_i \frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i^2 - \sigma_j^2}= \frac{1}{2}\cdot \left( \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j} -  \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}+\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i + \sigma_j}\right).
$$
The 21 non-zero entries can be grouped into four matrices (9+4+4+4) $A, B_{01}, B_{02},$ and $B_{12}$.
$$
A=\left(\begin{array}{lll}
M_{0000} & M_{0011} & M_{0022} \\
M_{1100} & M_{1111} & M_{1122} \\
M_{2200} & M_{2211} & M_{2222}
\end{array}\right).
$$
$$
{B}_{i j}=\left(\begin{array}{ll}
M_{i j i j} & M_{i j j i} \\
M_{j i i j} & M_{j i j i}
\end{array}\right).
$$

For our constitutive model, we compute
$$\frac{\partial^2 \hat{\psi}}{\partial \sigma_i^2} = \frac{1}{\sigma_i} \left(2\mu \frac{1}{\sigma_i} + \lambda \frac{1}{\sigma_i}\right) - \frac{1}{\sigma_i^2}\left[ 2\mu\log\sigma_i + \lambda (\log \sigma_0 + \log \sigma_1 + \log \sigma_2)\right]=\frac{1}{\sigma_i^2}\left[ (2\mu+\lambda) (1-\log \sigma_i) - \lambda (\log \sigma_{j} + \log\sigma_k)\right],$$ 
where $i\neq j \neq k,$ and
$$
\frac{\partial^2 \hat{\psi}}{\partial \sigma_i \partial \sigma_j} = \frac{\lambda}{\sigma_i \sigma_j}.
$$
In computing $\frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j},$ we need to pay special attention when the two singular values are close to each other. Using the results above, we can write
$$
\begin{aligned}
\frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j} &= \frac{\frac{1}{\sigma_i}(2\mu\log\sigma_i + \lambda C) - \frac{1}{\sigma_j}(2\mu\log\sigma_j + \lambda C)}{\sigma_i - \sigma_j}\\
&= \frac{1}{\sigma_i - \sigma_j}\lambda C \left(\frac{1}{\sigma_i}-\frac{1}{\sigma_j}\right) + \frac{2\mu}{\sigma_i - \sigma_j} \left(\frac{1}{\sigma_i}\log\sigma_i - \frac{1}{\sigma_j}\log\sigma_j \right)\\
&= \frac{1}{\sigma_i - \sigma_j}\lambda C \frac{\sigma_j-\sigma_i}{\sigma_i\sigma_j}+ \frac{2\mu}{\sigma_i - \sigma_j} \left({\sigma_j}\log\sigma_i - {\sigma_i}\log\sigma_j \right)\frac{1}{\sigma_i \sigma_j}\\
&= - \frac{\lambda C}{\sigma_i \sigma_j} - \frac{2\mu}{\sigma_i \sigma_j}\frac{{\sigma_i}\log\sigma_j - {\sigma_j}\log\sigma_i}{\sigma_i - \sigma_j}\\
&= -\frac{1}{\sigma_i \sigma_j} (\lambda C + 2\mu \operatorname{CalcXLogYMinusYLogXOverXMinusY}(\sigma_i, \sigma_j)).
\end{aligned}
$$
Here $C = \log \sigma_0+\log \sigma_1+\log \sigma_2.$

In implementation we write `ij=3*j+i` and `rs=3*s+r`, so that the fourth-order tensor is reshaped to a 9by9 matrix.









