Below we denote $F$ as the deformation gradient AFTER return mapping.

Write 
$$F = U \Sigma V^T,$$
with $\Sigma = \operatorname{diag}(\sigma_1, \sigma_2, \sigma_3).$

Define the energy density function $\psi$ to be
$$
\begin{aligned}
\psi(F) &= \psi(\sigma_1, \sigma_2, \sigma_3) \\
& =\mu\left[\left(\log \sigma_1\right)^2+\left(\log \sigma_2\right)^2+\left(\log \sigma_3\right)^2\right] +\frac{1}{2} \lambda\left[\left(\log \sigma_1\right)+\left(\log \sigma_2\right)+\left(\log \sigma_3\right)\right]^2.
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
\left(\frac{\partial {P}}{\partial {F}}({F})\right)_{i j r s}=\left(\frac{\partial P}{\partial F}(\Sigma)\right)_{k l m n} {U}_{i k} {U}_{r m} {V}_{s n} {V}_{j l}.
$$
Denote 
$$M_{klmn} = \left(\frac{\partial P}{\partial F}(\Sigma)\right)_{k l m n}.$$
This 9by9 or 3by3by3by3 matrix has 21 non-zero entries. See [http://alexey.stomakhin.com/research/sca2012_tech_report.pdf] for details. The results are the following
$$
M_{klmn}=\left\{\begin{array}{cl}
\frac{\partial^2 \hat{\psi}}{\partial \sigma_k \partial \sigma_m}, & \text { if } k=l, m=n, \text{ i.e. } M_{1111,1122,1133,2211,2222,2233,3311,3322,3333} \\
M_{ijij}, & \text { if }k=m=i, l=n=j, i\neq j, \text{ i.e. } M_{1212,1313,2121,2323,3131,3232}\\
M_{ijji}, & \text { if }k=n=i, l=m=j, i\neq j, \text{ i.e. } M_{1221,1331,2112,2332,3113,3223}\\
0, & \text{ ow,}
\end{array}\right.
$$
where
$$
M_{ijij} = \frac{\sigma_i \frac{\partial \hat{\psi}}{\partial \sigma_i} - \sigma_j \frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i^2 - \sigma_j^2} = \frac{1}{2}\cdot \left( \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j} +  \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}+\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i + \sigma_j}\right),\\
M_{ijji} = \frac{\sigma_j \frac{\partial \hat{\psi}}{\partial \sigma_i} - \sigma_i \frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i^2 - \sigma_j^2}= \frac{1}{2}\cdot \left( \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}-\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i - \sigma_j} -  \frac{\frac{\partial \hat{\psi}}{\partial \sigma_i}+\frac{\partial \hat{\psi}}{\partial \sigma_j}}{\sigma_i + \sigma_j}\right).
$$
For our constitutive model, we compute
$$\frac{\partial^2 \hat{\psi}}{\partial \sigma_i^2} = \frac{1}{\sigma_i} \left(2\mu \frac{1}{\sigma_i} + \lambda \frac{1}{\sigma_i}\right) - \frac{1}{\sigma_i^2}\left[ 2\mu\log\sigma_i + \lambda (\log \sigma_1 + \log \sigma_2 + \log \sigma_3)\right]=\frac{1}{\sigma_i^2}\left[ (2\mu+\lambda) (1-\log \sigma_i) - \lambda (\log \sigma_{j} + \log\sigma_k)\right],$$
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
In implementation we write `ij=3*j+i` and `rs=3*s+r`.