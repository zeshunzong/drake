## Elastic energy and its derivatives

We are given existing particles with positions and deformation gradients, denoted as 
$$x_p^0, F_p^0$$
respectively.

Now given grid velocities/positions 
$$x_i^*, v_i^*,$$
we compute the elastic energy and its first and second order derivatives.

To compute the elastic energy we need the new particle deformation gradient, which is  
$$F_p(x^*) = [I + \sum_i (x_i^*-x_i^0) \nabla N_i(x_p^0)^T]F^0_p.$$

Total energy is thus given by 
$$e(x^*) = \sum_p V_p^0 \psi(F_p(x^*)).$$

Differentiating energy w.r.t grid position,
$$\left[\frac{\partial e(x^*)}{\partial x_i^*}\right]_\alpha = \left[\sum_p V_p^0 \frac{\partial \psi}{\partial F}(F_p(x^*))(F^0_p)^T\nabla N_i(x_p^0)\right]_\alpha, \alpha=1,2,3.$$

Grid force is 
$$f_i(x^*) = -\frac{\partial e(x^*)}{\partial x_i^*}.$$

The second order derivative (hessian) of energy w.r.t grid positions is 
$$H(x^*)(i\alpha, j\rho) = H(3i+\alpha, 3j+\rho) = \frac{\partial^2 e}{\partial x^*_{j\rho} \partial x^*_{i\alpha}} = \sum_p V_p^0 \sum_{\beta, \gamma } \left\{[\nabla N_i(x_p^0)]_\beta [\nabla N_j(x_p^0)]_\gamma \sum_{\theta, \phi}\left\{[\frac{\partial^2 \psi}{\partial F^2}(F_p(x^*))]_{\alpha\theta, \rho \phi} [F^0_p]_{\beta \theta} [F^0_p]_{\gamma \phi}\right\}\right\}.$$ 

We can also compute the product of the hessian $H(x^*)(i\alpha, j\rho)$ with an arbitrary vector $z$, which is of length 3 * number of nodes. The result is 
$$y_i = (Hz)[3i:3i+3] = \sum_p V_p^0 A_p {(F^0_p)}^T \nabla N_i(x_p^0),$$
where
$$A_p = \sum_{\tau,\sigma} [\frac{\partial^2 \psi}{\partial F^2}(F_p(x^*))]_{\alpha\beta\tau \sigma} [B_p]_{\tau \sigma},$$
$$B_p = \sum_j z[3j:3j+3] \nabla N_j(x_p^0)^T F_p^0.$$

Note: in code, a fourth-order tensor is stored as a 9by9 matrix $A$ such that $A_{\alpha\beta\tau \sigma} = A[3\beta+\alpha, 3\sigma + \tau].$
