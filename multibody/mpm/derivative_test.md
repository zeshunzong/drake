## Elastic energy and its derivatives

We are given existing particles with positions and deformation gradients, denoted as 
$$x_p^0, F_p^0$$
respectively.

Now given grid velocities/positions 
$$x_i^*, v_i^*,$$
new particle deformation gradient is  
$$F_p(x^*) = [I + \sum_i (x_i^*-x_i^0) \nabla N_i(x_p^0)^T]F^0_p.$$

Total energy is given by 
$$e(x^*) = \sum_p V_p^0 \psi(F_p(x^*)).$$

Differentiating energy w.r.t grid position,
$$\left[\frac{e(x^*)}{x_i^*}\right]_\alpha = \left[\sum_p V_p^0 \frac{\partial \psi}{\partial F}(F_p(x^*))(F^n_p)^T\nabla N_i(x_p^0)\right]_\alpha, \alpha=1,2,3.$$

Grid force is 
$$f_i(x^*) = -\frac{e(x^*)}{x_i^*}.$$

The second order derivative (hessian) of energy w.r.t grid positions is 
$$H(x^*)(i\alpha, j\rho) = H(3i+\alpha, 3j+\rho) = \frac{\partial^2 e}{\partial x^*_{j\rho} \partial x^*_{i\alpha}} = \sum_p V_p^0 \sum_{\beta, \gamma } \left\{[\nabla N_i(x_p^0)]_\beta [\nabla N_j(x_p^0)]_\gamma \sum_{\theta, \phi}\left\{[\frac{\partial^2 \psi}{\partial F^2}]^p_{\alpha\theta, \rho \phi} [F^0_p]_{\beta \theta} [F^0_p]_{\gamma \phi}\right\}\right\}.$$ 
