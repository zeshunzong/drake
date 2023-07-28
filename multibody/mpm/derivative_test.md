### Derivative test

Our goal is to test whether the implementation of 
$$f=\frac{\partial e}{\partial x^*}$$
and 
$$h=\frac{\partial^2 e}{\partial (x^*)^2}$$
is correct.

2D simple grid.

              o------------------o
              |                  |
              |    p             |
              |                  |
              |                  |
              |                  |
              |                  |
              o------------------o


Randomly put particles inside (I think one particle can do the job).
Initialized a fixed F_p0, x_p0, V0

We have x_i0, the initial positions of grid nodes (say 00, 01, 10, 11).
Generate x_i*, use them as the independent variables.

The particle deformation gradient as a function of x* is 
$$F_p(x*) = [I + \sum_i (x_i^*-x_i^0) \nabla N_i(x_p^0)^T]F^0_p$$
This is a function of only the independent variables x_i*, everything else are fixed numbers.

Total energy is given by 
$$e(x^*) = \sum_p V_0 \psi(F_p(x^*))$$

Grid force is given by 
$$f_i(x^*) = \frac{e(x^*)}{x_i^*} = \sum_p V_0 \frac{\partial \psi}{\partial F}(F_p(x^*))(F^n_p)^T\nabla N_i(x_p^0)$$

The second order derivative is 
$$\frac{\partial^2 e}{\partial (x^*)^2}$$ 
some long formula omitted here.

The two above derivatives are computed by hand. And we can test if they are implemented correctly using AutoDiff.


That is, we can test, stand alone, whether the three expressions are correct.

However, note that the grid force is not what is computed in actual MPM. In MPM the grid force is computed via Kirchhoff stress, using only the F_p updated from previous step.

$$\frac{\partial \psi}{\partial F}(F^n_p)(F^n_p)^T$$