## Tolerances for Newton and CG.

We want, for newton to terminate,
$$||f'(x_0 + \Delta x)|| < \epsilon.$$
So 
$$||f'(x_0) + f''(x_0) \Delta x + O(\Delta x^2)|| < \epsilon.$$

CG solves $\Delta x$ as $f''(x_0) \Delta x \approx -f'(x_0),$ and we set the relative tolerance $\tau$ as  
$$\frac{||f''(x_0) \Delta x + f'(x_0)||}{||f'(x_0)||} < \tau.$$

So we need
$$\frac{||f'(x_0) + f''(x_0) \Delta x + O(\Delta x^2)||}{||f'(x_0)||} < \frac{||f'(x_0) + f''(x_0) \Delta x || + || O(\Delta x^2)||}{||f'(x_0)||} = \tau + \frac{||O(\Delta x^2)||}{||f'(x_0)||}< \frac{\epsilon}{||f'(x_0)||}.$$