`ClampToEpsilon(x, eps):`
$$
r=\left\{\begin{array}{cl}
\operatorname{sgn}(x) \cdot \varepsilon, & \text { if } x \in(-\varepsilon, \varepsilon) \\
x, & \text { ow. }
\end{array}\right.
$$
`CalcLogXPlus1OverX(x, eps):`
$$
\begin{aligned}
r=\frac{\log (x+1)}{x} & =\frac{x-\frac{x^2}{2}+\frac{x^3}{3}-\frac{x^4}{4}+\cdots}{x} \\
& =1-\frac{x}{2}+\frac{x^2}{3}-\frac{x^3}{4}+\cdots
\end{aligned}
$$
`CalcLogXMinusLogYOverXMinusY(x, y, eps):`
$$
\begin{aligned}
r=\frac{\log x-\log y}{x-y} & =\frac{1}{y} \frac{\log \frac{x}{y}}{\frac{x}{y}-1} \\
& =\frac{1}{y} \frac{\log \left[\left(\frac{x}{y}-1\right)+1\right]}{\left(\frac{x}{y}-1\right)}
\end{aligned}
$$
`CalcXLogYMinusYLogXOverXMinusY(x, y, eps):`
$$
\begin{aligned}
r & =\frac{x \log y-y \log x}{x-y} \\
& =\frac{(x-y) \log y-y(\log x-\log y)}{x-y} \\
& =\log y-y \cdot \frac{\log x-\log y}{x-y}
\end{aligned}
$$
`CalcExpXMinus1OverX(x, eps):`
$$
\begin{aligned}
r=\frac{e^x-1}{x} & =\frac{-1+1+x+\frac{x^2}{2}+\frac{x^3}{6}+\frac{x^4}{24}+\cdots}{x} \\
& =1+\frac{x}{2}+\frac{x^2}{6}+\frac{x^3}{24}+\cdots
\end{aligned}
$$
`CalcExpXMinusExpYOverXMinusY(x, y, eps):`
$$
r=\frac{e^x-e^y}{x-y}=e^y \frac{e^{(x-y)}-1}{(x-y)}
$$
