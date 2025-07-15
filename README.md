# Project: Pattern Formation

This repository is part of a the masters course Simulation and Modeling 2 at the University of Cologne.

**Authors**: Yeganeh, Vincent, and Gaurav

## Reaction Diffusion System
$$ \begin{aligned} \frac{\partial u}{\partial t} &= D_u \nabla^2 u + f(u, v) \\\\ \frac{\partial v}{\partial t} &= D_v \nabla^2 v + g(u, v) \end{aligned} $$

Where:
- $u(t, x, y)$ and $v(t, x, y)$ are the concentrations of two substances,
- $D_u$, $D_v$ are constant diffusion coefficients,
- $f(u, v)$ and $g(u, v)$ describe the reaction kinetics,
- The domain is $\Omega = [0, 1]^2$ with periodic boundary conditions


#### Gray-Scott Model
$$ \begin{aligned} f (u, v) &= −uv^2 + \alpha (1 − u) \\\\ g(u, v) &= uv^2 − (\alpha + \beta)v \end{aligned} $$

#### Solvers
- second order finite difference approximation / appropriate explicit discretization
- Crank – Nicolson finite difference approximation

## References
[1] A. M. Turing, “The chemical basis of morphogenesis,” Philosophical Transactions of the Royal
Society of London. Series B, Biological Sciences, vol. 237, no. 641, pp. 37–72, 1952.

## Presentation
https://www.canva.com/design/DAGp93yjrVA/YgR_wmOwD-Jna24BWWbEtA/edit?utm_content=DAGp93yjrVA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
