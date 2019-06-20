import numpy as np
import math
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

def exact_solution(alpha,beta):
    """Return function, the exact solution."""
    ab = alpha * beta
    c1 = (math.log(1 - ab) + math.log(ab) * ab / (1 - ab)) / (1 - beta)
    c2 = alpha / (1 - ab)
    return lambda k: c1 + c2 * np.log(k)


prms01 = dict(alpha = 0.65,beta=0.95) 
kgrid01 = np.linspace(1e-6, 2, 150) 
v_star = exact_solution(**prms01)
fig, ax = plt.subplots(1,1)
ax.plot(kgrid01, v_star(kgrid01), 'k-', lw=2)
ax.set_title('Value Function')
fig.show()


class GrowthModel01(object):
    def __init__(self, f, u, beta):
        self.f = f #function of one variable, net prodn
        self.u = u #function of one variable, utility
        self.beta = beta #float in (0,1), the discount rate
    def vdirect(self, c, k, w):
        u, f, beta = self.u, self.f, self.beta
        return u(c) + beta * w(f(k)-c)
    def ca(self, k, w): # w-greedy consumption policy
        f2max = lambda c: self.vdirect(c, k, w)
        return fminbound(lambda c: -f2max(c), 1e-6, self.f(k))
    def vindirect(self, k, w):
        cstar = self.ca(k, w)
        return self.vdirect(cstar, k, w)
    

   
gm01 = GrowthModel01(lambda k: k**prms01['alpha'], np.log, prms01['beta'])
#vvals = [gm01.vindirect(k, v_star) for k in kgrid01]
##plot the exact solution
#fig, ax = plt.subplots(1,1)
#ax.plot(kgrid01, v_star(kgrid01), 'k-', lw=2)
#ax.plot(kgrid01, vvals, 'r-', lw=1)
#ax.set_title('Value Function: Exact vs. Computed')
#fig.show()
   
     
    
def get_bellman_operator(gm, grid):
    """Return function, the approximate Bellman operator."""
    def T(wvals):
        """Return array, the updated values on the grid,
        given ‘wvals‘, the initial values on the grid."""
        w = lambda k: np.interp(k, grid, wvals) #SS’s Aw
        Tw = np.empty_like(wvals)
# set Tw[i] = max_c { u(c) + beta vnext(f(k_i) - c)}
        for i, k in enumerate(grid):
            Tw[i] = gm.vindirect(k, w) #pleasingly parallel
        return Tw
    return T

"""
Illustrate a basic numerical solution of the optimal growth problem
via value function iteration. (For more detail, see SS’s optgrowth.py.)
"""
def plot_value_iterations(n, gm, grid, wvals):
    bellman_operator = get_bellman_operator(gm, grid)
    fig, (ax1,ax2) = plt.subplots(2,1,sharex=True, figsize=(6,6))
    ax1.plot(grid, wvals, 'k--', lw=1, alpha=0.6, label='initial condition')
    for i in range(n):
        wvals = bellman_operator(wvals)
        ax1.plot(grid, wvals, color=plt.cm.jet(i / n), lw=1, alpha=0.6)
    for i in range(n):
        wvals = bellman_operator(wvals)
        ax2.plot(grid, wvals, color=plt.cm.jet(i / n), lw=1, alpha=0.6)
    ax1.plot(grid, v_star(grid), 'k-', lw=2, alpha=0.8, label='true value function')
    ax2.plot(grid, v_star(grid), 'k-', lw=2, alpha=0.8, label='true value function')
    ax1.set_ylim(-40,-20)
    ax2.set_ylim(-40,-30)
    ax1.set_xlim(np.min(grid), np.max(grid))
    ax1.legend(loc='upper left')
    ax1.set_title('Value Function Iteration: first {} iterations'.format(n))
    ax2.set_title('Value Function Iteration: next {} iterations'.format(n))
    
if __name__ == '__main__':
#    winit = 5 * np.log(kgrid01) - 25 #Take informed initial condition from SS
    winit = np.zeros_like(kgrid01)
    plot_value_iterations(100, gm01, kgrid01, winit)