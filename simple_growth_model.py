import numpy as np
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

def exact_solution(alpha, beta):
    c1 = np.log(1 - alpha * beta)/(1 - beta) + alpha * beta * np.log(alpha * beta)/((1 - alpha*beta)*(1 - beta))
    c2 = alpha/(1 - alpha * beta)
    print(c1, c2)
    return lambda k: c1 + c2*np.log(k)

pars = dict(alpha = 0.65, beta= 0.95)
x = np.linspace(10e-6, 2, 1000)
y = exact_solution(**pars)

z = [y(k) for k in x]

#plot the exact solution
#fig, ax = plt.subplots(1,1)
#ax.plot(x, z, 'k-', lw=2)
#ax.set_title('value function')
  


class GrowthModel():
    def __init__(self, f, u, beta):
        self.f = f
        self.u = u
        self.beta = beta
    def vdirect(self, c, k, w):
        u, f, beta = self.u, self.f, self.beta
        return u(c) + beta * w(f(k) - c)
    def c(self, k, w):
        f2max = lambda c: self.vdirect(c, k, w)
        return fminbound(lambda c: -f2max(c), 1e-6, self.f(k))
    def vindirect(self, k, w):
        cstar = self.c(k, w)
        return self.vdirect(cstar, k, w)
    def get_bellman_operator(self, grid):
        def T(wvals):
            w = lambda k: np.interp(k, grid, wvals)
            Tw = np.empty_like(wvals)
            for i, k in enumerate(grid):
                Tw[i] = self.vindirect(k, w)
            return Tw
        return T


gm = GrowthModel(lambda k: k**pars['alpha'], np.log, pars['beta'])
vvals = [gm.vindirect(k, y) for k in x]
#fig, ax = plt.subplots(1,1)
#ax.plot(x, vvals, 'k-', lw = 3)
#fig.show()

v_star = exact_solution(**pars)
#def plot_value_iterations(n, gm, grid, wvals):
#    bellman_operator = gm.get_bellman_operator(grid)
#    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (6,6))
#    ax1.plot(grid, wvals, 'k--', lw = 1, alpha = 0.6, label = 'initial condition')
#    for i in range(n):
#        wvals = bellman_operator(wvals)
#        ax1.plot(grid, wvals, color = plt.cm.jet(i/n), lw = 1, alpha = 0.6)
#    for i in range(n):
#        wvals = bellman_operator(wvals)
#        ax2.plot(grid, wvals, color = plt.cm.jet(i/n), lw = 1, alpha = 0.6)
#    ax1.plot(grid, v_star(grid), 'k--', lw = 2, alpha = 0.8, label = 'true value function')
#    ax2.plot(grid, v_star(grid), 'k--', lw = 2, alpha = 0.8, label = 'true value function')
#    ax1.set_ylim(-40, -20)
#    ax2.set_ylim(-40, -30)
#    ax1.set_xlim(np.min(grid), np.max(grid))
#    ax1.legend(loc = 'upper left')
#    ax1.set_title('Value function iteration : first {} iterations'.format(n))
#    ax2.set_title('Value function iteration : first {} iterations'.format(n))
    
#if __name__ == '__main__':
#    winit = np.zeros_like(x)
#    plot_value_iterations(50, gm, x, winit)
    
def fixedptld(f, v, itermax = 200, tol = 1e-3):
    ct, error = 0, 1+tol
    while ct < itermax and error > tol:
        ct += 1
        vnext = f(v)
        error = np.max(np.abs(vnext - v))
        v = vnext
    if ct == itermax:
        print ('covergence failed in {} iterations'.format(ct))
    else: print ('covergence in {} interations'.format(ct))
    return v
        
def plot_value_approximation(gm, grid, wvals):
    bellman_operator = gm.get_bellman_operator(grid)
    fig, ax = plt.subplots(1, 1)
    ax.plot(grid, wvals, 'k--', lw = 1, alpha = 0.6, label = 'initial condition')
    ax.plot(grid, v_star(grid), 'r-', lw = 2, alpha = 0.8, label = 'true value function')
    v = fixedptld(bellman_operator, wvals)
    ax.plot(grid, v, 'b-', lw = 1, alpha = 0.8, label = 'appriciate value function')
    ax.set_ylim(-40, -20)
    ax.set_title('Value function iteration: iterate to convegence')
    ax.legend(loc = 'upper left')
    
#if __name__ == '__main__':
#    winit = np.zeros_like(x)
##    winit = 5 * np.log(x) - 25
#    plot_value_approximation(gm, x, winit)
        
alpha, beta = pars['alpha'], pars['beta']
true_sigma = (1 - alpha * beta)* x**alpha
winit = 5 * np.log(x) - 25
bellman_operator = gm.get_bellman_operator(x)
fig, ax = plt.subplots(1, 1)

for niter in (2, 4, 6):
    vvals_approx = fixedptld(bellman_operator, winit, itermax = niter)
    vinterp = lambda k: np.interp(k, x, vvals_approx)
    sigma = [gm.c(k, vinterp) for k in x]
    ax.plot(x, sigma, color = plt.cm.jet(niter/5), lw = 2, alpha = 0.8, label = '{} iterations'.format(niter))#'{} iterations'.format(niter))

ax.plot(x, true_sigma, 'k-', lw = 2, alpha = 0.8, label = 'true optimal policy')
ax.legend(loc = 'upper left')
ax.set_title('policy from value function')
ax.set_ylim(0, 1)
ax.set_xlim(0, 2)
#ax.set_yticks((0, 1))
#ax.set_xticks((0, 2))
#plt.show()

    