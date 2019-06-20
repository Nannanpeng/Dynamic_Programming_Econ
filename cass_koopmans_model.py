import numpy as np
import matplotlib.pyplot as plt

class UtilityCapital():
    def __init__(self, gamma, alpha, A, beta):
        self.gamma = gamma
        self.alpha = alpha
        self.A = A
        self.beta = beta
    def u(self, c):
        if self.gamma == 1: return np.log(c)
        else: return c**(1 - self.gamma) / (1 - self.gamma)
    def u_prime(self, c):
        if self.gamma == 1: return 1/c
        else: return c**(-self.gamma)
    def u_prime_inv(self, c):
        if self.gamma == 1: return c
        else: return c**(-1/self.gamma)
    def f(self, k): 
        return self.A * k**self.alpha
    def f_prime(self, k): 
        return self.alpha * self.A * k**(self.alpha - 1)
    def f_prime_inv(self, k): 
        return (k / (self.A * self.alpha))**(1 / (self.alpha - 1))

class CassKoopmans():
    def __init__(self, uc, delta):
        self.uc = uc
        self.delta = delta
    def shooting_method(self, c, k):
        uc, delta = self.uc, self.delta
        T = len(c) - 1
        for t in range(T):
            k[t+1] = uc.f(k[t]) + (1 - delta) * k[t] - c[t]
            if k[t+1] < 0: 
                k[t+1] = 0
            if uc.beta * (uc.f_prime(k[t+1]) + (1 - delta)) == np.inf:
                c[t+1] = 0
            else: c[t+1] = uc.u_prime_inv(uc.u_prime(c[t]) / (uc.beta * (uc.f_prime(k[t+1]) + (1 - delta))))
        k[T+1] = uc.f(k[T]) + (1 - delta) * k[T] - c[T]
        return c, k
    def ConsumptionCapital(self, c, k):
        paths = self.shooting_method(c, k)
        fig, axes = plt.subplots(1, 2, figsize = (10, 4))
        colors = ['blue', 'red']
        titles = ['Consumption', 'Capital']
        ylabels = ['$c_t$', '$k_t$']  
        for path, color, title, y, ax in zip(paths, colors, titles, ylabels, axes):
            ax.plot(path, c = color, alpha = 0.7)
            ax.set(title = title, ylabel = y, xlabel = 't')
        ax.scatter(T+1, 0, s = 80)
        ax.axvline(T+1, color = 'k', ls = '--', lw = 1)
        plt.tight_layout()
        plt.show()
    def bisection_method(self, c, k, tol = 1e-4, max_iter = 1e4, terminal = 0):
        T = len(c) - 1
        i = 1
        c_high, c_low = self.uc.f(k[0]), 0
        path_c, path_k = self.shooting_method(c, k)
        while (np.abs(path_k[T+1] - terminal) > tol or path_k[T] == terminal) and i < max_iter:
            if path_k[T+1] - terminal > tol: 
                c_low = c[0]
            elif path_k[T+1] - terminal < -tol or path_k[T] == terminal:
                c_high = c[0]
            c[0] = (c_high + c_low) / 2
            path_c, path_k = self.shooting_method(c, k)
            i += 1
        if np.abs(path_k[T+1] - terminal) < tol and path_k[T] != terminal:
            print('Convergence succeeded in {} iterations'.format(i-1))
        else: print('Convergence failed and hit maximum iteration')
        u = self.uc.u_prime(path_c)
        return path_c, path_k, u
    def plot_paths(self, c, k, axes=None, ss=None):
        paths = self.bisection_method(c, k)
        T = len(paths[0])
        if axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        ylabels = ['$c_t$', '$k_t$', '$\mu_t$']
        titles = ['Consumption', 'Capital', 'Lagrange Multiplier']
        for path, y, title, ax in zip(paths, ylabels, titles, axes):
            ax.plot(path, label = "{} periods".format(T))
            ax.set(ylabel=y, title=title, xlabel='t')
        if ss is not None:
            axes[1].axhline(ss, c='k', ls='--', lw=1)
        axes[1].axvline(T, c='k', ls='--', lw=1)
        axes[1].scatter(T, paths[1][-1], s=80)
        plt.tight_layout()
    def plot_multiple_paths(self, T_list, k_init):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3))
        for T in T_list:
            c = np.zeros(T+1)
            k = np.zeros(T+2)
            c[0] = 0.3
            k[0] = k_init/3
            self.plot_paths(c, k, axes=axes, ss = k_init)
        
class MarketCassKoopmans():
    def __init__(self, uc, ck):
        self.uc = uc
        self.ck = ck
    def price(self, c):
        T = len(c) - 2
        q = np.zeros(T+1)
        q[0] = 1
        for t in range(1, T+1):
            q[t] = self.uc.beta**t * self.uc.u_prime(c[t])
        return q
    def wage(self, k):
        w = self.uc.f(k) - k * self.uc.f_prime(k)
        return w
    def capital_price(self, k):
        return self.uc.f_prime(k)
    def plot_paths(self, T_list, k_ss,c_ss):
        fix, axes = plt.subplots(2, 3, figsize=(13, 6))
        titles = ['Arrow-Hicks Prices', 'Labor Rental Rate', 'Capital Rental Rate',
                  'Consumption', 'Capital', 'Lagrange Multiplier']
        ylabels = ['$q_t^0$', '$w_t$', '$\eta_t$', '$c_t$', '$k_t$', '$\mu_t$']
        for T in T_list:
            c = np.zeros(T+1)
            k = np.zeros(T+2)
            c[0] = 0.3
            k[0] = k_ss / 3
            c, k, mu = self.ck.bisection_method(c, k)
            q, w, eta = self.price(c), self.wage(k)[:-1], self.capital_price(k)[:-1]
            plots = [q, w, eta, c, k, mu]
            for ax, plot, title, y in zip(axes.flatten(), plots, titles, ylabels):
                ax.plot(plot)
                ax.set(title=title, ylabel=y, xlabel='t')
                if title is 'Capital':
                    ax.axhline(k_ss, lw=1, ls='--', c='k')
                if title is 'Consumption':
                    ax.axhline(c_ss, lw=1, ls='--', c='k')
        plt.tight_layout()
        plt.show()    
    def plot_paths_gamma(self, gamma_list, T, k_ss, c_ss):
         fix, axes = plt.subplots(2, 3, figsize=(13, 6))
         titles = ['Arrow-Hicks Prices', 'Labor Rental Rate', 'Capital Rental Rate',
                  'Consumption', 'Capital', 'Lagrange Multiplier']
         ylabels = ['$q_t^0$', '$w_t$', '$\eta_t$', '$c_t$', '$k_t$', '$\mu_t$']
         for gamma in gamma_list:
             self.uc.gamma = gamma
             c = np.zeros(T+1)
             k = np.zeros(T+2)
             c[0] = 0.3
             k[0] = k_ss / 3
             c, k, mu = self.ck.bisection_method(c, k)
             q, w, eta = self.price(c), self.wage(k)[:-1], self.capital_price(k)[:-1]
             plots = [q, w, eta, c, k, mu]
             for ax, plot, title, y in zip(axes.flatten(), plots, titles, ylabels):
                 ax.plot(plot, label='$\gamma = {}$'.format(gamma))
                 ax.set(title=title, ylabel=y, xlabel='t')
                 if title is 'Capital':
                    ax.axhline(k_ss, lw=1, ls='--', c='k')
                 if title is 'Consumption':
                    ax.axhline(c_ss, lw=1, ls='--', c='k')
         axes[0, 0].legend()
         plt.tight_layout()
         plt.show()
         
parms = dict(gamma = 2, delta = 0.02, beta = 0.95, alpha = 0.33, A = 1)
T = 150
c, k = np.zeros(T+1), np.zeros(T+2) 

c[0] = 0.3
uc = UtilityCapital(parms['gamma'], parms['alpha'], parms['A'], parms['beta'])
ck = CassKoopmans(uc, parms['delta'])

rho = 1 / parms['beta'] - 1
k_ss = uc.f_prime_inv(rho + parms['delta']) 

k[0] = k_ss
#ck.ConsumptionCapital(c,k)
#ck.plot_paths(c, k)   
#ck.plot_paths(c,k)    
#ck.plot_multiple_paths((150, 75, 50, 25), k_ss)  
         
S_ss = parms['delta'] * k_ss
c_ss = uc.f(k_ss) - S_ss
s_ss = S_ss / uc.f(k_ss)       
T_list = (250, 150, 75, 50)      
mck = MarketCassKoopmans(uc, ck)
#mck.plot_paths(T_list, k_ss, c_ss)     
        
gamma_list = (1.1, 4, 6, 8)
T = 150
#mck.plot_paths_gamma(gamma_list, T, k_ss, c_ss)      
        
        
        






