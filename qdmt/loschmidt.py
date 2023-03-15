import numpy as np
from scipy.integrate import quad

import matplotlib.pyplot as plt

def f(z, g0, g1):
    def theta(k, g):
        return np.arctan2(np.sin(k), g-np.cos(k))/2
    def phi(k, g0, g1):
        return theta(k, g0)-theta(k, g1)
    def epsilon(k, g1):
        return -2*np.sqrt((g1-np.cos(k))**2+np.sin(k)**2)
    def integrand(k):
        return -1/(2*np.pi)*np.log(np.cos(phi(k, g0, g1))**2 + np.sin(phi(k, g0, g1))**2 * np.exp(-2*z*epsilon(k, g1)))

    return quad(integrand, 0, np.pi)[0]


def loschmidt_paper(t, g0, g1):
    return (f(t*1j, g0, g1)+f(-1j*t, g0, g1))

def loschmidts(T, g0, g1):
    return np.array([loschmidt_paper(t, g0, g1) for t in T])

def time_fisher_zero(t_star, n):
    return t_star * (n + 0.5)

def times_fisher_zero(g0, g1, t_max):
    def k_star(g0, g1):
        return np.arccos((1 + g0*g1)/(g0+g1))    # plt.plot(time_steps, measured_loschmidt, label = 'Measured' )


    def epsilon(k, g1):
        return 2*np.sqrt((g1-np.cos(k))**2+np.sin(k)**2)

    def t_star(g0, g1):
        return np.pi / epsilon(k_star(g0, g1), g1)

    print('k_star: ', k_star(g0, g1))
    t_star = t_star(g0, g1)
    print('t_star: ', k_star(g0, g1))

    n_max = np.floor(t_max / t_star - 0.5)
    n_max = int(n_max)

    return [time_fisher_zero(t_star, n) for n in range(0, n_max + 1)]


if __name__=="__main__":
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 20
    max_time = 1.8
    g0, g1 = 1.5, 0.2
    ltimes = np.linspace(0.0, max_time, 100)
    correct_ls = [loschmidt_paper(t, g0, g1) for t in ltimes]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(ltimes, correct_ls, '-')

    ax.set_ylabel(r'$\lambda(t)$')
    ax.set_xlabel(r'$t$')

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylim(0.0, 0.6)
    ax.set_xlim(0.0, 1.8)

    ax.grid(True)
    plt.show()
