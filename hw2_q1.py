from random import seed, uniform, normalvariate, betavariate, randint
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from dp_generator import DirichletProcessGenerator_StickBreaking

seed(20230503)

# generate data
z_20 = [normalvariate(0, 1) for _ in range(20)]
z_200 = [normalvariate(0, 1) for _ in range(200)]
z_2000 = [normalvariate(0, 1) for _ in range(2000)]

def y_data_simulator(n_num_data):
    y_data_list = []
    for _ in range(n_num_data):
        u = uniform(0,1)
        if u<0.5:
            y_data_list.append(normalvariate(-2.5, 0.5))
        elif u<0.8:
            y_data_list.append(normalvariate(0.5, 0.7))
        else:
            y_data_list.append(normalvariate(1.5, 2))
    return y_data_list

y_20 = y_data_simulator(20)
y_200 = y_data_simulator(200)
y_2000 = y_data_simulator(2000)


# fit

class Post_DP(DirichletProcessGenerator_StickBreaking):
    def __init__(self, data, set_seed) -> None:
        self.y = data
        self.n = len(data)
        seed(set_seed)
        self.atom_loc = None
        self.atom_weight = None

    
    def atom_sampler(self, prior_precision:float, prior_base_measure_sampler, num_atom: int):
        #location
        atom_loc = []
        for _ in range(num_atom):
            u = uniform(0,1)
            if u < prior_precision / (prior_precision + self.n):
                atom_loc.append(prior_base_measure_sampler())
            else:
                atom_loc.append(self.y[randint(0, self.n-1)])
        
        left_stick_length = 1.0
        atom_weight = []
        for _ in range(num_atom-1):
            portion = betavariate(1, prior_precision+self.n)
            weight = portion * left_stick_length
            atom_weight.append(weight)
            left_stick_length = left_stick_length * (1 - portion)
        atom_weight.append(left_stick_length)
        return atom_loc, atom_weight

if __name__=="__main__":
    
    std_norm_sampler = partial(sp.stats.norm.rvs, loc=0, scale=1)
    std_norm_cdf = partial(sp.stats.norm.cdf, loc=0, scale=1)
    def mixture_cdf(x):
        return 0.5*sp.stats.norm.cdf(x, loc=-2.5, scale=0.5) + 0.3*sp.stats.norm.cdf(x, loc=0.5, scale=0.7) + 0.2*sp.stats.norm.cdf(x, loc=1.5, scale=2)

    #prior settings
    prior_G0 = [partial(sp.stats.norm.rvs, loc=0, scale=1),
                partial(sp.stats.norm.rvs, loc=2, scale=1),
                partial(sp.stats.norm.rvs, loc=0, scale=0.1),
                partial(sp.stats.norm.rvs, loc=0, scale=2)]
    prior_G0_str = [r"N(0,1)",r"N(2,1)",r"N(0,$0.1^2$)",r"N(0,$2^2$)"]
    prior_alpha = [1, 10, 100, 1000]


    z_test = False
    y_test = True

    if z_test:
        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(z_20, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 3000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), std_norm_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=20, "+ r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str, fontsize=12)
        plt.show()

        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(z_200, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 3000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), std_norm_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=200, "+r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str)
        plt.show()

        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(z_2000, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 3000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), std_norm_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=2000, "+r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str, fontsize=12)
        plt.show()

    if y_test:
        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(y_20, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 4000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), mixture_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=20, "+r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str, fontsize=12)
        plt.show()

        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(y_200, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 4000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), mixture_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=200, "+r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str, fontsize=12)
        plt.show()

        fig, axs = plt.subplots(len(prior_G0), len(prior_alpha), figsize=(len(prior_alpha)*4, len(prior_G0)*4))
        fig.tight_layout(pad=3, w_pad=3, h_pad=3)
        for g_idx, (g0, g0_str) in enumerate(zip(prior_G0, prior_G0_str)):
            for a_idx, alpha in enumerate(prior_alpha):
                for i in range(20):
                    fit_inst = Post_DP(y_2000, 20230503+i)
                    atom_loc, atom_weight = fit_inst.atom_sampler(alpha, g0, 9000)
                    grid, increments, sample_path = fit_inst.cumulatative_dist_func(atom_loc, atom_weight, -10, 10)
                    axs[g_idx, a_idx].step(grid, sample_path, where='post', c='blue', alpha=0.15)
                axs[g_idx, a_idx].plot(np.linspace(-10, 10, 200), mixture_cdf(np.linspace(-10, 10, 200)), c='red', label=r'true')
                axs[g_idx, a_idx].legend()
                axs[g_idx, a_idx].set_title("n=2000, "+r'$\alpha$='+str(alpha)+r', $G_0$='+g0_str, fontsize=12)
        plt.show()

