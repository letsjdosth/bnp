from math import inf, log
from random import seed, uniform, betavariate, randint

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyBayes.MCMC_Core import MCMC_MH, MCMC_Diag, MCMC_base
from pyBayes.util_MCMC_proposal import unif_proposal_log_pdf, unif_proposal_sampler
from dp_generator import DirichletProcessGenerator_StickBreaking

seed(20230503)
np.random.seed(20230513)

#generate data
pois5_300 = sp.stats.poisson.rvs(5,size=300)
pois_mixture_300 = []
for i in range(300):
    u = uniform(0, 1)
    if u<0.7:
        pois_mixture_300.append(sp.stats.poisson.rvs(3))
    else:
        pois_mixture_300.append(sp.stats.poisson.rvs(11))




class HW2_Q2_alpha_lambda_sampler(MCMC_base):
    def __init__(self, data: list, initial, hyper=None):
        #G~DP(alpha, G0=Pois(lambda))
        self.y_list = data
        self.n = len(data)
        self.y_dict = {}
        for y in self.y_list:
            try:
                self.y_dict[y] += 1
            except KeyError:
                self.y_dict[y] = 1
        self.n_star = len(self.y_dict.keys())

        self.MC_sample = [initial]
        
        if hyper==None:
            self.hyper_a_alpha = 1
            self.hyper_b_alpha = 1
            self.hyper_a_lambda = 1
            self.hyper_b_lambda = 1
        else:
            self.hyper_a_alpha = hyper["a_alpha"]
            self.hyper_b_alpha = hyper["b_alpha"]
            self.hyper_a_lambda = hyper["a_lambda"]
            self.hyper_b_lambda = hyper["b_lambda"]

    def _calculate_marginal_log_likelihood(self, alpha, lamb):
        log_val = self.n_star * log(alpha) - sum([log(alpha+i) for i in range(0, self.n)])
        for y_star_j, n_j in self.y_dict.items():
            pois_pmf_val = sp.stats.poisson.pmf(y_star_j, lamb)
            log_val += log(pois_pmf_val)
            for i in range(1, n_j):
                log_val += log(alpha*pois_pmf_val + i)
        return log_val
    
    

    def sampler(self, **kwargs):
        # alpha, lambda
        
        def _log_target(sample):
            alpha = sample[0]
            lamb = sample[1]
            log_val = self._calculate_marginal_log_likelihood(alpha, lamb)
            log_val += sp.stats.gamma.logpdf(alpha, self.hyper_a_alpha, 1/self.hyper_b_alpha)
            log_val += sp.stats.gamma.logpdf(lamb, self.hyper_a_lambda, 1/self.hyper_b_lambda)
            return log_val

        def _proposal_sampler(last):
            # alpha, lambda
            new_alpha = unif_proposal_sampler([last[0]], 0, inf, 2)
            new_lambda = unif_proposal_sampler([last[1]], 0, inf, 0.5)
            return [new_alpha[0], new_lambda[0]]

        def _proposal_log_density(from_smpl, to_smpl):
            # alpha, lambda
            log_val = unif_proposal_log_pdf([from_smpl[0]], [to_smpl[0]], 0, inf, 2)
            log_val += unif_proposal_log_pdf([from_smpl[1]], [to_smpl[1]], 0, inf, 0.5)
            return log_val
        
        initial = self.MC_sample[-1]
        MCMC_inst = MCMC_MH(_log_target, _proposal_log_density, _proposal_sampler, initial)
        MCMC_inst.generate_samples(3, verbose=False)
        new_a_lambda_sample = MCMC_inst.MC_sample[-1]
        self.MC_sample.append(new_a_lambda_sample)

class Post_DP_Q2(DirichletProcessGenerator_StickBreaking):
    def __init__(self, data, set_seed) -> None:
        self.y = data
        self.n = len(data)
        seed(set_seed)
        self.atom_loc = None
        self.atom_weight = None

    def atom_sampler(self, a_lamb_sample, num_atom: int):
        precision = a_lamb_sample[0]
        lamb = a_lamb_sample[1]

        atom_loc = []
        for _ in range(num_atom):
            unif_sample = uniform(0,1)
            if unif_sample < precision/(precision+len(self.y)):
                loc = sp.stats.poisson.rvs(lamb)
            else:
                loc = self.y[randint(0, len(self.y)-1)]
            atom_loc.append(loc)
            
        left_stick_length = 1.0
        atom_weight = []
        for _ in range(num_atom-1):
            portion = betavariate(1, precision+len(self.y))
            weight = portion * left_stick_length
            atom_weight.append(weight)
            left_stick_length = left_stick_length * (1 - portion)
        atom_weight.append(left_stick_length)
        return atom_loc, atom_weight
    

if __name__=="__main__":
    case1 = True
    case2 = True

    #need: add histogram by atom-weight boxplot
    if case1:
        plt.hist(pois5_300, bins=20)
        plt.show()
        fit_inst = HW2_Q2_alpha_lambda_sampler(pois5_300, [10, 10], None)
        fit_inst.generate_samples(2000)
        diag_inst = MCMC_Diag()
        diag_inst.set_mc_samples_from_list(fit_inst.MC_sample)
        diag_inst.set_variable_names([r"$\alpha$",r"$\lambda$"])
        diag_inst.burnin(1000)
        diag_inst.show_traceplot((1,2))
        
        cdf_inst = Post_DP_Q2(pois5_300, 20230504)
        density_boxplot_mat = []

        for mc_sample in diag_inst.MC_sample:
            atom_loc, atom_weight = cdf_inst.atom_sampler(mc_sample, 3000)
            unified_grid = [i for i in range(0, 20)]
            unified_weight = cdf_inst.atom_loc_unifier_by_expansion(unified_grid, atom_loc, atom_weight)
            density_boxplot_mat.append(unified_weight)

            grid, increments, sample_path = cdf_inst.cumulatative_dist_func(atom_loc, atom_weight, 0, 30)

            plt.step(grid, sample_path, where='post', label=r'test', alpha=0.2, c='blue')
        plt.step(np.arange(0,30), 0.7*sp.stats.poisson.cdf(np.arange(0,30), mu=3) + 0.3*sp.stats.poisson.cdf(np.arange(0,30), mu=11), where='post', c='red')
        plt.show()

        plt.boxplot(np.array(density_boxplot_mat))
        plt.show()

    if case2:
        plt.hist(pois_mixture_300, bins=20)
        plt.show()
        fit_inst = HW2_Q2_alpha_lambda_sampler(pois_mixture_300, [10, 10], None)
        fit_inst.generate_samples(2000)
        diag_inst = MCMC_Diag()
        diag_inst.set_mc_samples_from_list(fit_inst.MC_sample)
        diag_inst.set_variable_names([r"$\alpha$",r"$\lambda$"])
        diag_inst.burnin(1000)
        # diag_inst.thinning(10)
        diag_inst.show_traceplot((1,2))
        
        cdf_inst = Post_DP_Q2(pois_mixture_300, 20230504)
        density_boxplot_mat = []

        for mc_sample in diag_inst.MC_sample:
            atom_loc, atom_weight = cdf_inst.atom_sampler(mc_sample, 3000)
            unified_grid = [i for i in range(0, 20)]
            unified_weight = cdf_inst.atom_loc_unifier_by_expansion(unified_grid, atom_loc, atom_weight)
            density_boxplot_mat.append(unified_weight)

            grid, increments, sample_path = cdf_inst.cumulatative_dist_func(atom_loc, atom_weight, 0, 30)

            plt.step(grid, sample_path, where='post', label=r'test', alpha=0.2, c='blue')
        plt.step(np.arange(0,30), 0.7*sp.stats.poisson.cdf(np.arange(0,30), mu=3) + 0.3*sp.stats.poisson.cdf(np.arange(0,30), mu=11), where='post', c='red')
        plt.show()

        plt.boxplot(np.array(density_boxplot_mat))
        plt.show()