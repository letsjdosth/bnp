from math import sqrt, exp, log, pi
from random import seed, uniform, betavariate, normalvariate, randint, choices
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag

seed(20230515)
np.random.seed(20230515)

def data_simulator_beta_mixture(n_num_data):
    y_data_list = []
    for _ in range(n_num_data):
        u = uniform(0,1)
        if u<0.3:
            y_data_list.append(betavariate(0.1, 0.9))
        elif u<0.8:
            y_data_list.append(betavariate(5, 10))
        else:
            y_data_list.append(betavariate(12, 3))
    return y_data_list

sim1_grid = [x/100 for x in range(100)]
sim1_pdf = [0.3*sp.stats.beta.pdf(x/100, 0.1, 0.9)+0.5*sp.stats.beta.pdf(x/100, 5, 10)+0.2*sp.stats.beta.pdf(x/100, 12, 3) for x in range(100)]

sim1_y_200 = data_simulator_beta_mixture(200)
sim1_y_2000 = data_simulator_beta_mixture(2000)

plt.hist(sim1_y_2000, bins=50, density=True)
plt.plot(sim1_grid, sim1_pdf)
plt.show()

def data_simulator_logit_normal(n_num_data):
    y_data_list = []
    for _ in range(n_num_data):
        mu = 0
        sigma = 1.7
        x = normalvariate(mu, sigma)
        y_data_list.append(1/(1+np.exp(-x)))
    return y_data_list

def pdf_logit_normal(x):
    mu = 0
    sigma = 1.7
    return 1/sqrt(2*pi*sigma**2) * exp(-(1/(2*sigma**2))*(log(x/(1-x))-mu)**2)*(1/x + 1/(1-x))

sim2_grid = [x/100 for x in range(1,100)]
sim2_pdf = [pdf_logit_normal(x/100) for x in range(1,100)]

sim2_y_200 = data_simulator_logit_normal(200)
sim2_y_2000 = data_simulator_logit_normal(2000)

plt.hist(sim2_y_2000, bins=50, density=True)
plt.plot(sim2_grid, sim2_pdf)
plt.show()



def pois_log_prior_pmf_for_k(k):
    "on 0,1,2,... "
    return sp.stats.poisson.logpmf(k, 1)


class Random_Bernstein_Poly_Posterior(MCMC_Gibbs):
    def __init__(self, data: list[float], log_prior_pmf_for_k, dp_prior_F0_distfunc, dp_prior_M: int) -> None:
        self.x_data = data
        self.n = len(data)
        self.log_prior_pmf_for_k = log_prior_pmf_for_k
        self.dp_prior_F0_distfunc = dp_prior_F0_distfunc
        self.dp_prior_M = dp_prior_M
        
    def _theta_group_indicator(self, y, k):
        for j in range(1,k+1):
            if y <= j/k:
                return j

    def _full_conditional_sampler_for_k(self, last_param):
        #param
        # 0  1
        #[k, [y_1,...,y_n]]
        def log_target_mass(k):
            log_mass = self.log_prior_pmf_for_k(k)
            for x, y in zip(self.x_data, last_param[1]):
                theta_grp_indicator = self._theta_group_indicator(y,k)
                log_mass += sp.stats.beta.logpdf(x, a=theta_grp_indicator, b=k-theta_grp_indicator+1)
        def proposal_sampler(from_smpl):
            window = 2
            candid = randint(max(1,from_smpl-window), from_smpl+window)
            return candid
        def log_proposal_mass(from_smpl, to_smpl):
            window = 2
            window_length = from_smpl+window - max(1,from_smpl-window) + 1
            return -log(window_length)
        k_mcmc_inst = MCMC_MH(log_target_mass, log_proposal_mass, proposal_sampler, [last_param[0]])
        k_mcmc_inst.generate_samples(2, verbose=False)
        new_k = k_mcmc_inst.MC_sample[-1][0]
        new_sample = [new_k, last_param[1]]
        return new_sample
    
    def _b_F0_weight_func(self, x, k): #depend on F0
        b_val = 0
        for j in range(1,k+1):
            j_term = self.dp_prior_F0_distfunc(j/k) - self.dp_prior_F0_distfunc((j-1)/k)
            j_term *= sp.stats.beta.pdf(x, a=j, b=k-j+1)
            b_val += j_term
        return b_val

    def _full_conditional_sampler_for_y(self, last_param):
        #param
        # 0  1
        #[k, [y_1,...,y_n]]
        k = last_param[0]
        
        for v, x in enumerate(self.x_data):
            b_val_at_x = self._b_F0_weight_func(x, k)
            q_vec = [self.dp_prior_M * b_val_at_x] # #q_{v,0}
            for w, y in enumerate(last_param):
                if v==w:
                    q_vec.append(0)
                else:
                    theta_grp_indicator = self._theta_group_indicator(y,k)
                    q_vec.append(sp.stats.beta.pdf(x, a=theta_grp_indicator, b=k-theta_grp_indicator+1))
            
            q = choices([x-1 for x in range(self.n+1)], weights=q_vec)[0]
            if q == -1:
                pass #implement here. draw y from psi(y)
            else:
                y_v = last_param[q]