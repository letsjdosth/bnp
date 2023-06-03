from math import log, inf
from random import randint, seed, choices
import csv

import numpy as np
import scipy.stats as sp_stats

from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_Diag
from pyBayes.rv_gen_gamma import Sampler_univariate_InvGamma

class Hw3Q1_MarginalSampler(MCMC_Gibbs):
    def __init__(self, initial, y_obs: list, hyper: tuple | None =None):
        #param
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        self.MC_sample = [initial]
        self.y = y_obs
        self.n = len(y_obs)

        self.inv_gamma_sampler_inst = Sampler_univariate_InvGamma()

        if hyper is None:
            self.hyper_a_phi = 0.01
            self.hyper_b_phi = 0.01
            self.hyper_a_mu = 0
            self.hyper_b_mu = 1
            self.hyper_a_tau2 = 0.01
            self.hyper_b_tau2 = 0.01
            self.hyper_a_alpha = 0.01
            self.hyper_b_alpha = 0.01
        else:
            self.hyper_a_phi, self.hyper_b_phi, self.hyper_a_mu, self.hyper_b_mu, self.hyper_a_tau2, self.hyper_b_tau2, self.hyper_a_alpha, self.hyper_b_alpha = hyper
    
    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_theta(new)
        new = self.full_conditional_sampler_phi(new)
        new = self.full_conditional_sampler_mu(new)
        new = self.full_conditional_sampler_tau2(new)
        new = self.full_conditional_sampler_eta(new)
        new = self.full_conditional_sampler_alpha(new)
        self.MC_sample.append(new)
    
    def theta_counter(self, theta_vec:list):
        "n_star, theta_stars, n_js, first-class func"
        distinguished_theta_num_n_star = 0
        distinguished_theta_stars = []
        distinguished_theta_star_count = []
        sorted_thetas = sorted(theta_vec)
        last_theta = inf #dangerous
        for theta in sorted_thetas:
            # tol = 1e-6
            if last_theta != theta:
                distinguished_theta_num_n_star += 1
                distinguished_theta_stars.append(theta)
                distinguished_theta_star_count.append(1)
                last_theta = theta
            else:
                distinguished_theta_star_count[-1] += 1
        return distinguished_theta_num_n_star, distinguished_theta_stars, distinguished_theta_star_count
        

    def full_conditional_sampler_theta(self, last_param):
        #param
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        phi = last_param[1]
        mu = last_param[2]
        tau2 = last_param[3]
        alpha = last_param[5]
        
        #update new
        new_sample = self.deep_copier(last_param)
        new_theta = new_sample[0] #pointer
        for i, y in enumerate(self.y):
            new_theta_i = None
            
            unif_sample = sp_stats.uniform.rvs()
            q0_weight_for_g0 = sp_stats.norm.pdf(y, mu, (tau2+phi)**0.5) * alpha
            _, theta_stars, n_j_stars = self.theta_counter(new_theta[:i]+new_theta[i+1:])
            prev_theta_weights = [nj * sp_stats.norm.pdf(y, t, phi**0.5) for t,nj in zip(theta_stars, n_j_stars)]
            normalizing_const_of_weight = q0_weight_for_g0 + sum(prev_theta_weights)

            if unif_sample < q0_weight_for_g0/normalizing_const_of_weight:
                post_g0_var = 1/(1/phi + 1/tau2)
                post_g0_mean = (y/phi + mu/tau2)*post_g0_var
                new_theta_i = sp_stats.norm.rvs(post_g0_mean, post_g0_var**0.5)
            else:
                new_theta_i = choices(theta_stars, weights=prev_theta_weights)[0]
            new_theta[i] = new_theta_i
            # new_sample[0] = new_theta #it is not needed
        return new_sample
    
    def full_conditional_sampler_phi(self, last_param):
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        new_sample = [x for x in last_param]
        thetas = last_param[0]
        shape = self.hyper_a_phi + self.n/2
        rate = self.hyper_b_phi
        for y_i, t_i in zip(self.y, thetas):
            rate += (0.5 * (y_i-t_i)**2)
        new_phi = self.inv_gamma_sampler_inst.sampler(shape, rate)

        # new_phi = 1 #true
        new_sample[1] = new_phi
        return new_sample
    
    def full_conditional_sampler_mu(self, last_param):
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        tau2 = last_param[3]
        
        new_sample = [x for x in last_param]
        
        n_star, theta_stars, _ = self.theta_counter(new_sample[0])
        
        post_var = 1/(1/self.hyper_b_mu + n_star/tau2)
        post_mean = (self.hyper_a_mu/self.hyper_b_mu + sum(theta_stars)/tau2)*post_var
        new_mu = sp_stats.norm.rvs(post_mean, post_var**0.5)
        
        # new_mu = 0.05 #true
        new_sample[2] = new_mu
        return new_sample
    
    def full_conditional_sampler_tau2(self, last_param):
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        mu = last_param[2]
        
        new_sample = [x for x in last_param]
        n_star, theta_stars, _ = self.theta_counter(new_sample[0])

        shape = self.hyper_a_tau2 + n_star/2
        rate = self.hyper_b_tau2
        for t_i in theta_stars:
            rate += (0.5 * (t_i-mu)**2)
        new_tau2 = self.inv_gamma_sampler_inst.sampler(shape, rate)
        new_sample[3] = new_tau2
        return new_sample
    
    def full_conditional_sampler_eta(self, last_param):
        #  0                      1    2   3     4    5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        alpha = last_param[5]
        
        new_sample = [x for x in last_param]
        new_eta = sp_stats.beta.rvs(alpha+1, self.n)
        new_sample[4] = new_eta
        return new_sample
    
    def full_conditional_sampler_alpha(self, last_param):
        #  0                      1    2   3     4     5
        # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
        eta = last_param[4]

        new_sample = [x for x in last_param]
        n_star, _, _ = self.theta_counter(new_sample[0])

        shape1 = self.hyper_a_alpha + n_star
        shape2 = shape1 - 1
        rate = self.hyper_b_alpha - log(eta)
        mixture1_weight = shape2 / (self.n*rate + shape2)

        new_alpha = None
        unif_sample = sp_stats.uniform.rvs()
        if unif_sample < mixture1_weight:
            new_alpha = sp_stats.gamma.rvs(a=shape1, scale=1/rate)
        else:
            new_alpha = sp_stats.gamma.rvs(a=shape2, scale=1/rate)

        new_sample[5] = new_alpha
        return new_sample
    
def posterior_predictive_density_estimator(grid, posterior_samples, n_sample_size):
    #  0                      1    2   3     4     5
    # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
    # 0       1            2
    #(n_star, theta_stars, n_js)
    temp_gibbs_inst = Hw3Q1_MarginalSampler([None], [None], None)
    prob_vec_at_grid = []
    for y in grid:
        prob_vec_at_y = []
        for s in posterior_samples:
            unif_sample = sp_stats.uniform.rvs()
            post_g0_weight = s[5]/(s[5]+n_sample_size)
            if unif_sample < post_g0_weight:
                theta_0 = sp_stats.norm.rvs(s[2], s[3]**0.5)
            else:
                _, theta_stars, n_js = temp_gibbs_inst.theta_counter(s[0])
                theta_0 = choices(theta_stars, weights=n_js)
            # print(theta_0, s[1]) # for debug
            post_prob_y0 = sp_stats.norm.pdf(y, theta_0, s[1]**0.5)
            prob_vec_at_y.append(post_prob_y0)
        expected_at_y = sum(prob_vec_at_y)/len(prob_vec_at_y)
        prob_vec_at_grid.append(expected_at_y)
    return prob_vec_at_grid

def prior_predictive_density_estimator(grid, hyper, prior_simul_size=300):
    # hyperparameters
    #  0            1            2           3           4             5             6              7
    # [hyper_a_phi, hyper_b_phi, hyper_a_mu, hyper_b_mu, hyper_a_tau2, hyper_b_tau2, hyper_a_alpha, hyper_b_alpha]
    #  0                      1    2   3     4     5
    # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
    # 0       1            2
    #(n_star, theta_stars, n_js)
    prior_samples = []
    for _ in range(prior_simul_size):
        invgamma_sampler_inst = Sampler_univariate_InvGamma()
        phi_prior_sample = invgamma_sampler_inst.sampler(hyper[0], hyper[1])
        mu_prior_sample = sp_stats.norm.rvs(hyper[2], hyper[3]**0.5)
        tau2_prior_sample = invgamma_sampler_inst.sampler(hyper[4], hyper[5])
        theta_prior_sample = None
        alpha_prior_sample = sp_stats.gamma.rvs(a=hyper[6], scale=1/hyper[7])
        prior_samples.append([[theta_prior_sample], phi_prior_sample, mu_prior_sample, tau2_prior_sample, None, alpha_prior_sample])

    prob_vec_at_grid = []
    for y in grid:
        prob_vec_at_y = []
        for s in prior_samples:
            theta_0 = sp_stats.norm.rvs(s[2], s[3]**0.5)
            prior_prob_y0 = sp_stats.norm.pdf(y, theta_0, s[1]**0.5)
            # print(s[2], s[3]**0.5, theta_0, y, prior_prob_y0) #for debug
            prob_vec_at_y.append(prior_prob_y0)
        expected_at_y = sum(prob_vec_at_y)/len(prob_vec_at_y)
        prob_vec_at_grid.append(expected_at_y)
    return prob_vec_at_grid

def sample_reader_from_csv_HW3Q1(filename):
    #  0                      1    2   3     4     5
    # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
    with open(filename+".csv", "r", newline="") as csv_f:
        csv_reader = csv.reader(csv_f)
        MC_sample = []
        for row in csv_reader:
            theta = [float(x) for x in row[0][1:-1].split(',')]
            s = [theta, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])]
            MC_sample.append(s)
    return MC_sample
            

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data = []
    with open("data/hwk3-data.txt", "r", newline="\n") as f:
        for line in f:
            data.append(float(line))
    # print(data[:20])
    print(len(data)) #250
    # plt.hist(data, bins=50)
    # plt.show()

    seed(20230528)
    np.random.seed(20230528)
    
    # hyperparameters
    #  0            1            2           3           4             5             6              7
    # [hyper_a_phi, hyper_b_phi, hyper_a_mu, hyper_b_mu, hyper_a_tau2, hyper_b_tau2, hyper_a_alpha, hyper_b_alpha]
    hyperparam_set1 = [0.01, 0.01, 0, 1, 0.01, 0.01, 0.01, 0.01] #less informative
    hyperparam_set2 = [1, 0.01, 0, 1, 0.01, 0.01, 0.01, 0.01] #inflated phi
    hyperparam_set3 = [0.01, 0.01, 0, 1, 1, 0.01, 0.01, 0.01] #inflated tau2
    hyperparam_set4 = [0.01, 0.01, 5, 0.1, 0.01, 0.01, 0.01, 0.01] #wrong mean of mu

    hyperparam_set01 = [0.01, 0.01, 0, 1, 0.01, 0.01, 2, 15] #alpha with E(n*)=1
    hyperparam_set02 = [0.01, 0.01, 0, 1, 0.01, 0.01, 2, 4] #alpha with E(n*)=3
    hyperparam_set03 = [0.01, 0.01, 0, 1, 0.01, 0.01, 2, 0.9] #alpha with E(n*)=10
    hyperparam_set04 = [0.01, 0.01, 0, 1, 0.01, 0.01, 2, 0.1] #alpha with E(n*)=48

    now_hyper = hyperparam_set1

    #  0                      1    2   3     4    5
    # [[theta_1,...,theta_n], phi, mu, tau2, eta, alpha]
    #                     0     1    2    3   4    5
    gibbs_iter_initial = [data, 0.6, 0.5, 10, 0.02, 0.5]
    
    gibbs_inst = Hw3Q1_MarginalSampler(gibbs_iter_initial, data, hyper=now_hyper)
    diag_inst = MCMC_Diag()
    
    mcmc_run = True
    if mcmc_run:
        gibbs_inst.generate_samples(500, print_iter_cycle=50)
        MC_sample = gibbs_inst.MC_sample
        diag_inst.set_mc_samples_from_list(MC_sample)
        # diag_inst.write_samples("hw3q1_MCMCsamples")
    else:    
        MC_sample = sample_reader_from_csv_HW3Q1("hw3q1_MCMCsamples")
        print(len(MC_sample))
        diag_inst.set_mc_samples_from_list(MC_sample)

    diag_inst.burnin(200)

    diag_inst_theta = MCMC_Diag()
    diag_inst_theta.set_mc_samples_from_list(diag_inst.get_specific_dim_samples(0))
    diag_inst_theta.set_variable_names(["theta"+str(i+1) for i in range(len(diag_inst_theta.MC_sample[0]))])
    diag_inst_other_params = MCMC_Diag()
    diag_inst_other_params.set_mc_samples_from_list([x[1:] for x in diag_inst.MC_sample])
    diag_inst_other_params.set_variable_names(["phi", "mu", "tau2", "eta", "alpha"])

    n_star_vec = []
    for t in diag_inst_theta.MC_sample:
        n_star, theta_stars, n_js = gibbs_inst.theta_counter(t)
        n_star_vec.append([n_star])

    diag_inst_n_star = MCMC_Diag()
    diag_inst_n_star.set_mc_samples_from_list(n_star_vec)
    diag_inst_n_star.set_variable_names(["n_star"])

    # ==
    diag_inst_theta.show_traceplot((4,5), [x for x in range(20)])
    diag_inst_theta.show_cred_interval_plot(mark="median")
    diag_inst_other_params.show_traceplot((2,3))
    diag_inst_other_params.show_hist((2,3))
    diag_inst_n_star.show_traceplot((1,1))
    diag_inst_n_star.show_hist((1,1))

    # ==
    grid = np.linspace(-7, 7, 50).tolist()
    density_pt_est_on_grid = posterior_predictive_density_estimator(grid, diag_inst.MC_sample, 250)
    density_true_on_grid = [0.2*sp_stats.norm.pdf(x, -5, 1)+0.5*sp_stats.norm.pdf(x, 0, 1)+0.3*sp_stats.norm.pdf(x, 3.5, 1) for x in grid]
    density_prior_pt_est_on_grid = prior_predictive_density_estimator(grid, now_hyper)


    plt.hist(data, bins=50, density=True, color="orange", label='data')
    plt.plot(grid, density_prior_pt_est_on_grid, color="green", label='prior')
    plt.plot(grid, density_pt_est_on_grid, color="blue", label='posterior')
    plt.plot(grid, density_true_on_grid, color="red", label='true')
    plt.legend()
    plt.show()
