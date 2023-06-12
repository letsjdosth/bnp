import csv
from math import sqrt, exp, log, pi, ceil
from random import seed, uniform, betavariate, normalvariate, randint, choices
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag
from pyBayes.rv_gen_dirichlet import Sampler_Dirichlet


class Random_Bernstein_Poly_Posterior_MCMC(MCMC_Gibbs):
    def __init__(self, data: list[float], initial, 
                 log_prior_pmf_for_k, 
                 dp_prior_M: int, dp_prior_F0_distfunc, dp_prior_log_f0_densityfunc,
                 proposal_k_window=3, proposal_y_window=3) -> None:
        self.MC_sample = [initial]
        self.x_data = data
        self.n = len(data)
        self.log_prior_pmf_for_k = log_prior_pmf_for_k
        self.dp_prior_F0_distfunc = dp_prior_F0_distfunc
        self.dp_prior_log_f0_densityfunc = dp_prior_log_f0_densityfunc
        self.dp_prior_M = dp_prior_M

        self.proposal_k_window = proposal_k_window
        self.proposal_y_window = proposal_y_window

    def _theta_group_indicator(self, y, k):
        for j in range(1,k+1):
            if y <= j/k:
                return j

    def _full_conditional_sampler_for_k(self, last_param):
        #param
        # 0  1
        #[k, [y_1,...,y_n]]
        def log_target_mass(k):
            k = k[0]
            log_mass = self.log_prior_pmf_for_k(k)
            for x, y in zip(self.x_data, last_param[1]):
                theta_grp_indicator = self._theta_group_indicator(y,k)
                log_mass += sp.stats.beta.logpdf(x, a=theta_grp_indicator, b=k-theta_grp_indicator+1)
            return log_mass
        def proposal_sampler(from_smpl):
            from_smpl = from_smpl[0]
            window = self.proposal_k_window
            candid = randint(max(1,from_smpl-window), from_smpl+window)
            return [candid]
        def log_proposal_mass(from_smpl, to_smpl):
            from_smpl = from_smpl[0]
            window = self.proposal_k_window
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
        
        new_y_vec = []
        for v, x in enumerate(self.x_data):
            b_val_at_x = self._b_F0_weight_func(x, k)
            q_vec = [self.dp_prior_M * b_val_at_x] # #q_{v,0}
            for w, y in enumerate(last_param[1]):
                if v==w:
                    q_vec.append(0)
                else:
                    theta_grp_indicator = self._theta_group_indicator(y,k)
                    q_vec.append(sp.stats.beta.pdf(x, a=theta_grp_indicator, b=k-theta_grp_indicator+1))
            
            q = choices([x-1 for x in range(self.n+1)], weights=q_vec)[0]
            if q == -1:
                def log_target_density(y_place):
                    y_place = y_place[0]
                    theta_grp_indicator = self._theta_group_indicator(y_place,k)
                    return self.dp_prior_log_f0_densityfunc(y_place) + sp.stats.beta.logpdf(x, a=theta_grp_indicator, b=k-theta_grp_indicator+1)
                def proposal_sampler(from_smpl):
                    return [uniform(0,1)] #hmm
                def log_proposal_density(from_smpl, to_smpl):
                    return 0
                y_mcmc_inst = MCMC_MH(log_target_density, log_proposal_density, proposal_sampler,
                                      initial=[x]) #hmm2
                y_mcmc_inst.generate_samples(2, verbose=False)
                new_y_v = y_mcmc_inst.MC_sample[-1][0]
            else:
                new_y_v = last_param[1][q]
            new_y_vec.append(new_y_v)
        new_sample = [k, new_y_vec]
        return new_sample
    
    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self._full_conditional_sampler_for_k(new)
        new = self._full_conditional_sampler_for_y(new)
        self.MC_sample.append(new)


class Random_Bernstein_Poly_Posterior_Density_Work:
    def __init__(self, MC_sample: list, dp_prior_M: int, dp_prior_F0_distfunc) -> None:
        self.MC_sample = MC_sample
        #MC_sample
        # 0  1
        #[k, [y_1,...,y_n]]

        self.dp_prior_F0_distfunc = dp_prior_F0_distfunc
        self.dp_prior_M = dp_prior_M
        
        self.dir_generator_inst = Sampler_Dirichlet(20230518)

    def _counter_for_r(self, k, y_vec:list):
        sorted_y_vec = sorted(y_vec)
        r_vec = []
        j=1
        r=0
        for y in sorted_y_vec:
            if y <= j/k:
                r += 1
            else:
                while y > j/k:
                    r_vec.append(r)
                    j += 1
                    r = 0
                r = 1
        r_vec.append(r)
        while j < k:
            r_vec.append(0)
            j += 1
        # print(r_vec) # for debugging
        # assert len(r_vec)==k #after debugging, delete this line
        return r_vec

    def _dir_param_a_constructor(self, k, include_a0=False):
        a_vec = []
        if include_a0:
            a_vec.append(self.dp_prior_M * self.dp_prior_F0_distfunc(0))
            
        for j in range(1, k+1):
            a = self.dp_prior_M * (self.dp_prior_F0_distfunc(j/k) - self.dp_prior_F0_distfunc((j-1)/k))
            a_vec.append(a)
        return a_vec

    def _conditional_sampler_for_w(self):
        #param
        # 0  1
        #[k, [y_1,...,y_n]]
        w_vec_list = []
        for k, y_vec in self.MC_sample:
            a_vec = self._dir_param_a_constructor(k)
            r_vec = self._counter_for_r(k, y_vec)
            assert len(a_vec)==len(r_vec)
            dir_param_vec = [a + r for a, r in zip(a_vec, r_vec)]
            w_vec = self.dir_generator_inst.sampler(dir_param_vec)
            w_vec_list.append(w_vec)
        # print(w_vec_list) # for debugging
        return w_vec_list

    def _eval_posterior_density(self, w_vec, grid):
        k = len(w_vec)
        val_on_grid = []
        for x in grid:
            val = 0
            for i, w_j in enumerate(w_vec):
                j = i+1
                val += (w_j * sp.stats.beta.pdf(x, j, k-j+1))
            val_on_grid.append(val)
        return val_on_grid

    def get_posterior_density(self, grid):
        w_vec_list = self._conditional_sampler_for_w()
        post_density_list = []
        for w_vec in w_vec_list:
            post_density_list.append(self._eval_posterior_density(w_vec, grid))
        return post_density_list

    def get_posterior_cred_interval(self, grid, prob_level, center='mean'):
        tail_prob_level = (1-prob_level)/2
        post_density_ndarray = np.array(self.get_posterior_density(grid))
        mean_vec = np.mean(post_density_ndarray, axis=0)
        quantile_vec = np.quantile(post_density_ndarray, q=[tail_prob_level, 0.5, 1-tail_prob_level], axis=0)
        quantile_vec = quantile_vec.tolist()
        ci_on_grid = [quantile_vec[0], quantile_vec[2]]
        
        center_on_grid = None
        if center == "median":
            center_on_grid = quantile_vec[1]
        elif center == "mean":
            center_on_grid = mean_vec
        else:
            raise ValueError("center can be either 'mean' or 'median")
        return center_on_grid, ci_on_grid


def sample_reader_from_csv_bernstein_poly_prior(filename):
    #MC_sample
    # 0  1
    #[k, [y_1,...,y_n]]
    with open(filename+".csv", "r", newline="") as csv_f:
        csv_reader = csv.reader(csv_f)
        MC_sample = []
        for row in csv_reader:
            k = int(row[0])
            y = [float(row[i]) for i in range(1, len(row))]
            MC_sample.append([k,y])
    return MC_sample
            

if __name__=="__main__":
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
    sim1_y_500 = data_simulator_beta_mixture(500)

    plt.hist(sim1_y_200, bins=50, density=True, alpha=0.5)
    plt.plot(sim1_grid, sim1_pdf, color='red')
    plt.show()

    def data_simulator_logit_normal(n_num_data, mu, sigma):
        y_data_list = []
        for _ in range(n_num_data):
            x = normalvariate(mu, sigma)
            y_data_list.append(1/(1+np.exp(-x)))
        return y_data_list

    def pdf_logit_normal(x, mu, sigma):
        return 1/sqrt(2*pi*sigma**2) * exp(-(1/(2*sigma**2))*(log(x/(1-x))-mu)**2)*(1/(x*(1-x)))

    def data_simulator_logit_mixture_normal(n_num_data):
        n1 = 0
        w1 = 0.4
        mu1 = -1
        sigma1 = 0.3

        n2 = 0
        # w2 = 1-w1
        mu2 = 1.2
        sigma2 = 0.6
        
        for _ in range(n_num_data):
            u = uniform(0, 1)
            if u < w1:
                n1 += 1
            else:
                n2 += 1
        sample_list1 = data_simulator_logit_normal(n1, mu1, sigma1)
        sample_list2 = data_simulator_logit_normal(n2, mu2, sigma2)
        return sample_list1 + sample_list2
    
    def pdf_logit_mixture_normal(x):
        w1 = 0.4
        mu1 = -1
        sigma1 = 0.3

        w2 = 1-w1
        mu2 = 1.2
        sigma2 = 0.6
        return w1*pdf_logit_normal(x,mu1,sigma1)+w2*pdf_logit_normal(x,mu2,sigma2)
    
    sim2_grid = [x/100 for x in range(1,100)]
    sim2_pdf = [pdf_logit_mixture_normal(x/100) for x in range(1,100)]

    sim2_y_200 = data_simulator_logit_mixture_normal(200)
    sim2_y_500 = data_simulator_logit_mixture_normal(500)

    plt.hist(sim2_y_200, bins=50, density=True, alpha=0.5)
    plt.plot(sim2_grid, sim2_pdf, color='red')
    plt.show()

    def data_simulator_bernstein_poly_prior(n_num_data):
        y_data_list = []
        k = 60
        #F = 0.5*beta_dist(60,30)+0.5*beta_dist(1,2)
        for _ in range(n_num_data):
            u = uniform(0,1)
            if u < 0.5:
                v = betavariate(60, 30)
            else:
                v = betavariate(1, 2)
            j = ceil(v*k)
            y = betavariate(j, k-j+1)
            y_data_list.append(y)
        return y_data_list

    def approx_pdf_bernstein_poly_prior(x):
        k = 60
        #F = 0.5*beta_dist(60,30)+0.5*beta_dist(1,2)
        mc_iter = 1000
        mc_integ_vals = []
        for _ in range(mc_iter):
            u = uniform(0,1)
            if u < 0.5:
                v = betavariate(60, 30)
            else:
                v = betavariate(1, 2)
            j = ceil(v*k)
            y = sp.stats.beta.pdf(x, j, k-j+1)
            mc_integ_vals.append(y)
        return sum(mc_integ_vals)/mc_iter


    sim3_grid = [x/100 for x in range(100)]
    sim3_F_pdf = [0.5*sp.stats.beta.pdf(x/100, 60, 30)+0.5*sp.stats.beta.pdf(x/100, 1, 2) for x in range(100)]
    sim3_pdf = [approx_pdf_bernstein_poly_prior(x/100) for x in range(100)]

    sim3_y_200 = data_simulator_bernstein_poly_prior(200)
    sim3_y_500 = data_simulator_bernstein_poly_prior(500)

    plt.hist(sim3_y_200, bins=60, density=True, alpha=0.5)
    plt.plot(sim3_grid, sim3_pdf, c='orange')
    plt.plot(sim3_grid, sim3_F_pdf, c='red')
    plt.show()

    def pois_log_prior_pmf_for_k(k):
        "on 0,1,2,... "
        return sp.stats.poisson.logpmf(k, 1)
    def pois_log_prior_pmf_for_k_lam50(k):
        "on 0,1,2,... "
        return sp.stats.poisson.logpmf(k, 50)
    def unif_cdf(y):
        return y
    def unif_log_pdf(y):
        return 0

    # ===

    sim1 = False
    sim2 = True
    sim3 = False

    if sim1:
        sim1_run = False

        test_data = sim1_y_200
        test_data_len = len(test_data)
        test_MCMC_inst = Random_Bernstein_Poly_Posterior_MCMC(
            test_data, [20, [uniform(0,1) for _ in range(test_data_len)]],
            pois_log_prior_pmf_for_k, 1, unif_cdf, unif_log_pdf)
        test_MCMC_diag_inst1 = MCMC_Diag()

        if sim1_run:
            test_MCMC_inst.generate_samples(3000, print_iter_cycle=50)

            test_MCMC_diag_writer_inst = MCMC_Diag()
            test_MCMC_diag_writer_inst.set_mc_samples_from_list([[x[0]]+x[1] for x in test_MCMC_inst.MC_sample])
            test_MCMC_diag_writer_inst.write_samples("k_y_flatten_from_sim1_y_200_")
            test_MCMC_diag_inst1.set_mc_sample_from_MCMC_instance(test_MCMC_inst)
        else:
            raw_MC_samples = sample_reader_from_csv_bernstein_poly_prior("./bernstein_poly_prior_sim1/k_y_flatten_from_sim1_y_200_")
            test_MCMC_diag_inst1.set_mc_samples_from_list(raw_MC_samples)

        test_MCMC_diag_inst1.set_variable_names(["k", "y"])
        test_MCMC_diag_inst1.burnin(500)
        test_MCMC_diag_inst1.show_traceplot_specific_dim(0, True)
        test_MCMC_diag_inst1.show_hist_specific_dim(0, True)
        
        test_MCMC_diag_inst2 = MCMC_Diag()
        test_MCMC_diag_inst2.set_mc_samples_from_list(test_MCMC_diag_inst1.get_specific_dim_samples(1))
        test_MCMC_diag_inst2.set_variable_names(["y"+str(i+1) for i in range(len(test_MCMC_diag_inst2.MC_sample[0]))])
        test_MCMC_diag_inst2.show_traceplot((1,3), [0,1,2])

        test_MCMC_diag_inst1.thinning(5)

        test_density_inst = Random_Bernstein_Poly_Posterior_Density_Work(test_MCMC_diag_inst1.MC_sample, 1, unif_cdf)
        # for i, s in enumerate(test_density_inst.MC_sample):
        #     if i%200==0:
        #         print(test_density_inst._counter_for_r(*s))

        num_grid_pt = 100
        grid = [(x+0.5)/num_grid_pt for x in range(num_grid_pt)]
        # density_list = test_density_inst.get_posterior_density(grid)
        # for d in density_list:
        #     plt.plot(grid, d, color='blue')
            
        mean_on_grid, ci_on_grid = test_density_inst.get_posterior_cred_interval(grid, 0.95)
        plt.plot(grid, mean_on_grid, color='blue')
        plt.plot(grid, ci_on_grid[0], color='gray')
        plt.plot(grid, ci_on_grid[1], color='gray')

        plt.hist(test_data, bins=50, density=True, alpha=0.5)
        plt.plot(sim1_grid, sim1_pdf, color='red')
        plt.show()

    if sim2:
        sim2_run = False

        test_data = sim2_y_200
        test_data_len = len(test_data)
        
        test_MCMC_inst = Random_Bernstein_Poly_Posterior_MCMC(
            test_data, [20, [uniform(0,1) for _ in range(test_data_len)]],
            pois_log_prior_pmf_for_k_lam50, 1, unif_cdf, unif_log_pdf)
        test_MCMC_diag_inst1 = MCMC_Diag()

        if sim2_run:
            test_MCMC_inst.generate_samples(3000, print_iter_cycle=50)
            
            test_MCMC_diag_writer_inst = MCMC_Diag()
            test_MCMC_diag_writer_inst.set_mc_samples_from_list([[x[0]]+x[1] for x in test_MCMC_inst.MC_sample])
            test_MCMC_diag_writer_inst.write_samples("k_y_flatten_from_sim2_y_200_lam50pois")

            test_MCMC_diag_inst1.set_mc_sample_from_MCMC_instance(test_MCMC_inst)
        else:
            raw_MC_samples = sample_reader_from_csv_bernstein_poly_prior("./bernstein_poly_prior_sim2_lamb50(290min)/k_y_flatten_from_sim2_y_200_lam50pois")
            test_MCMC_diag_inst1.set_mc_samples_from_list(raw_MC_samples)
       
        
        test_MCMC_diag_inst1.set_variable_names(["k", "y"])
        test_MCMC_diag_inst1.burnin(500)
        test_MCMC_diag_inst1.show_traceplot_specific_dim(0, True)
        test_MCMC_diag_inst1.show_hist_specific_dim(0, True)
        
        test_MCMC_diag_inst2 = MCMC_Diag()
        test_MCMC_diag_inst2.set_mc_samples_from_list(test_MCMC_diag_inst1.get_specific_dim_samples(1))
        test_MCMC_diag_inst2.set_variable_names(["y"+str(i+1) for i in range(len(test_MCMC_diag_inst2.MC_sample[0]))])
        test_MCMC_diag_inst2.show_traceplot((1,3), [0,1,2])

        test_MCMC_diag_inst1.thinning(5) # be careful to the timing of thinning

        test_density_inst = Random_Bernstein_Poly_Posterior_Density_Work(test_MCMC_diag_inst1.MC_sample, 1, unif_cdf)
        # for i, s in enumerate(test_density_inst.MC_sample):
        #     if i%200==0:
        #         print(test_density_inst._counter_for_r(*s))

        num_grid_pt = 100
        grid = [(x+0.5)/num_grid_pt for x in range(num_grid_pt)]
        # density_list = test_density_inst.get_posterior_density(grid)
        # for d in density_list:
        #     plt.plot(grid, d, color='blue')
        
        mean_on_grid, ci_on_grid = test_density_inst.get_posterior_cred_interval(grid, 0.95)
        plt.plot(grid, mean_on_grid, color='blue')
        plt.plot(grid, ci_on_grid[0], color='gray')
        plt.plot(grid, ci_on_grid[1], color='gray')

        plt.hist(test_data, bins=50, density=True, alpha=0.5)
        plt.plot(sim2_grid, sim2_pdf, color='red')
        plt.show()
        
    if sim3:
        sim3_run = False

        test_data = sim3_y_200
        test_data_len = len(test_data)
        test_MCMC_inst = Random_Bernstein_Poly_Posterior_MCMC(
            test_data, [50, [uniform(0,1) for _ in range(test_data_len)]],
            pois_log_prior_pmf_for_k_lam50, 1, unif_cdf, unif_log_pdf)
        test_MCMC_diag_inst1 = MCMC_Diag()

        if sim3_run:
            test_MCMC_inst.generate_samples(3000, print_iter_cycle=50)

            test_MCMC_diag_writer_inst = MCMC_Diag()
            test_MCMC_diag_writer_inst.set_mc_samples_from_list([[x[0]]+x[1] for x in test_MCMC_inst.MC_sample])
            test_MCMC_diag_writer_inst.write_samples("k_y_flatten_from_sim3_y_200_")
            test_MCMC_diag_inst1.set_mc_sample_from_MCMC_instance(test_MCMC_inst)
        else:
            raw_MC_samples = sample_reader_from_csv_bernstein_poly_prior("./bernstein_poly_prior_sim3(261min)/k_y_flatten_from_sim3_y_200_")
            test_MCMC_diag_inst1.set_mc_samples_from_list(raw_MC_samples)

        test_MCMC_diag_inst1.set_variable_names(["k", "y"])
        test_MCMC_diag_inst1.burnin(500)
        test_MCMC_diag_inst1.show_traceplot_specific_dim(0, True)
        test_MCMC_diag_inst1.show_hist_specific_dim(0, True)
        
        test_MCMC_diag_inst2 = MCMC_Diag()
        test_MCMC_diag_inst2.set_mc_samples_from_list(test_MCMC_diag_inst1.get_specific_dim_samples(1))
        test_MCMC_diag_inst2.set_variable_names(["y"+str(i+1) for i in range(len(test_MCMC_diag_inst2.MC_sample[0]))])
        test_MCMC_diag_inst2.show_traceplot((1,3), [0,1,2])

        test_MCMC_diag_inst1.thinning(5)

        test_density_inst = Random_Bernstein_Poly_Posterior_Density_Work(test_MCMC_diag_inst1.MC_sample, 1, unif_cdf)
        # for i, s in enumerate(test_density_inst.MC_sample):
        #     if i%200==0:
        #         print(test_density_inst._counter_for_r(*s))

        num_grid_pt = 100
        grid = [(x+0.5)/num_grid_pt for x in range(num_grid_pt)]
        # density_list = test_density_inst.get_posterior_density(grid)
        # for d in density_list:
        #     plt.plot(grid, d, color='blue')
            
        mean_on_grid, ci_on_grid = test_density_inst.get_posterior_cred_interval(grid, 0.95)
        plt.plot(grid, mean_on_grid, color='blue')
        plt.plot(grid, ci_on_grid[0], color='gray')
        plt.plot(grid, ci_on_grid[1], color='gray')

        plt.hist(sim3_y_500, bins=60, density=True, alpha=0.5)
        plt.plot(sim3_grid, sim3_pdf, c='red')
        plt.plot(sim3_grid, sim3_F_pdf, c='orange')
        plt.show()
