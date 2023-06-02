from math import exp, log
import time
from random import gammavariate

import numpy as np
import scipy.stats as sp_stats
from scipy.special import digamma
import matplotlib.pyplot as plt

class Hw3Q1_MeanfieldVB:
    def __init__(self, initial, y_obs: list, truncation_N:int) -> None:
        
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]

        self.initial = initial
        self.optim_path = [initial]
        self.y = y_obs
        self.n = len(y_obs)

        # truncation
        self.truncation_N = truncation_N
        
        # fixed parameters
        self.hyper_m = 0.05
        self.hyper_s2 = 1
        self.hyper_alpha = 5

        self.hyper_a_phi = 100
        self.hyper_b_phi = 100

        

    def deep_copier(self, x_iterable) -> list:
        rep_x_list = []
        for x in x_iterable:
            try:
                _ = iter(x)
                rep_x_list.append(self.deep_copier(x))
            except TypeError:
                rep_x_list.append(x)
        return rep_x_list

    def one_iter_optimizer(self):
        new = self.deep_copier(self.optim_path[-1])
        # update iteratively
        new = self._update_param_gamma(new)
        new = self._update_param_xi(new)
        new = self._update_param_pi(new)
        new = self._update_param_beta(new)
        self.optim_path.append(new)

    def _pi_sum_over_n(self, pi_mat: list[list[float]]):
        #pi_mat : n by N
        np_pi_mat = np.array(pi_mat, dtype="float32")
        col_sum = np.sum(np_pi_mat, axis=0) #column sum
        return col_sum.tolist()

    def _update_param_gamma(self, last_param):
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]

        new_gamma = []
        pi_col_sum = self._pi_sum_over_n(last_param[4])
        for l in range(self.truncation_N-1):
            new_gamma_l = [1 + pi_col_sum[l], self.hyper_alpha + sum(pi_col_sum[l+1:])]
            new_gamma.append(new_gamma_l)

        new_param = last_param #pointer
        new_param[2] = new_gamma
        return new_param
    
    def _update_param_xi(self, last_param):
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]

        new_xi = []
        pi_col_sum = self._pi_sum_over_n(last_param[4])
        b1b2inv = last_param[0]/last_param[1]
        for l in range(self.truncation_N):
            sum_y_i_pi_il = 0
            for y_i, pi_i in zip(self.y, last_param[4]):
                sum_y_i_pi_il += (y_i * pi_i[l])
            
            new_xi_l1 = (self.hyper_m+self.hyper_s2*b1b2inv*sum_y_i_pi_il)/(1+self.hyper_s2*b1b2inv*pi_col_sum[l])
            new_xi_l2 = 1/(1/self.hyper_s2 + b1b2inv*pi_col_sum[l])
            new_xi.append([new_xi_l1, new_xi_l2])
        
        new_param = last_param #pointer
        new_param[3] = new_xi
        return new_param


    def _update_param_pi(self, last_param):
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]
        beta1 = last_param[0]
        beta2 = last_param[1]

        new_pi = []
        gamma_until_lminus1_sum_vec = [0]
        for gamma_l1, gamma_l2 in last_param[2]:
            cum_sum = gamma_until_lminus1_sum_vec[-1] + digamma(gamma_l2) - digamma(gamma_l1+gamma_l2)
            gamma_until_lminus1_sum_vec.append(cum_sum)

        for y_i in self.y:
            W_i_vec = []
            for l, (xi_l1, xi_l2) in enumerate(last_param[3]):
                W_il = 0.5*(digamma(beta1)-log(beta2))
                W_il -= (0.5*beta1/beta2 * (y_i**2 - 2*y_i*xi_l1 + xi_l2 + xi_l1**2))
                if l < self.truncation_N-1:
                    gamma_l1 = last_param[2][l][0]
                    gamma_l2 = last_param[2][l][1]
                    W_il += (digamma(gamma_l1) - digamma(gamma_l1+gamma_l2))
                W_il += gamma_until_lminus1_sum_vec[l]
                W_i_vec.append(W_il)
            unnormalized_e_Wi = [exp(w) for w in W_i_vec]
            nomalizing_const = sum(unnormalized_e_Wi)
            new_pi_i = [w/nomalizing_const for w in unnormalized_e_Wi]
            new_pi.append(new_pi_i)
        
        new_param = last_param #pointer
        new_param[4] = new_pi
        return new_param

    def _update_param_beta(self, last_param):
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]
        
        new_beta1 = self.hyper_a_phi + self.n/2
        new_beta2 = self.hyper_b_phi
        for pi_i, y_i in zip(last_param[4], self.y):
            for l, xi_l in enumerate(last_param[3]):
                new_beta2 += (0.5*pi_i[l]*(y_i**2 - 2*y_i*xi_l[0] + xi_l[1] + xi_l[0]**2))
        new_param = last_param
        new_param[0] = new_beta1
        new_param[1] = new_beta2
        return new_param

    def _l2_norm(self, v1, v2):
        norm_square = 0
        for s1, s2 in zip(v1, v2):
            try:
                _ = iter(s1)
                norm_square += self._l2_norm(s1, s2)
            except TypeError:
                norm_square += (s1-s2)**2
        return norm_square**0.5

    def run(self, tol, print_iter_cycle=100):
        start_time = time.time()
        i = 0
        while True:
            i += 1
            last = self.optim_path[-1]
            self.one_iter_optimizer()
            new = self.optim_path[-1]

            diff = self._l2_norm(last, new)
            elap_time = time.time()-start_time
            if diff < tol:
                break
            
            if i%print_iter_cycle == 0:
                print("iteration", i, ", l2 norm diff:", diff, ", elapsed time:", elap_time//60,"min ", elap_time%60,"sec")

        print("iteration", i, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

    def construct_estimate_density(self, grid, MC_integral_num_iter=50):
        #variational parameter
        # 0      1      2                                  3                           4
        #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]
        optim_pt = self.optim_path[-1]
        p_y_post_vec = []
        for y in grid:
            p_y_post = 0
            # l = 0
            eqplV = optim_pt[2][0][0]/sum(optim_pt[2][0])
            product_1minusV = optim_pt[2][0][1]/sum(optim_pt[2][0])
            phi = gammavariate(optim_pt[0], 1/optim_pt[1])
            eqNy = sp_stats.norm.pdf(y, optim_pt[3][0][0], (optim_pt[3][0][1]+1/phi)**0.5)
            p_y_post += (eqplV*eqNy)

            for l in range(1, self.truncation_N-1):
                gamma_l1, gamma_l2 = optim_pt[2][l]
                eqplV = product_1minusV * gamma_l1/(gamma_l1+gamma_l2)
                product_1minusV *= (gamma_l2/(gamma_l1+gamma_l2))
                eqNy_sum = 0
                for _ in range(MC_integral_num_iter):
                    phi = gammavariate(optim_pt[0], 1/optim_pt[1])
                    eqNy = sp_stats.norm.pdf(y, optim_pt[3][l][0], (optim_pt[3][l][1]+1/phi)**0.5)
                    eqNy_sum += eqNy
                eqNy_mean = eqNy_sum / MC_integral_num_iter
                p_y_post += (eqplV*eqNy_mean)
            
            # l = N
            eqplV = product_1minusV
            eqNy_sum = 0
            for _ in range(MC_integral_num_iter):
                phi = gammavariate(optim_pt[0], 1/optim_pt[1])
                eqNy = sp_stats.norm.pdf(y, optim_pt[3][self.truncation_N-1][0], (optim_pt[3][self.truncation_N-1][1]+1/phi)**0.5)
                eqNy_sum += eqNy
            eqNy_mean = eqNy_sum / MC_integral_num_iter
            p_y_post += (eqplV*eqNy_mean)
            p_y_post_vec.append(p_y_post)
        return p_y_post_vec


if __name__=="__main__":
    data = []
    with open("data/hwk3-data.txt", "r", newline="\n") as f:
        for line in f:
            data.append(float(line))
    print(len(data)) #250

    truncation_N = 100
    initial = [1, 1,
               [[1,1] for _ in range(truncation_N-1)], 
               [[0,10] for _ in range(truncation_N)],
               [[1/truncation_N for _ in range(truncation_N)] for _ in range(len(data))]]
    vb_inst = Hw3Q1_MeanfieldVB(initial, data, truncation_N)
    vb_inst.run(tol=0.1)
    optim_pt = vb_inst.optim_path[-1]
    #variational parameter
    # 0      1      2                                  3                           4
    #[beta1, beta2, [(gamma_l1,gamma_l2),l=1,...,N-1], [(xi_l1,xi_l2), l=1,...,N], (pi_i1,...,pi_iN),i=1,...,n]
    
    print("data:", data[0:10])
    data_idx_list = [i for i in range(10)]
    print("phi~ gamma(",optim_pt[0],",", optim_pt[1],")")
    for i in data_idx_list:
        Li = None
        temp_pi = 0
        for l, pi_il in enumerate(optim_pt[4][i]):
            if temp_pi < pi_il:
                Li = l
                temp_pi = pi_il
        print("L"+str(i), " = ", Li, " with prob ", temp_pi)
        print("Z_L"+str(i)+"~ N(", optim_pt[3][Li][0], ",", optim_pt[3][Li][0], ")")

    grid = np.linspace(-7, 7, 50).tolist()
    density_pt_est_on_grid = vb_inst.construct_estimate_density(grid)
    density_true_on_grid = [0.2*sp_stats.norm.pdf(x, -5, 1)+0.5*sp_stats.norm.pdf(x, 0, 1)+0.3*sp_stats.norm.pdf(x, 3.5, 1) for x in grid]


    plt.hist(data, bins=50, density=True, color="orange", label='data')
    plt.plot(grid, density_pt_est_on_grid, color="blue", label='posterior')
    plt.plot(grid, density_true_on_grid, color="red", label='true')
    plt.legend()
    plt.show()