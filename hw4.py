import csv
from math import log
from random import seed, betavariate, gammavariate, choices, randint
import numpy as np
from scipy.stats import multivariate_normal as sp_mvn
from scipy.stats import invwishart as sp_inv_wishart
from pyBayes.MCMC_Core import MCMC_Gibbs
from pyBayes.rv_gen_gamma import Sampler_InvWishart

# ### dataset::airquality
# A data frame with 153 observations on 6 variables.
# [,1]	Ozone	numeric	Ozone (ppb)
# [,2]	Solar.R	numeric	Solar R (lang)
# [,3]	Wind	numeric	Wind (mph)
# [,4]	Temp	numeric	Temperature (degrees F)
# [,5]	Month	numeric	Month (1--12)
# [,6]	Day	numeric	Day of month (1--31)
# Daily readings of the following air quality values for May 1, 1973 (a Tuesday) to September 30, 1973.
# Ozone: Mean ozone in parts per billion from 1300 to 1500 hours at Roosevelt Island
# Solar.R: Solar radiation in Langleys in the frequency band 4000–7700 Angstroms from 0800 to 1200 hours at Central Park
# Wind: Average wind speed in miles per hour at 0700 and 1000 hours at LaGuardia Airport
# Temp: Maximum daily temperature in degrees Fahrenheit at La Guardia Airport.

data_airquality = []
#0     1       2    3
#Ozone Solar.R Wind Temp
with open("data/airquality_data.csv", "r", newline="") as csv_f:
    csv_reader = csv.reader(csv_f)
    next(csv_reader)
    for row in csv_reader:
        data_airquality.append((int(row[1]), int(row[2]), float(row[3]), int(row[4])))
# print(data_airquality)
# print(len(data_airquality)) #111

class HW4DensityEst(MCMC_Gibbs):
    def __init__(self, initial, yx_data, N_truncation):
        self.MC_sample = [initial]
        self.yx_data = yx_data

        #samplers
        seed(20230614)
        np.random.seed(20230614)
        self.sampler_np = np.random.default_rng(20230614)
        self.sampler_invWishart = Sampler_InvWishart(20230614)

        #setting
        self.N = N_truncation
        self.nonsingular_adjust = 0.001

        #hyperparams
        self.hyper_v0 = 100
        self.hyper_Lambda0 = np.eye(4)*10

        self.hyper_alpha_a = 1
        self.hyper_alpha_b = 1
        self.hyper_m_mu = np.zeros(4)
        self.hyper_s_mu = np.eye(4)*10
        self.hyper_tau0_a = 10
        self.hyper_tau0_b = np.eye(4)*10
    
    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_mu_l_sigma_l(new)
        new = self.full_conditional_sampler_w_l(new)
        new = self.full_conditional_sampler_Li(new)
        new = self.full_conditional_sampler_mu0_tau0(new)
        new = self.full_conditional_sampler_alpha(new)
        self.MC_sample.append(new)
    
    def full_conditional_sampler_mu_l_sigma_l(self, last_param):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        new_sample = last_param #pointer
        
        inv_tau_0 = np.linalg.inv(last_param[4])
        prod_inv_tau_0_mu_0 = inv_tau_0 @ last_param[3]
        for j, (_, Sigma_l) in enumerate(last_param[0]):
            yx_list_j = []
            for yx_i, L_i in zip(self.yx_data, last_param[2]):
                if L_i==j:
                    yx_list_j.append(np.array(yx_i))

            #update mu_l
            if len(yx_list_j)==0:
                new_mu_l = self.sampler_np.multivariate_normal(last_param[3], last_param[4])
            else:
                inv_Sigma_l = np.linalg.inv(Sigma_l)
                mu_l_precision = inv_tau_0 + inv_Sigma_l*len(yx_list_j)
                mu_l_cov = np.linalg.inv(mu_l_precision)

                mu_l_mean = mu_l_cov @ (prod_inv_tau_0_mu_0 + inv_Sigma_l@np.sum(yx_list_j, axis=0))
                new_mu_l = self.sampler_np.multivariate_normal(mu_l_mean, mu_l_cov)

            #update Sigma_l
            if len(yx_list_j)== 0:
                # new_Sigma_l = self.sampler_invWishart.sampler_iter(1, self.hyper_v0, self.hyper_Lambda0, "inv")[0]
                new_Sigma_l = sp_inv_wishart.rvs(self.hyper_v0, self.hyper_Lambda0)
            else:
                v = self.hyper_v0 + len(yx_list_j)
                lamb = self.hyper_Lambda0
                for yx in yx_list_j:
                    lamb += np.outer(yx - new_mu_l, yx - new_mu_l)
                new_Sigma_l = self.sampler_invWishart.sampler_iter(1, v, lamb, "inv")[0]
                # new_Sigma_l = sp_inv_wishart.rvs(v, lamb)
                
            if np.linalg.det(new_Sigma_l) == 0:
                new_Sigma_l += (np.eye(4)*self.nonsingular_adjust)
            
            #test
            new_Sigma_l = np.eye(4)*100 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            new_sample[0][j] = (new_mu_l, new_Sigma_l)
        return new_sample
    
    def _Li_counter(self, Li_vec):
        count_result = [0 for _ in range(self.N)]
        for l in Li_vec:
            count_result[l] += 1
        return count_result

    def full_conditional_sampler_w_l(self, last_param):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        new_sample = last_param #pointer
        new_w = []
        Li_counter = self._Li_counter(last_param[2])
        left_stick = 1
        for j in range(self.N-1):
            beta_al = 1 + Li_counter[j]
            beta_bl = last_param[5] + sum(Li_counter[(j+1):])
            Z_l = betavariate(beta_al, beta_bl)
            new_w_j = Z_l * left_stick
            new_w.append(new_w_j)
            left_stick *= (1-Z_l)
        
        w_N = 1.0-sum(new_w)
        if w_N <= 0:
            w_N = 1e-16
        new_w.append(w_N)
        
        assert len(new_w)==self.N
        new_sample[1] = new_w
        return new_sample
    
    def full_conditional_sampler_Li(self, last_param):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        new_sample = last_param #pointer
        for i, yx_i in enumerate(self.yx_data):
            weight_i = []
            for (mu_l, sigma_l), w_l in zip(last_param[0], last_param[1]):
                pdf_val = sp_mvn.pdf(yx_i, mu_l, sigma_l, allow_singular=True) #hmm
                weight = w_l * pdf_val
                weight_i.append(weight)
            new_Li = choices([k for k in range(self.N)], weights=weight_i)[0]
            new_sample[2][i] = new_Li
        return new_sample
    
    def full_conditional_sampler_mu0_tau0(self, last_param):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        
        new_sample = last_param #pointer
        
        #update mu0
        inv_tau_0 = np.linalg.inv(last_param[4])
        inv_s_mu = np.linalg.inv(self.hyper_s_mu)
        mu_0_precision = inv_s_mu + inv_tau_0*self.N
        mu_0_cov = np.linalg.inv(mu_0_precision)
        sum_mu_l = np.zeros(4)
        for mu_l, _ in last_param[0]:
            sum_mu_l += mu_l
        mu_0_mean = mu_0_cov @ (inv_s_mu@self.hyper_m_mu + inv_tau_0 @ sum_mu_l)
        new_mu0 = self.sampler_np.multivariate_normal(mu_0_mean, mu_0_cov)

        tau_0_v = self.hyper_tau0_a + self.N
        tau_0_lamb = self.hyper_tau0_b
        for mu_l, _ in last_param[0]:
            tau_0_lamb += np.outer(mu_l - new_mu0, mu_l - new_mu0)
            
        # new_tau0 = self.sampler_invWishart.sampler_iter(1, tau_0_v, tau_0_lamb, "inv")[0]
        new_tau0 = sp_inv_wishart.rvs(tau_0_v, tau_0_lamb)
        
        if np.linalg.det(new_tau0) == 0:
            new_tau0 += (np.eye(4)*self.nonsingular_adjust)
        
        new_sample[3] = new_mu0
        new_sample[4] = new_tau0
        # new_sample[4] = np.eye(4)*10

        return new_sample
    
    def full_conditional_sampler_alpha(self, last_param):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        new_sample = last_param #pointer
        
        w_N = last_param[1][-1]
        shape = 1/(self.hyper_alpha_b - log(w_N))
        new_alpha = gammavariate(self.N + self.hyper_alpha_a - 1, shape)
        new_sample[5] = new_alpha
        return new_sample
    
if __name__=="__main__":
    seed(20230614)
    # sample
    #  0                           1                2                3     4      5
    # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
    trunc_N = 100
    initial = [[(data_airquality[i], np.eye(4)) for i in range(trunc_N)], 
               [1/trunc_N for _ in range(trunc_N)], 
               [i%trunc_N for i in range(len(data_airquality))], 
               [0,0,0,0], 
               np.eye(4), 
               10]
    gibbs_inst = HW4DensityEst(initial, data_airquality, trunc_N)
    gibbs_inst.generate_samples(50)
    for smpl in gibbs_inst.MC_sample[-3:-1]:
        print(smpl[5]) #alpha
        print(smpl[2]) #Li
        print([x[0] for x in smpl[0]])
        print([round(y,4) for y in smpl[1]]) #w1
        print(smpl[0][smpl[2][0]])