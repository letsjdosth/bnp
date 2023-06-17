import csv
from math import log
from random import seed, betavariate, gammavariate, choices, randint
import numpy as np
from scipy.stats import multivariate_normal as sp_mvn
from scipy.stats import invwishart as sp_inv_wishart
from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_Diag
from pyBayes.rv_gen_gamma import Sampler_InvWishart
import matplotlib.pyplot as plt

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
data_ozone = [z[0] for z in data_airquality]
data_solar_r = [z[1] for z in data_airquality]
data_wind = [z[2] for z in data_airquality]
data_temp = [z[3] for z in data_airquality]


class HW4DensityReg(MCMC_Gibbs):
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
        self.hyper_v0 = 3000
        self.hyper_Lambda0 = np.eye(4)

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
                # new_Sigma_l = self.sampler_invWishart.sampler_iter(1, v, lamb, "inv")[0]
                new_Sigma_l = sp_inv_wishart.rvs(v, lamb)
                
            if np.linalg.det(new_Sigma_l) == 0:
                new_Sigma_l += (np.eye(4)*self.nonsingular_adjust)
            
            #test
            new_Sigma_l = np.eye(4)*20 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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

class HW4DensityReg_Infer:
    def __init__(self, yx_data, MC_sample) -> None:
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        self.MC_sample = MC_sample
        self.yx_data = yx_data
        self.N = len(MC_sample[0][0])

    def _q_x(self, x, post_sample):
        q_weights = []
        for (mu_l, sigma_l), w_l in zip(post_sample[0], post_sample[1]):
            mu_l_x = mu_l[1:]
            sigma_l_x = sigma_l[1:, 1:]
            q_l = w_l * sp_mvn.pdf(x, mu_l_x, sigma_l_x)
            q_weights.append(q_l)
        return q_weights
    
    def _lambda_tau2(self, x, post_sample):
        lambda_tau2_tuples = []
        for (mu_l, sigma_l) in post_sample[0]:
            mu_y = mu_l[0]
            mu_x = mu_l[1:]
            sigma_yy = sigma_l[0,0]
            sigma_yx = sigma_l[0,1:]
            inv_sigma_xx = np.linalg.inv(sigma_l[1:,1:])
            lambda_l = mu_y + sigma_yx @ inv_sigma_xx @ (x - mu_x)
            tau2_l = sigma_yy - sigma_yx @ inv_sigma_xx @ np.transpose(sigma_yx)
            lambda_tau2_tuples.append((lambda_l, tau2_l))
        return lambda_tau2_tuples

    def y_given_x_G_rv_gen(self, x):
        #x in R3
        y_samples = []
        for post_sample in self.MC_sample:
            lambda_tau2_tuples = self._lambda_tau2(x, post_sample)
            qx = self._q_x(x, post_sample)
            
            chosen_l = choices([i for i in range(self.N)], weights=qx)[0]
            lambda_l, tau2_l = lambda_tau2_tuples[chosen_l]
            new_sample = sp_mvn.rvs(lambda_l, tau2_l)
            y_samples.append(new_sample)
        return y_samples
        
    def y_given_x_G_mean_quantile(self, x):
        samples = self.y_given_x_G_rv_gen(x)
        quant_est = np.quantile(samples, [0.025, 0.5, 0.975])
        mean_est = np.mean(samples)
        return mean_est, quant_est

    def yx_pair_joint_rv_gen(self):
        # sample
        #  0                           1                2                3     4      5
        # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
        joint_samples = []
        for post_sample in self.MC_sample:
            chosen_l = choices([i for i in range(self.N)], weights=post_sample[1])[0]
            mu_l, sigma_l = post_sample[0][chosen_l]
            new_sample = sp_mvn.rvs(mu_l, sigma_l)
            joint_samples.append(new_sample)
        return joint_samples
    
    def yx_pair_joint_rv_gen_for_plot(self):
        joint_samples = self.yx_pair_joint_rv_gen()
        return (np.transpose(joint_samples)).tolist()

def sample_reader_from_csv_hw4(filename):
    # sample
    #  0                           1                2                3     4      5
    # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
    pass

if __name__=="__main__":
    seed(20230614)
    np.random.seed(20230614)
    # sample
    #  0                           1                2                3     4      5
    # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
    trunc_N = 30

    run = True #No False now
    diag_inst = MCMC_Diag()
    if run:
        initial = [[(data_airquality[i], np.eye(4)) for i in range(trunc_N)], 
                [1/trunc_N for _ in range(trunc_N)], 
                [i%trunc_N for i in range(len(data_airquality))], 
                [0,0,0,0],
                np.eye(4),
                10]
        gibbs_inst = HW4DensityReg(initial, data_airquality, trunc_N)
        gibbs_inst.generate_samples(1000, print_iter_cycle=50)
        diag_inst.set_mc_sample_from_MCMC_instance(gibbs_inst)
        diag_inst.write_samples("hw4_samples")
    else:
        # reader is not implemented
        mc_samples = sample_reader_from_csv_hw4("hw4_samples")
        diag_inst.set_mc_samples_from_list(mc_samples)

    diag_inst.set_variable_names(["mu_sigma", "w", "L", "mu0", "tau0", "alpha"])
    diag_inst.burnin(200)

    ###########################
    # sample
    #  0                           1                2                3     4      5
    # [(mu_l,Sigma_l;l=1,2,...,N), (w_l;l=1,...,N), (L_i;i=1,...,n), mu_0, tau_0, alpha]
    diag_inst.show_traceplot((1,2),[3,5])
    diag_inst_alpha = MCMC_Diag()
    alpha_list = [[a] for a in diag_inst.get_specific_dim_samples(5)]
    diag_inst_alpha.set_mc_samples_from_list(alpha_list)
    diag_inst_alpha.set_variable_names(['alpha'])
    diag_inst_alpha.show_hist((1,1))

    diag_inst_w = MCMC_Diag()
    diag_inst_w.set_mc_samples_from_list(diag_inst.get_specific_dim_samples(1))
    diag_inst_w.set_variable_names(['w'+str(l) for l in range(1,trunc_N+1)])
    diag_inst_w.show_traceplot((1,4), [0,1,2,3])
    diag_inst_w.show_hist((1,4), [0,1,2,3])

    diag_inst_L = MCMC_Diag()
    diag_inst_L.set_mc_samples_from_list(diag_inst.get_specific_dim_samples(2))
    diag_inst_L.set_variable_names(['L'+str(i) for i in range(len(data_airquality))])
    diag_inst_L.show_traceplot((1,4), [0,1,2,3])
    diag_inst_L.show_hist((1,4), [0,1,2,3])

    for smpl in diag_inst.MC_sample[-4:-1]:
        print(smpl[2]) #Li

    diag_inst_mu_l = MCMC_Diag()
    diag_inst_mu_l.set_mc_samples_from_list([m[0][0] for m in diag_inst.get_specific_dim_samples(0)])
    diag_inst_mu_l.set_variable_names(['mu_'+str(l) for l in range(1,trunc_N+1)])
    diag_inst_mu_l.show_traceplot((1,4), [0,1,2,3])


    ###########################
    # Inference
    infer_inst = HW4DensityReg_Infer(data_airquality, diag_inst.MC_sample)
    # "1",41, 190, 7.4, 67, 5, 1
    # "2",36, 118, 8, 72, 5, 2
    # "3",12, 149, 12.6, 74, 5, 3
    # "4",18, 313, 11.5, 62, 5, 4
    # "7",23, 299, 8.6, 65, 5, 7
    # "8",19, 99, 13.8, 59, 5, 8
    print(infer_inst.y_given_x_G_mean_quantile(np.array([190, 7.4, 67])))
    print(infer_inst.y_given_x_G_mean_quantile(np.array([118, 8, 72])))
    print(infer_inst.y_given_x_G_mean_quantile(np.array([149, 12.6, 74])))
    print(infer_inst.y_given_x_G_mean_quantile(np.array([313, 11.5, 62])))
    print(infer_inst.y_given_x_G_mean_quantile(np.array([299, 8.6, 65])))
    print(infer_inst.y_given_x_G_mean_quantile(np.array([99, 13.8, 59])))

    joint_samples = infer_inst.yx_pair_joint_rv_gen_for_plot()
    #solar.R
    plt.scatter(data_solar_r, data_ozone, alpha=0.5, c='orange', label='data')
    plt.scatter(joint_samples[1], joint_samples[0], alpha=0.3, c='blue', label='posterior samples')
    plt.xlim(0, 350)
    plt.ylim(0, 200)
    plt.xlabel("solar.R")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()
    #wind
    plt.scatter(data_wind, data_ozone, alpha=0.5, c='orange', label='data')
    plt.scatter(joint_samples[2], joint_samples[0], alpha=0.3, c='blue', label='posterior samples')
    plt.xlim(0, 25)
    plt.ylim(0, 200)
    plt.xlabel("wind")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()
    #temp
    plt.scatter(data_temp, data_ozone, alpha=0.5, c='orange', label='data')
    plt.scatter(joint_samples[3], joint_samples[0], alpha=0.3, c='blue', label='posterior samples')
    plt.xlim(50, 100)
    plt.ylim(0, 200)
    plt.xlabel("temp")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()

    # > colMeans(without_na_airquality)
    #      Ozone    Solar.R       Wind       Temp      Month        Day 
    #  42.099099 184.801802   9.939640  77.792793   7.216216  15.945946
        
    # ozone vs solar_R
    solar_r_grid = np.linspace(0, 350, 50)
    y_vs_solar_r_lower = []
    y_vs_solar_r_mean = []
    y_vs_solar_r_upper = []
    for solar_r in solar_r_grid:
        quant = infer_inst.y_given_x_G_mean_quantile(np.array([solar_r, 10, 78]))
        y_vs_solar_r_mean.append(quant[0])
        y_vs_solar_r_lower.append(quant[1][0])
        y_vs_solar_r_upper.append(quant[1][2])
    plt.scatter(data_solar_r, data_ozone, alpha=0.5, c='orange', label='data')
    plt.plot(solar_r_grid, y_vs_solar_r_mean, c='red', label='posterior mean')
    plt.plot(solar_r_grid, y_vs_solar_r_lower, c='grey', label='95% CI')
    plt.plot(solar_r_grid, y_vs_solar_r_upper, c='grey')
    plt.xlim(0, 350)
    plt.ylim(0, 200)
    plt.xlabel("solar.R")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()

    
    # ozone vs Wind
    wind_grid = np.linspace(0, 25, 50)
    y_vs_wind_lower = []
    y_vs_wind_mean = []
    y_vs_wind_upper = []
    for wind in wind_grid:
        quant = infer_inst.y_given_x_G_mean_quantile(np.array([185, wind, 78]))
        y_vs_wind_mean.append(quant[0])
        y_vs_wind_lower.append(quant[1][0])
        y_vs_wind_upper.append(quant[1][2])
    plt.scatter(data_wind, data_ozone, alpha=0.5, c='orange', label='data')
    plt.plot(wind_grid, y_vs_wind_mean, c='red', label='posterior mean')
    plt.plot(wind_grid, y_vs_wind_lower, c='grey', label='95% CI')
    plt.plot(wind_grid, y_vs_wind_upper, c='grey')
    plt.xlim(0, 25)
    plt.ylim(0, 200)
    plt.xlabel("wind")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()
        
    # ozone vs Temp
    temp_grid = np.linspace(50, 100, 50)
    y_vs_temp_lower = []
    y_vs_temp_mean = []
    y_vs_temp_upper = []
    for temp in temp_grid:
        quant = infer_inst.y_given_x_G_mean_quantile(np.array([185, 10, temp]))
        y_vs_temp_mean.append(quant[0])
        y_vs_temp_lower.append(quant[1][0])
        y_vs_temp_upper.append(quant[1][2])
    plt.scatter(data_temp, data_ozone, alpha=0.5, c='orange', label='data')
    plt.plot(temp_grid, y_vs_temp_mean, c='red', label='posterior mean')
    plt.plot(temp_grid, y_vs_temp_lower, c='grey', label='95% CI')
    plt.plot(temp_grid, y_vs_temp_upper, c='grey')
    plt.xlim(50, 100)
    plt.ylim(0, 200)
    plt.xlabel("temp")
    plt.ylabel("ozone")
    plt.legend()
    plt.show()
    