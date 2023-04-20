import csv
import time
from random import seed, normalvariate

import numpy as np
import matplotlib.pyplot as plt

from info_criteria import InfomationCriteria


class LM_base:
    def __init__(self, response_vec, design_matrix, rnd_seed=None) -> None:
        self.x = design_matrix
        self.y = response_vec

        self.num_data = design_matrix.shape[0]
        self.dim_beta = design_matrix.shape[1]

        self.MC_sample = []

        if rnd_seed:
            seed(rnd_seed)
        self.np_rng = np.random.default_rng()

        self.xtx = np.transpose(self.x) @ self.x
        self.xty = np.transpose(self.x) @ self.y
    
    def deep_copier(self, x_iterable) -> list:
        rep_x_list = []
        for x in x_iterable:
            try:
                _ = iter(x)
                rep_x_list.append(self.deep_copier(x))
            except TypeError:
                rep_x_list.append(x)
        return rep_x_list

    def sampler(self, **kwargs):
        pass

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        start_time = time.time()
        for i in range(1, num_samples):
            self.sampler(iter_idx=i)
            
            if i==100 and verbose:
                elap_time_head_iter = time.time()-start_time
                estimated_time = (num_samples/100)*elap_time_head_iter
                print("estimated running time: ", estimated_time//60, "min ", estimated_time%60, "sec")

            if i%print_iter_cycle == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", i, "/", num_samples)
            elif i%print_iter_cycle == 0 and verbose and pid is None:
                print("iteration", i, "/", num_samples)
        elap_time = time.time()-start_time
        
        if pid is not None and verbose:
            print("pid:",pid, "iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose:
            print("iteration", num_samples, "/", num_samples, " done! (elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")

    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)

    def _component_sampler_for_big_coeff_vec(self, m_array, D_inv, sigma2):
        # for full conditional sampler for a big beta
        cov_mat = np.linalg.inv(D_inv + self.xtx/sigma2)
        mean_vec = cov_mat @ (D_inv@m_array + self.xty/sigma2)
        new_coeff_vec = self.np_rng.multivariate_normal(mean_vec, cov_mat)
        return new_coeff_vec


class InfomationCriteria_for_LM(InfomationCriteria):
    def __init__(self, response_vec, design_matrix, beta_samples, sigma2_samples):
        self.beta_samples = beta_samples
        self.sigma2_samples = sigma2_samples
        self.y = response_vec
        self.x = design_matrix
        self.data = [(y,x) for y, x in zip(self.y, self.x)]
        self.MC_sample = np.array([(s,b) for s, b in zip(self.sigma2_samples, self.beta_samples)], dtype=object)

    def _dic_log_likelihood_given_full_data(self, param_vec):
        sigma2 = param_vec[0]
        beta = param_vec[1]
        n = len(self.x)
        residual = self.y-(self.x@beta)
        exponent = np.dot(residual, residual) / (-2*sigma2)
        return (-n/2)*np.log(sigma2) + exponent
    
    def _waic_regular_likelihood_given_one_data_pt(self, param_vec, data_point_vec):
        sigma2 = param_vec[0]
        beta = param_vec[1]
        y, x = data_point_vec
        residual = y-(x@beta)
        exponent = np.dot(residual, residual) / (-2*sigma2)
        return sigma2**(-1/2) * np.exp(exponent)

class Regression_Model_Checker:
    def __init__(self, response_vec, design_mat, beta_samples, sigma2_samples):
        self.y = response_vec
        self.x = design_mat
        self.beta_samples = beta_samples
        self.sigma2_samples = sigma2_samples

        self.mean_beta = np.mean(self.beta_samples, axis=0)
        self.mean_sigma2 = np.mean(self.sigma2_samples)
    
        self.fitted = self.x @ self.mean_beta
        self.residuals = self.y - self.fitted
        self.standardized_residuals = self.residuals/(self.mean_sigma2**0.5)

    def show_residual_plot(self, show=True):
        x_axis = self.fitted
        y_axis = self.residuals
        plt.plot(x_axis, y_axis, 'bo')
        plt.xlabel("y-fitted")
        plt.ylabel("standardized residual")
        plt.title("residual plot")
        plt.axhline(0)
        plt.axhline(1.96, linestyle='dashed')
        plt.axhline(-1.96, linestyle='dashed')
        if show:
            plt.show()

    def show_residual_normalProbplot(self, show=True):
        from scipy.stats import probplot
        probplot(self.residuals, plot=plt)
        plt.xlabel("theoretical quantiles")
        plt.ylabel("observed values")
        plt.title("normal probability plot")

        if show:
            plt.show()

    def show_posterior_predictive_at_new_point(self, design_row, reference_response_val=None, show=True, color=None, x_lab=None, x_lim=None):
        predicted = []
        for beta, sigma2 in zip(self.beta_samples, self.sigma2_samples):
            new_y = (design_row @ beta) + normalvariate(0, sigma2**0.5)
            predicted.append(new_y)
        
        if color is None:
            plt.hist(predicted, bins=50, density=True, histtype="step")
            if reference_response_val is not None:
                plt.axvline(reference_response_val)
        else:
            desig_color = "C"+str(color%10)
            plt.hist(predicted, bins=50, density=True, histtype="step", color=desig_color)
            if reference_response_val is not None:
                plt.axvline(reference_response_val, color=desig_color)
        if x_lab:
            plt.xlabel(x_lab)
        else:
            plt.xlabel("predicted at:"+str(design_row))
        if x_lim:
            plt.xlim(x_lim)
        if show:
            plt.show()        

    def show_posterior_predictive_at_given_data_point(self, data_idx, show=True, x_lab=None):
        design_row = self.x[data_idx,:]
        ref_y = self.y[data_idx]
        self.show_posterior_predictive_at_new_point(design_row, ref_y, show, color=data_idx, x_lab=x_lab)



