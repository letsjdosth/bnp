from math import log
import numpy as np

class InfomationCriteria:
    def __init__(self, MC_sample, data):
        self.MC_sample = np.array(MC_sample)
        self.num_dim = len(MC_sample[0])
        self.data = data

    def _dic_log_likelihood_given_full_data(self, param_vec):
        # Inherit this class and implement this method when using DIC
        return 0
    
    def _waic_regular_likelihood_given_one_data_pt(self, param_vec, data_point_vec):
        # Inherit this class and implement this method when using DIC
        return 0
    



    def _dic_deviance_D(self, param_vec):
        # -2 * log_likelihood
        return self._dic_log_likelihood_given_full_data(param_vec) * (-2)

    def DIC(self, pt_estimate_method = "mean"): 
        # Deviance Information Criterion
        # Caution: need to test. this function is not tested

        pt_est = None
        if pt_estimate_method == "mean":
            pt_est = np.mean(self.MC_sample, axis=0)
        else:
            raise ValueError("only mean pt_estimate_method is implemented now :D")

        deviance_at_pt_est = self._dic_deviance_D(pt_est)
        deviances_at_all_samples = [self._dic_deviance_D(x) for x in self.MC_sample]
        expected_deviance = np.mean(deviances_at_all_samples)
        return expected_deviance * 2 - deviance_at_pt_est


    def DIC_alt(self, pt_estimate_method = "mean"): 
        # Deviance Information Criterion (alternative version, using var)
        # Caution: need to test. this function is not tested

        pt_est = None
        if pt_estimate_method == "mean":
            pt_est = np.mean(self.MC_sample, axis=0)
        else:
            raise ValueError("only mean pt_estimate_method is implemented now :D")

        deviance_at_pt_est = self._dic_deviance_D(pt_est)
        log_likelihood_vec = [self._dic_log_likelihood_given_full_data(x) for x in self.MC_sample]
        p_alt = 2*np.var(log_likelihood_vec)
        return deviance_at_pt_est + 2*p_alt


    def WAIC(self, pt_estimate_method = "mean"):
        # Watanabe-Akaike Information Criterion
        # Caution: need to test. this function is not tested
       
        lppd = 0
        p_waic1 = 0
        num_param_sample = self.MC_sample.shape[0]
        for y in self.data:
            sum_likelihood_for_y = 0
            sum_log_likelihood_for_y = 0
            for param in self.MC_sample:
                likelihood_at_param = self._waic_regular_likelihood_given_one_data_pt(param, y)
                sum_likelihood_for_y += likelihood_at_param
                sum_log_likelihood_for_y += log(likelihood_at_param)
            p_waic1 += (log(sum_likelihood_for_y/num_param_sample) - sum_log_likelihood_for_y/num_param_sample)
            lppd += log(sum_likelihood_for_y/num_param_sample)
        p_waic1 *= 2

        waic = 2* p_waic1 - 2* lppd # we can cancel lppd and one lppd term in p_waic1
        return waic


    def WAIC_alt(self, pt_estimate_method = "mean"):
        # Watanabe-Akaike Information Criterion (using var)
        # Caution: need to test. this function is not tested
       
        lppd = 0
        p_waic2 = 0
        num_param_sample = self.MC_sample.shape[0]
        for y in self.data:
            sum_likelihood_for_y = 0
            vec_log_likelihood_for_y = []
            for param in self.MC_sample:
                likelihood_at_param = self._waic_regular_likelihood_given_one_data_pt(param, y)
                sum_likelihood_for_y += likelihood_at_param
                vec_log_likelihood_for_y.append(log(likelihood_at_param))
            p_waic2 += np.var(vec_log_likelihood_for_y)
            lppd += log(sum_likelihood_for_y/num_param_sample)

        waic = 2* p_waic2 - 2* lppd
        return waic

