from math import log, exp, inf, pi, floor
from random import seed, randint, uniform
from statistics import variance
from functools import partial, lru_cache

import numpy as np
import scipy.stats as sp_stats
import matplotlib.pyplot as plt

from pyBayes.MCMC_Core import MCMC_Gibbs, MCMC_MH, MCMC_Diag
from pyBayes.util_MCMC_proposal import unif_proposal_log_pdf, unif_proposal_sampler, unif_spherical_proposal_sampler, unif_spherical_proposal_log_pdf
from pyBayes.ts_arma_spectral_density import ARMA

class Bernstein_Spec(MCMC_Gibbs):
    def __init__(self, initial, L_trunctation, fourier_freq_on01_seq: list, periodogram_seq: list):
        super().__init__(initial)
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]

        self.ffreq = fourier_freq_on01_seq
        self.periodogram = periodogram_seq

        self.L_trunctation = L_trunctation
        
        def unif01_log_pdf(u):
            return 0
        self.hyper_g0 = unif01_log_pdf
        self.hyper_M = 1

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler_k(new)
        new = self.full_conditional_sampler_tau(new)
        new = self.full_conditional_sampler_Z_block(new)
        new = self.full_conditional_sampler_V_block(new)
        self.MC_sample.append(new)
    
    def _p_vec_constructor(self, V_vec): #1st-class
        "return: [p1, p2, ..., pL, p0]"
        p_vec = []
        prod_until_last = 1
        for v in V_vec:
            p_vec.append(prod_until_last * v)
            prod_until_last *= (1-v)
        p_vec.append(prod_until_last)
        return p_vec

    def _zp_tuple_constructor(self, Z_vec, V_vec, sorted_by_Z: bool): #1st-class
        "returned list[tuple(z,p)] are sorted by z in the ascending order"
        p_vec = self._p_vec_constructor(V_vec)
        zp_vec = []
        for z, p in zip(Z_vec, p_vec):
            zp_vec.append((z, p))
        if sorted_by_Z:
            zp_vec.sort(key=lambda x: x[0])
        return zp_vec

    def _w_k_vec_constructor(self, k, Z_vec, V_vec): #1st-class
        sorted_zp_vec = self._zp_tuple_constructor(Z_vec, V_vec, sorted_by_Z=True)
        w_k_vec = []
        now_j = 1
        now_p = 0
        for z, p in sorted_zp_vec:
            while z > (now_j/k):
                w_k_vec.append(now_p)
                now_j += 1
                now_p = 0
            now_p += p
        w_k_vec.append(now_p)

        if len(w_k_vec) < k:
            i = k - len(w_k_vec)
            for _ in range(i):
                w_k_vec.append(0)

        try:
            assert len(w_k_vec) == k #for debug
        except AssertionError:
            raise ValueError(str(len(w_k_vec))+", k="+str(k)+" are different. requirement: k < L")
        return w_k_vec

    def _new_w_k_vec_constructor(self, k, Z_vec, V_vec): #1st-class
        unsorted_zp_vec = self._zp_tuple_constructor(Z_vec, V_vec, sorted_by_Z=False)
        w_k_vec = [0 for _ in range(k)]
        
        for z, p in unsorted_zp_vec:
            j_minus_1 = floor(z*k)
            w_k_vec[j_minus_1] += p
        return w_k_vec

    def _bernstein_approxed_spec_density(self, fourier_freq_on01_seq, k, tau, Z_vec, V_vec): #1st-class
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        f_vec = []
        w_k_vec = self._w_k_vec_constructor(k, Z_vec, V_vec)
        for w in fourier_freq_on01_seq: #0<=w<=1
            f_w = 0
            for i, w_jk in enumerate(w_k_vec):
                j = i + 1
                f_w += (w_jk * sp_stats.beta.pdf(w, j, k-j+1))
            f_w *= tau
            if f_w == 0:
                f_w = 0.000000000000001 #problematic
            f_vec.append(f_w)
        return f_vec

    def _log_whittle_likelihood(self, fourier_freq_on01_seq, k, tau, Z_vec, V_vec):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        f_vec = self._bernstein_approxed_spec_density(fourier_freq_on01_seq, k, tau, Z_vec, V_vec)
        log_whittle_likelihood = 0
        for f, u in zip(f_vec, self.periodogram):
            log_whittle_likelihood += (-log(f) - u/f)
        return log_whittle_likelihood
    
    def _log_prior_density_k(self, k: int, upper): # can override
        if 1 <= k <= upper:
            return -log(upper)
        else:
            return -inf

    def full_conditional_sampler_k(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        
        upper_lim = self.L_trunctation-1 #tunning parameter
        def log_target_pmf(smpl):
            k = smpl[0]
            target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=k, tau=last_param[1], Z_vec=last_param[2], V_vec=last_param[3])
            target += self._log_prior_density_k(k, upper_lim)
            return target
        def proposal_sampler(from_smpl):
            k = from_smpl[0]
            new_k = None
            if k == 1:
                new_k = randint(1,2)
            elif k == upper_lim:
                new_k = randint(upper_lim-1, upper_lim)
            else:
                new_k = randint(k-1, k+1)
            return [new_k]
        def log_proposal_pmf(from_smpl, to_smpl):
            k = from_smpl[0]
            if k == 1 or k == upper_lim:
                return -log(2)
            else:
                return -log(3)
        
        mh_inst_for_k = MCMC_MH(log_target_pmf, log_proposal_pmf, proposal_sampler, [last_param[0]])
        mh_inst_for_k.generate_samples(2, verbose=False)
        new_k = mh_inst_for_k.MC_sample[-1][0]
        
        new_sample = last_param #pointer
        new_sample[0] = new_k
        return new_sample

    def _log_prior_density_tau(self, tau): # can override
        gamma_shape = 0.01
        gamma_rate = 0.01
        return sp_stats.gamma.logpdf(tau, a=gamma_shape, scale=1/gamma_rate)

    def full_conditional_sampler_tau(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        def log_target_pdf(smpl):
            tau = smpl[0]
            target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=last_param[0], tau=tau, Z_vec=last_param[2], V_vec=last_param[3])
            target += self._log_prior_density_tau(tau)
            return target
        window = 1
        proposal_sampler = partial(unif_proposal_sampler, lower_lim=0, upper_lim=inf, window=window)
        proposal_log_pdf = partial(unif_proposal_log_pdf, lower_lim=0, upper_lim=inf, window=window)

        mh_inst_for_tau = MCMC_MH(log_target_pdf, proposal_log_pdf, proposal_sampler, [last_param[1]])
        mh_inst_for_tau.generate_samples(2, verbose=False)
        new_tau = mh_inst_for_tau.MC_sample[-1][0]

        new_sample = last_param
        new_sample[1] = new_tau
        return new_sample


    def full_conditional_sampler_Z(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        
        new_sample = last_param #pointer
        
        window = 0.4
        proposal_sampler = partial(unif_proposal_sampler, lower_lim=0, upper_lim=1, window=window)
        proposal_log_pdf = partial(unif_proposal_log_pdf, lower_lim=0, upper_lim=1, window=window)
            
        for i in range(self.L_trunctation+1):
            def log_target_pdf(smpl): #smpl=[z_i]
                z_i = smpl[0]
                eval_z_vec = [z for z in new_sample[2]]
                eval_z_vec[i] = z_i
                target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=last_param[0], tau=last_param[1], Z_vec=eval_z_vec, V_vec=last_param[3])
                target += self.hyper_g0(z_i)
                return target
            mh_inst_for_z_i = MCMC_MH(log_target_pdf, proposal_log_pdf, proposal_sampler, [last_param[2][i]])
            mh_inst_for_z_i.generate_samples(2, verbose=False)
            new_z_i = mh_inst_for_z_i.MC_sample[-1][0]
            new_sample[2][i] = new_z_i
        
        return new_sample

    def full_conditional_sampler_Z_block(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        
        new_sample = last_param #pointer
        
        proposal_log_pdf = unif_spherical_proposal_log_pdf        
        def proposal_sampler(from_smpl): #smpl=Z vec
            new = []
            for l, z_l in enumerate(from_smpl):
                # window = 2*(l+1)/(l+1+2*(len(self.periodogram)*2)**0.5)
                window = 0.1
                new_z_l = (unif_spherical_proposal_sampler([z_l], 0, 1, window))[0]
                new.append(new_z_l)
            return new
            
        def log_target_pdf(smpl): #smpl=Z vec
            eval_z_vec = smpl
            target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=last_param[0], tau=last_param[1], Z_vec=eval_z_vec, V_vec=last_param[3])
            for z_i in eval_z_vec:
                target += self.hyper_g0(z_i)
            return target

        mh_inst_for_z_i = MCMC_MH(log_target_pdf, proposal_log_pdf, proposal_sampler, last_param[2])
        mh_inst_for_z_i.generate_samples(2, verbose=False)
        new_z = mh_inst_for_z_i.MC_sample[-1]
        new_sample[2] = new_z
        return new_sample


    def full_conditional_sampler_V(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        
        new_sample = last_param #pointer

        for i in range(self.L_trunctation):
            window = 2*(i+1)/(i+1+2*(len(self.periodogram)*2)**0.5)
            proposal_sampler = partial(unif_proposal_sampler, lower_lim=0, upper_lim=1, window=window)
            proposal_log_pdf = partial(unif_proposal_log_pdf, lower_lim=0, upper_lim=1, window=window)

            def log_target_pdf(smpl): #smpl=[v_i]
                v_i = smpl[0]
                eval_v_vec = [v for v in new_sample[3]]
                eval_v_vec[i] = v_i
                target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=last_param[0], tau=last_param[1], Z_vec=last_param[2], V_vec=eval_v_vec)
                target += (log(self.hyper_M) + (self.hyper_M-1)*log(1-v_i))
                return target
            mh_inst_for_v_i = MCMC_MH(log_target_pdf, proposal_log_pdf, proposal_sampler, [last_param[3][i]])
            mh_inst_for_v_i.generate_samples(2, verbose=False)
            new_v_i = mh_inst_for_v_i.MC_sample[-1][0]
            new_sample[3][i] = new_v_i
        return new_sample

    def full_conditional_sampler_V_block(self, last_param):
        #param
        #  0  1    2                      3
        # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
        #update new
        
        new_sample = last_param #pointer
       
        proposal_log_pdf = unif_spherical_proposal_log_pdf        
        def proposal_sampler(from_smpl): #smpl=V vec
            new = []
            for l, v_l in enumerate(from_smpl):
                window = 2*(l+1)/(l+1+2*(len(self.periodogram)*2)**0.5)
                new_v_l = (unif_spherical_proposal_sampler([v_l], 0, 1, window))[0]
                new.append(new_v_l)
            return new
    
        def log_target_pdf(smpl): #smpl=V vec
            eval_v_vec = smpl
            target = self._log_whittle_likelihood(fourier_freq_on01_seq=self.ffreq, k=last_param[0], tau=last_param[1], Z_vec=last_param[2], V_vec=eval_v_vec)
            for v_l in eval_v_vec:
                target += (log(self.hyper_M) + (self.hyper_M-1)*log(1-v_l))
            return target
        
        mh_inst_for_v_i = MCMC_MH(log_target_pdf, proposal_log_pdf, proposal_sampler, last_param[3])
        mh_inst_for_v_i.generate_samples(2, verbose=False)
        new_v = mh_inst_for_v_i.MC_sample[-1]
        new_sample[3] = new_v
        return new_sample

if __name__=="__main__":
    seed(20230529)
    np.random.seed(20230529)
    arma_inst2 = ARMA(1, seed=20230529)
    arma_inst2.set_ar_coeff_from_reciprocal_roots([(0.8, 1), (0.8, -1)], polar_radi_angle=True)
    # arma_inst2.set_ar_coeff_from_reciprocal_roots([(0.7, 2.7), (0.7, -2.7)], polar_radi_angle=True)
    print("ar2 coeff", arma_inst2.ar_coeff)
    print("ar2 polyroot", arma_inst2.ar_polyonmial_root(reciprocal=True))

    spec2, grid2 = arma_inst2.spectral_density(512, domain_0pi=False)
    x, _ = arma_inst2.generate_random_path(512)
            
    # plt.plot(x)
    # plt.title("ar2: one realization")
    # plt.show()
    

    x_fft = np.fft.rfft(x)
    periodogram_x = [f*f.conjugate()/(512*2*np.pi) for f in x_fft]
    # arma_inst2.plot_spectral_density(512, domain_0pi=False, show=False)
    # plt.plot(grid2, periodogram_x)
    # plt.title("ar2: spectra")
    # plt.show()



    grid2_on_01 = [x*2 for x in grid2]
    #param
    #  0  1    2                      3
    # [k, tau, [Z1, Z2, ..., ZL, Z0], [V1, V2, ..., VL]]
    est_L = 20 #max(20, 512^(1/3)=8)
    print("sample var/(2pi):", variance(x)/(2*pi))
    # est_initial = [5, variance(x), [uniform(0,1) for _ in range(est_L+1)], [0.5 for _ in range(est_L)]]
    
    #last iter endpoint
    est_initial = [13, 0.32948520373276974, 
                [0.23488387997338106, 0.29550024299873995, 0.2170532762109657, 0.9519122855562393, 0.05916340248861795, 0.8523653714552344, 0.09501693374342278, 0.622904990091043, 0.8825931874313059, 0.5597808540963337, 0.6376517711626044, 0.334149643525184, 0.18735810041138648, 0.5016993893791308, 0.19844622619833235, 0.06368287002603722, 0.7496273018570578, 0.5, 0.5, 0.5, 0.5], 
                [0.23015169047404696, 0.7947411380367597, 0.18587052904123152, 0.02040788581143469, 0.4596901944732255, 0.1286425696228655, 0.15455789292621458, 0.8542513473949733, 0.29829528692253493, 0.2515790025775968, 0.7685330718462459, 0.499951290751285, 0.9835679777557805, 0.32855661227954913, 0.81271704240959, 0.6619543277119133, 0.971678772182387, 0.043583474919033034, 0.7527401173434533, 0.3162824768569266]] 
    est_inst = Bernstein_Spec(est_initial, est_L, grid2_on_01, periodogram_x)
    est_inst.generate_samples(5000, print_iter_cycle=100)
    
    
    plt.plot(grid2, periodogram_x, color='gray')
    for i in range(1, 100, 2):
        # print(est_inst.MC_sample[-i])
        est_f_on_grid = est_inst._bernstein_approxed_spec_density(
            grid2_on_01, est_inst.MC_sample[-i][0], est_inst.MC_sample[-i][1], est_inst.MC_sample[-i][2], est_inst.MC_sample[-i][3])
        plt.plot(grid2, est_f_on_grid, alpha=0.2, color='green')
    arma_inst2.plot_spectral_density(512, domain_0pi=False, show=False)
    plt.show()

    diag_inst1 = MCMC_Diag()
    diag_inst1.set_mc_sample_from_MCMC_instance(est_inst)
    diag_inst1.set_variable_names(["k", "tau", "Z", "V"])
    diag_inst1.write_samples("cgr_100iter_block_ver")
    diag_inst1.show_traceplot((1,2), [0,1])
    diag_inst_Z = MCMC_Diag()
    diag_inst_Z.set_mc_samples_from_list(diag_inst1.get_specific_dim_samples(2))
    diag_inst_Z.set_variable_names(["Z"+str(i+1) for i in range(est_L)]+["Z0"])
    diag_inst_Z.show_traceplot((2,4), [0,1,2,3,4,5,6,7])
    
    diag_inst_V = MCMC_Diag()
    diag_inst_V.set_mc_samples_from_list(diag_inst1.get_specific_dim_samples(3))
    diag_inst_V.set_variable_names(["V"+str(i+1) for i in range(est_L)])
    diag_inst_V.show_traceplot((2,4),[0,1,2,3,4,5,6,7])
