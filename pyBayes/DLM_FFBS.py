import numpy as np

from DLM_Core import DLM_full_model, DLM_univariate_y_without_V_in_D0

class DLM_BackwardSampling_Smoothing:
    def __init__(self, dlm_fitter_filtered: DLM_full_model | DLM_univariate_y_without_V_in_D0, seed_val=None):
        self.filtering_inst = dlm_fitter_filtered
        self.DLM_model = self.filtering_inst.DLM_model

        self.backward_smoothing_samples = []
        self.data_T = self.filtering_inst.y_len
        self.data_n = len(self.filtering_inst.m0)

        if seed_val is not None:
            self.np_random_inst = np.random.default_rng(seed_val)
        else:
            self.np_random_inst = np.random.default_rng(seed_val)

    def run_bs(self): # always from last T
        self.backward_smoothing_samples = []
        a, R = self.filtering_inst.get_prior_a_R()
        m, C = self.filtering_inst.get_posterior_m_C()

        theta_T = self.np_random_inst.multivariate_normal(np.reshape(m[-1],(2,)), C[-1])
        self.backward_smoothing_samples.append(np.reshape(theta_T,(2,1)))


        for t in range(self.data_T-1, 0, -1):
            Bt = C[t-1]@ np.transpose(self.DLM_model.G_sys_eq_transition[t]) @ np.linalg.inv(R[t])
            It = np.reshape(m[t-1],(self.data_n,)) + Bt @ (np.reshape(self.backward_smoothing_samples[-1],(self.data_n,)) - np.reshape(a[t],(self.data_n,)))
            Lt = C[t-1] - Bt @ R[t] @ np.transpose(Bt)
            theta_t = self.np_random_inst.multivariate_normal(It, Lt)
            self.backward_smoothing_samples.append(np.reshape(theta_t, (self.data_n,1)))
            if t == 1:
                #for time 0,
                theta_t = self.np_random_inst.multivariate_normal(It, Lt)
                self.backward_smoothing_samples.append(np.reshape(theta_t, (self.data_n,1)))

        #at 0, using prior, #<<<< need to check
        # C0 = self.filtering_inst.C0
        # m0 = self.filtering_inst.m0
        # if C0 is None:
        #     C0 = self.filtering_inst.C0st @ self.filtering_inst.S_precision_rate[-1] / (self.filtering_inst.n_precision_shape[-1] - 0.5) #need to check
        # B0 = C0 @ np.transpose(self.DLM_model.G_sys_eq_transition[0]) @ np.linalg.inv(R[0])
        # I0 = m0 + B0 @ np.transpose(self.backward_smoothing_samples[-1] - np.transpose(a[0]))
        # L0 = C0 - B0 @ R[0] @ np.transpose(B0)
        # theta_0 = self.np_random_inst.multivariate_normal(np.transpose(I0)[0], L0)
        # self.backward_smoothing_samples.append(np.reshape(theta_0, (2,1)))


    def get_onesample_path(self, col_vec = False):
        if col_vec:
            return list(reversed(self.backward_smoothing_samples))
        else:
            return list(reversed([np.reshape(x, (2,)) for x in self.backward_smoothing_samples]))



if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    from DLM_Core import DLM_model_container, DLM_simulator, DLM_without_W_by_discounting, DLM_visualizer
    from util_time_series import difference_oper, autocorr
    
    # ===
    delta2 = 0.8
    true_model2_inst = DLM_model_container(102)
    true_model2_inst.set_F_const_design_mat(np.array([[1],[0]]))
    true_model2_inst.set_G_const_transition_mat(np.array([
        [1,1],
        [0,1]
    ]))
    true_model2_inst.set_V_const_obs_eq_covariance(np.array([[100]]))
    true_model2_inst.set_W_const_state_error_cov(100*np.array([
        [1-(delta2**2), (1-delta2)**2],
        [(1-delta2)**2, (1-delta2)**3]
    ]))
    simulator2_inst = DLM_simulator(true_model2_inst, 20230318)
    simulator2_inst.simulate_data(np.array([100, 0]), 
                                100*np.array([[1-delta2**2, (1-delta2)**2],[(1-delta2)**2, (1-delta2)**3]]))

    data2_theta, data2_y = simulator2_inst.get_theta_y()

    fig_sim2, ax_sim2 = plt.subplots(3,3,figsize=(12, 4))
    fig_sim2.tight_layout()
    ax_sim2[0,0].plot(data2_y)
    ax_sim2[0,0].set_title("y")
    ax_sim2[0,1].plot(difference_oper(data2_y))
    ax_sim2[0,1].set_title("diff1(y)")
    ax_sim2[0,2].plot(difference_oper(difference_oper(data2_y)))
    ax_sim2[0,2].set_title("diff2(y)")
    ax_sim2[1,0].bar(range(51), autocorr(data2_y,50))
    ax_sim2[1,0].set_title("y,acf")
    ax_sim2[1,1].bar(range(51), autocorr(difference_oper(data2_y),50))
    ax_sim2[1,1].set_title("diff1(y),acf")
    ax_sim2[1,2].bar(range(51), autocorr(difference_oper(difference_oper(data2_y)),50))
    ax_sim2[1,2].set_title("diff2(y),acf")
    ax_sim2[2,0].plot([theta[0] for theta in data2_theta])
    ax_sim2[2,0].set_title("theta1")
    ax_sim2[2,1].plot([theta[1] for theta in data2_theta])
    ax_sim2[2,1].set_title("theta2")
    plt.show()

        
    model2e_container_inst = DLM_model_container(102)
    model2e_container_inst.set_F_const_design_mat(np.array([[1],[0]]))
    model2e_container_inst.set_G_const_transition_mat(np.array([[1,1],[0,1]]))
    model2e_container_inst.set_V_const_obs_eq_covariance([100])
    
    model2e_inst = DLM_without_W_by_discounting(data2_y, model2e_container_inst, [100,0], np.array([[1,0],[0,1]]), 0.8)
    model2e_inst.run()
    model2e_inst.run_retrospective_analysis()

    vis_inst = DLM_visualizer(model2e_inst, 0.95, False)
    vis_inst.show_smoothing((1,1), [0], show=False)

    ffbs_inst = DLM_BackwardSampling_Smoothing(model2e_inst)
    for i in range(5):
        ffbs_inst.run_bs()
        ex_path = ffbs_inst.get_onesample_path(col_vec=False)
        plt.plot(range(1, len(ex_path)+1), ex_path, color="green")
    plt.show()