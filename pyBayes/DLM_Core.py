from random import seed

import numpy as np
import scipy.stats as scss
import matplotlib.pyplot as plt


class DLM_model_container:
    # notation: following West(1997)
    # obser Y_t = F'_t \theta_t + v_t
    # state \theta_t = G_t \theta_{t-1} + w_t
    # error v_t ~ N(0, V_t), w_t~N(0, W_t). mutually indep, internally indep
    # mean.resp \mu = F'_t \theta_t
    def __init__(self, y_length: int):
        self.y_len = y_length

        self.F_obs_eq_design = None
        self.G_sys_eq_transition = None
        self.V_obs_eq_covariance = None
        self.W_sys_eq_covariance = None
        self.Wst_sys_eq_scale_free_covariance = None

        #optional
        self.u_covariate = None
        self.u_coeff_obs_eq_seq = None
        self.u_coeff_state_eq_seq = None

    # == setters: with an input as a sequence ==
    def set_Vt_obs_eq_covariance(self, Vt_seq: list[np.ndarray]):
        self.V_obs_eq_covariance = Vt_seq

    def set_Wt_state_error_cov(self, Wt_seq: list[np.ndarray]):
        self.W_sys_eq_covariance = Wt_seq

    def set_Wst_t_state_error_scale_free_cov(self, Wst_t_seq: list[np.ndarray]):
        self.Wst_sys_eq_scale_free_covariance = Wst_t_seq

    def set_Ft_design_mat(self, Ft_seq: list[np.ndarray]):
        "not F_t^T, but F_t, the column vector"
        self.F_obs_eq_design = Ft_seq

    def set_Gt_transition_mat(self, Gt_seq: np.ndarray):
        self.G_sys_eq_transition = Gt_seq

    def set_ut_covariate_and_coeff(self,
            ut_covariates_seq: list[np.ndarray],
            obs_eq_coeff_seq: list[np.ndarray], state_eq_coeff_seq: list[np.ndarray]):
        self.u_covariate = ut_covariates_seq
        self.u_coeff_obs_eq_seq = obs_eq_coeff_seq
        self.u_coeff_state_eq_seq = state_eq_coeff_seq

    # == setters: with an input as a point (when the value is constant on time in the model) ==
    def set_V_const_obs_eq_covariance(self, V: np.ndarray):
        self.V_obs_eq_covariance = [V for _ in range(self.y_len)]

    def set_W_const_state_error_cov(self, W: np.ndarray):
        self.W_sys_eq_covariance = [W for _ in range(self.y_len)]

    def set_Wst_const_state_error_scale_free_cov(self, Wst: np.ndarray):
        self.Wst_sys_eq_scale_free_covariance = [Wst for _ in range(self.y_len)]

    def set_F_const_design_mat(self, F: np.ndarray):
        self.F_obs_eq_design = [F for _ in range(self.y_len)]

    def set_G_const_transition_mat(self, G: np.ndarray):
        self.G_sys_eq_transition = [G for _ in range(self.y_len)]

    def set_u_const_covariate_and_coeff(self,
            u: np.ndarray,
            obs_eq_coeff: np.ndarray, state_eq_coeff: np.ndarray):
        self.u_covariate = [u for _ in range(self.y_len)]
        self.u_coeff_obs_eq_seq = [obs_eq_coeff for _ in range(self.y_len)]
        self.u_coeff_state_eq_seq = [state_eq_coeff for _ in range(self.y_len)]
    # == end setters ==

    def set_u_no_covariate(self):
        self.set_u_const_covariate_and_coeff(np.array([0]), np.array([0]), np.array([0]))


class DLM_utility:
    def vectorize_seq(self, seq):
        try:
            seq[0][0]
        except TypeError:
            return self._make_vectorize_seq(seq)
        else:
            return seq

    def _make_vectorize_seq(self, one_dim_seq: list):
        return [[x] for x in one_dim_seq]
    
    def container_checker(self, model_inst: DLM_model_container, check_F:bool, check_G:bool, check_W:bool, check_Wst:bool, check_V:bool):
        if check_F:
            if model_inst.F_obs_eq_design is None:
                raise ValueError("specify Ft")
        if check_G:
            if model_inst.G_sys_eq_transition is None:
                raise ValueError("specify Gt")
        if check_W:
            if model_inst.W_sys_eq_covariance is None:
                raise ValueError("specify Wt")
        if check_Wst:
            if model_inst.Wst_sys_eq_scale_free_covariance is None:
                raise ValueError("specify Wst_t")
        if check_V:
            if model_inst.V_obs_eq_covariance is None:
                raise ValueError("specify Vt")


class DLM_simulator:
    def __init__(self, model_inst: DLM_model_container, set_seed=None):
        #you should set F,G,W,V in the model_inst
        self.util_inst = DLM_utility()
        self.util_inst.container_checker(model_inst, True, True, True, False, True)

        self.DLM_model = model_inst

        self.theta_seq = []
        self.y_seq = []
        if set_seed is not None:
            seed(set_seed)
            self.rv_generator = np.random.default_rng(seed=set_seed)
        else:
            self.rv_generator = np.random.default_rng()

    def simulate_data(self, initial_m0, initial_C0):
    #                       E(theta_0|D0), var(theta_0|D0)
        theta_0 = self.rv_generator.multivariate_normal(initial_m0, initial_C0)
        self.theta_seq.append(theta_0)
        for t in range(self.DLM_model.y_len):
            theta_last = self.theta_seq[-1]
            theta_t = self.DLM_model.G_sys_eq_transition[t] @ theta_last + \
                    self.rv_generator.multivariate_normal(np.zeros(theta_last.shape), self.DLM_model.W_sys_eq_covariance[t])
            y_t = np.transpose(self.DLM_model.F_obs_eq_design[t]) @ theta_t + \
                    self.rv_generator.multivariate_normal(np.zeros(self.DLM_model.V_obs_eq_covariance[t].shape[0]), self.DLM_model.V_obs_eq_covariance[t])

            self.theta_seq.append(theta_t)
            self.y_seq.append(y_t)
    
    def get_theta_y(self):
        return self.theta_seq[1:], self.y_seq


class DLM_full_model:
    def __init__(self, y_observation, model_inst: DLM_model_container, initial_m0_given_D0: np.ndarray, initial_C0_given_D0: np.ndarray):
        self.util_inst = DLM_utility()
        self.util_inst.container_checker(model_inst, True, True, True, False, True)

        #input
        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.DLM_model = model_inst
        self.m0 = initial_m0_given_D0
        self.C0 = initial_C0_given_D0

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0] #index 0 -> m0|D0 (initial value will be deleted after self.run())
        self.C_posterior_var = [initial_C0_given_D0] #index 0 -> C0|D0 (initial value will be deleted after self.run())
        self.a_prior_mean = [] #index 0 -> a1=E[theta_1|D0]
        self.R_prior_var = [] #index 0 -> R1=Var(theta_1|D0)
        self.f_one_step_forecast_mean = [] #index 0 -> f1=E[y1|D0]
        self.Q_one_step_forecast_var = [] #index 0 -> Q1=Var(y1|D0)
        self.e_one_step_forecast_err = [] #index 0 -> e1 = y1-f1
        self.A_adaptive_vector = [] #index 0 -> A1

        self.t_smoothing_given_Dt = None
        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []
        self.rB_retrospective_gain_B = []

        self.t_forecasting_given_Dt = None
        self.K_forecasting_given_Dt = None
        self.fa_forecast_state_mean_a = []
        self.fR_forecast_state_var_R = []
        self.ff_forecast_obs_mean_f = []
        self.fQ_forecast_obs_var_Q = []

        
        

    def _one_iter(self, t):
        # prior
        Gt = self.DLM_model.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]
        Rt = Gt @ self.C_posterior_var[-1] @ np.transpose(Gt) + self.DLM_model.W_sys_eq_covariance[t-1]

        # one-step forecast
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at #E[y_t|D_{t-1}]
        Qt = np.transpose(Ft) @ Rt @ Ft + self.DLM_model.V_obs_eq_covariance[t-1]

        # posterior
        At = Rt @ Ft @ np.linalg.inv(Qt)
        et = self.y_observation[t-1] - ft
        mt = at + (At @ et)
        Ct = Rt - (At @ Qt @ np.transpose(At))

        #save
        self.m_posterior_mean.append(mt)
        self.C_posterior_var.append(Ct)
        self.a_prior_mean.append(at)
        self.R_prior_var.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Q_one_step_forecast_var.append(Qt)
        self.A_adaptive_vector.append(At)
        self.e_one_step_forecast_err.append(et)

    def run(self):
        for t in range(1, self.y_len+1):
            self._one_iter(t)
        # delete initial value
        self.m_posterior_mean = self.m_posterior_mean[1:] #delete initial
        self.C_posterior_var = self.C_posterior_var[1:] #delete initial

    #retrospective analysis
    def _retro_one_iter(self, t, k):
        i = t-k
        B_i = self.C_posterior_var[i-1] @ np.transpose(self.DLM_model.G_sys_eq_transition[i]) @ np.linalg.inv(self.R_prior_var[i])
        a_t_k = self.m_posterior_mean[i-1] + B_i @ (self.ra_reversed_retrospective_a[-1] - self.a_prior_mean[i])
        R_t_k = self.C_posterior_var[i-1] + B_i @ (self.rR_reversed_retrospective_R[-1] - self.R_prior_var[i]) @ np.transpose(B_i)

        self.ra_reversed_retrospective_a.append(a_t_k)
        self.rR_reversed_retrospective_R.append(R_t_k)

    def run_retrospective_analysis(self, t_of_given_Dt=None):
        if t_of_given_Dt is None:
            t = self.y_len
        else:
            t = t_of_given_Dt
        
        self.t_smoothing_given_Dt = t
        self.ra_reversed_retrospective_a = [self.m_posterior_mean[t-1]]
        self.rR_reversed_retrospective_R = [self.C_posterior_var[t-1]]
        
        for k in range(1,t):
            self._retro_one_iter(t, k)
    
    #forecast analysis
    def _forecast_one_iter(self, t_k):
        G_t_k = self.DLM_model.G_sys_eq_transition[t_k-1]
        a_t_k = G_t_k @ self.fa_forecast_state_mean_a[-1]
        R_t_k = G_t_k @ self.fR_forecast_state_var_R[-1] @ np.transpose(G_t_k) + self.DLM_model.W_sys_eq_covariance[t_k-1]
        
        F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
        f_t_k = np.transpose(F_t_k) @ a_t_k
        Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.DLM_model.V_obs_eq_covariance[t_k-1]

        self.fa_forecast_state_mean_a.append(a_t_k)
        self.fR_forecast_state_var_R.append(R_t_k)
        self.ff_forecast_obs_mean_f.append(f_t_k)
        self.fQ_forecast_obs_var_Q.append(Q_t_k)
        

    def run_forecast_analysis(self, t_of_given_Dt, K_forecast_end_time):
        t = t_of_given_Dt
        self.fa_forecast_state_mean_a = [self.m_posterior_mean[t-1]] #m_t=a_t(0)
        self.fR_forecast_state_var_R = [self.C_posterior_var[t-1]] #C_t=R_t(0)
        self.t_forecasting_given_Dt = t_of_given_Dt
        self.K_forecasting_given_Dt = K_forecast_end_time

        for t_k in range(t_of_given_Dt + 1, K_forecast_end_time + 1):
            self._forecast_one_iter(t_k)

        self.fa_forecast_state_mean_a = self.fa_forecast_state_mean_a[1:] #delete a_t(0)
        self.fR_forecast_state_var_R = self.fR_forecast_state_var_R[1:] #delete R_t(0)

    def cal_forecast_state_covariance(self, t_of_given_Dt, t_plus_k, t_plus_j):
        if t_plus_k > t_plus_j:
            K, j = t_plus_k-t_of_given_Dt, t_plus_j-t_of_given_Dt #k>j
        elif t_plus_k < t_plus_j:
            K, j = t_plus_j-t_of_given_Dt, t_plus_k-t_of_given_Dt #k>j
        else:
            return self.fR_forecast_state_var_R[t_plus_k - t_of_given_Dt - 1]

        fCt = [self.fR_forecast_state_var_R[j - 1]] #Ct(j,j)
        for k in range(j+1, K+1):
            fCt.append(self.DLM_model.G_sys_eq_transition[t_of_given_Dt+k-1] @ fCt[-1])
        return fCt[-1]

    # == getters ==
    def get_posterior_m_C(self):
        return self.m_posterior_mean, self.C_posterior_var

    def get_prior_a_R(self):
        return self.a_prior_mean, self.R_prior_var

    def get_one_step_forecast_f_Q(self):
        return self.f_one_step_forecast_mean, self.Q_one_step_forecast_var

    def get_one_step_forecast_error_e(self):
            return self.e_one_step_forecast_err

    def get_retrospective_a_R(self):
        a_smoothed = list(reversed(self.ra_reversed_retrospective_a))
        R_smoothed = list(reversed(self.rR_reversed_retrospective_R))
        return a_smoothed, R_smoothed

    def get_forecast_a_R(self):
        return self.fa_forecast_state_mean_a, self.fR_forecast_state_var_R

    def get_forecast_f_Q(self):
        return self.ff_forecast_obs_mean_f, self.fQ_forecast_obs_var_Q

    

class DLM_full_model_with_reference_prior(DLM_full_model):
    def __init__(self, y_observation, model_inst: DLM_model_container):
        super().__init__(y_observation, model_inst, initial_m0_given_D0=None, initial_C0_given_D0=None)
        #delete initial None
        self.m_posterior_mean = []
        self.C_posterior_var = []
        
        self.K_improper_postker_order2 = []
        self.k_improper_postker_order1 = []

    def _one_iter_in_improper_period(self, t):
        Wt_inv = np.linalg.inv(self.DLM_model.W_sys_eq_covariance[t-1])
        Gt = self.DLM_model.G_sys_eq_transition[t-1]

        Pt = np.transpose(Gt) @ Wt_inv @ Gt + self.K_improper_postker_order2[-1]
        WinvGPinv = Wt_inv @ Gt @ np.linalg.inv(Pt)
        Ht = Wt_inv - WinvGPinv @ np.transpose(Gt) @ Wt_inv
        ht = WinvGPinv @ self.k_improper_postker_order1[-1]

        #V is known
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        Vt_inv = np.linalg.inv(self.DLM_model.V_obs_eq_covariance[t-1])
        Kt = Ht + Ft @ Vt_inv @ np.transpose(Ft)
        kt = ht + Ft @ Vt_inv @ self.y_observation[t-1]

        self.K_improper_postker_order2.append(Kt)
        self.k_improper_postker_order1.append(kt)

    def run(self):
        theta_dim = self.DLM_model.W_sys_eq_covariance[0].shape[0]
        self.K_improper_postker_order2 = [np.zeros((theta_dim, theta_dim))]
        self.k_improper_postker_order1 = [np.zeros((theta_dim))]

        for t in range(1, self.y_len+1):
            if t > theta_dim:
                self._one_iter(t)
            elif t == (theta_dim):
                self._one_iter_in_improper_period(t)
                K_t_inv = np.linalg.inv(self.K_improper_postker_order2[-1])
                k_t = self.k_improper_postker_order1[-1]
                self.m_posterior_mean.append(K_t_inv@k_t)
                self.C_posterior_var.append(K_t_inv)
                self.a_prior_mean.append(None)
                self.R_prior_var.append(None)
                self.f_one_step_forecast_mean.append(None)
                self.Q_one_step_forecast_var.append(None)
                self.A_adaptive_vector.append(None)
                self.e_one_step_forecast_err.append(None)
            else:
                self._one_iter_in_improper_period(t)
                self.m_posterior_mean.append(None)
                self.C_posterior_var.append(None)
                self.a_prior_mean.append(None)
                self.R_prior_var.append(None)
                self.f_one_step_forecast_mean.append(None)
                self.Q_one_step_forecast_var.append(None)
                self.A_adaptive_vector.append(None)
                self.e_one_step_forecast_err.append(None)

        self.K_improper_postker_order2 = self.K_improper_postker_order2[1:]
        self.k_improper_postker_order1 = self.k_improper_postker_order1[1:]

    def run_retrospective_analysis(self, t_of_given_Dt=None):
        raise NotImplementedError("maybe..later..") # chap 4.10.4

    def run_forecast_analysis(self, t_of_given_Dt, K_end_step):
        theta_dim = self.DLM_model.W_sys_eq_covariance[0].shape[0]
        if t_of_given_Dt < theta_dim+2:
            raise AttributeError("t of Dt should be in the range of proper posteriors")
        return super().run_forecast_analysis(t_of_given_Dt, K_end_step)


class DLM_without_W_by_discounting(DLM_full_model):
    def __init__(self, y_observation, 
                model_inst_having_F_G_V: DLM_model_container,
                initial_m0_given_D0: np.ndarray,
                initial_C0_given_D0: np.ndarray,
                discount_factor_for_W: float):

        self.util_inst = DLM_utility()
        self.util_inst.container_checker(model_inst_having_F_G_V, True, True, False, False, True)

        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.DLM_model = model_inst_having_F_G_V
        self.m0 = initial_m0_given_D0
        self.C0 = initial_C0_given_D0
        self.delta_W = discount_factor_for_W

        #manipulate D0 (structurally it is bad...)
        self.DLM_model.W_sys_eq_covariance = []

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0] #index 0 -> m0|D0 (initial value will be deleted after self.run())
        self.C_posterior_var = [initial_C0_given_D0] #index 0 -> C0|D0 (initial value will be deleted after self.run())
        self.a_prior_mean = [] #index 0 -> a1=E[theta_1|D0]
        self.R_prior_var = [] #index 0 -> R1=Var(theta_1|D0)
        self.f_one_step_forecast_mean = [] #index 0 -> f1=E[y1|D0]
        self.Q_one_step_forecast_var = [] #index 0 -> Q1=Var(y1|D0)
        self.e_one_step_forecast_err = [] #index 0 -> e1 = y1-f1
        self.A_adaptive_vector = [] #index 0 -> A1
        
        self.t_smoothing_given_Dt = None
        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []
        self.rB_retrospective_gain_B = []

        self.t_forecasting_given_Dt = None
        self.K_forecasting_given_Dt = None
        self.fa_forecast_state_mean_a = []
        self.fR_forecast_state_var_R = []
        self.ff_forecast_obs_mean_f = []
        self.fQ_forecast_obs_var_Q = []


    def _make_Rt_Wt_using_discounting_factor(self, Pt): #override here for component-discount model
        Rst_t = Pt/self.delta_W
        applied_Wt = Pt*(1-self.delta_W)/self.delta_W
        return Rst_t, applied_Wt
        
    def _one_iter(self, t):
        # prior
        Gt = self.DLM_model.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]
        Pt = Gt @ self.C_posterior_var[-1] @ np.transpose(Gt)
        Rt, applied_Wt = self._make_Rt_Wt_using_discounting_factor(Pt)
        self.DLM_model.W_sys_eq_covariance.append(applied_Wt) #bad

        # one-step forecast
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at #E[y_t|D_{t-1}]
        Qt = np.transpose(Ft) @ Rt @ Ft + self.DLM_model.V_obs_eq_covariance[t-1]

        # posterior
        At = Rt @ Ft @ np.linalg.inv(Qt)
        et = self.y_observation[t-1] - ft
        mt = at + (At @ et)
        Ct = Rt - (At @ Qt @ np.transpose(At))

        #save
        self.m_posterior_mean.append(mt)
        self.C_posterior_var.append(Ct)
        self.a_prior_mean.append(at)
        self.R_prior_var.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Q_one_step_forecast_var.append(Qt)
        self.A_adaptive_vector.append(At)
        self.e_one_step_forecast_err.append(et)


    def _forecast_one_iter(self, t_k, t_of_given_Dt):
        G_t_k = self.DLM_model.G_sys_eq_transition[t_k-1]
        a_t_k = G_t_k @ self.fa_forecast_state_mean_a[-1]

        if t_k > self.y_len: #extrapolation over y_len
            G_t_1 = self.DLM_model.G_sys_eq_transition[t_of_given_Dt]
            Pst_1 = G_t_1 @ self.C_posterior_var[-1] @ np.transpose(G_t_1)
            _, Wt_1 = self._make_Rt_Wt_using_discounting_factor(Pst_1)

            R_t_k = G_t_k @ self.fR_forecast_state_var_R[-1] @ np.transpose(G_t_k) + Wt_1 #modified by chap6.3

            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.DLM_model.V_obs_eq_covariance[t_k-1]

        else:
            R_t_k = G_t_k @ self.fR_forecast_state_var_R[-1] @ np.transpose(G_t_k) + self.DLM_model.W_sys_eq_covariance[t_k-1]
        
            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.DLM_model.V_obs_eq_covariance[t_k-1]

        self.fa_forecast_state_mean_a.append(a_t_k)
        self.fR_forecast_state_var_R.append(R_t_k)
        self.ff_forecast_obs_mean_f.append(f_t_k)
        self.fQ_forecast_obs_var_Q.append(Q_t_k)
        

    def run_forecast_analysis(self, t_of_given_Dt, K_forecast_end_time):
        t = t_of_given_Dt
        self.fa_forecast_state_mean_a = [self.m_posterior_mean[t-1]]
        self.fR_forecast_state_var_R = [self.C_posterior_var[t-1]]
        self.t_forecasting_given_Dt = t_of_given_Dt
        self.K_forecasting_given_Dt = K_forecast_end_time

        for t_k in range(t_of_given_Dt + 1, K_forecast_end_time + 1):
            self._forecast_one_iter(t_k, t_of_given_Dt)

        self.fa_forecast_state_mean_a = self.fa_forecast_state_mean_a[1:]
        self.fR_forecast_state_var_R = self.fR_forecast_state_var_R[1:]



class DLM_univariate_y_without_V_in_D0:
    #chapter 4.5 of West
    #conjugate analysis. when V is unknown, y is univariate
    def _checker(self, model_inst: DLM_model_container):
        self.util_inst.container_checker(model_inst, True, True, False, True, False)

    def __init__(self, y_observation, model_inst_having_F_G_Wst: DLM_model_container,
                initial_m0_given_D0: np.ndarray, initial_C0st_given_D0: np.ndarray, n0_given_D0:float, S0_given_D0:float):
                #D0 should have F,G,Wst
        self.util_inst = DLM_utility()
        self._checker(model_inst_having_F_G_Wst)

        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.DLM_model = model_inst_having_F_G_Wst
        self.m0 = initial_m0_given_D0
        self.C0st = initial_C0st_given_D0
        self.C0 = None
        self.n0 = n0_given_D0
        self.S0 = S0_given_D0

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0] # index 0 -> m_0=E[theta_0|D0]
        self.Cst_posterior_var = [initial_C0st_given_D0] # index 0 -> v*C*_0=Var(theta_0|D0,v)
        self.C_posterior_scale = [S0_given_D0*initial_C0st_given_D0] # index 0 -> C0=Scale(theta_0|D0)
        self.a_prior_mean = [] # index 0 -> a_1=E[theta_1|D_0]
        self.Rst_prior_var = [] # index 0 -> v*R*_1=Var(theta_1|D_0,v)
        self.R_prior_scale = [] # index 0 -> R_1=Scale(theta_1|D_0)
        self.f_one_step_forecast_mean = [] # index 0 -> f_1=E[y_1|D_0]
        self.Qst_one_step_forecast_var = [] # index 0 -> v*Q*_1=Var(y_1|D_0,v)
        self.Q_one_step_forecast_scale = [] # index 0 -> Q=Scale(y_1|D_0)
        self.A_adaptive_vector = [] # index 0 -> A_1
        self.e_one_step_forecast_err = [] #index 0 -> e1 = y1-f1
        self.n_precision_shape = [n0_given_D0] #shape: {n_t}/2 at t. index 0 -> n0
        self.S_precision_rate = [S0_given_D0] #rate: {n_t}{S_t}/2 at t. index 0 -> S0
        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []
        self.fa_forecast_state_mean_a = []
        self.fR_forecast_state_scale_R = []
        self.ff_forecast_obs_mean_f = []
        self.fQ_forecast_obs_scale_Q = []
        self.t_smoothing_given_Dt = None
        self.t_forecasting_given_Dt = None
        self.K_forecasting_given_Dt = None


    def _one_iter(self, t):
        #conditional on V
        ##prior
        Gt = self.DLM_model.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]
        Rst_t = Gt @ self.Cst_posterior_var[-1] @ np.transpose(Gt) + self.DLM_model.Wst_sys_eq_scale_free_covariance[t-1]
        ##one_step_forecast
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at
        Qst_t = 1 + (np.transpose(Ft) @ Rst_t @ Ft)

        ##posterior
        et = self.y_observation[t-1] - ft
        At = Rst_t @ Ft / Qst_t
        mt = at + (At @ et)
        Cst_t = Rst_t - (At @ np.transpose(At))*Qst_t

        #precision
        nt = self.n_precision_shape[-1] + 1
        S_t1 = self.S_precision_rate[-1]
        St = (S_t1*(nt-1)/nt) + (et @ et) / (nt * Qst_t[0][0])
        
        #unconditional on V
        Ct = St * Cst_t
        Rt = S_t1 * Rst_t
        Qt = S_t1 * Qst_t

        # direct calculation
        # print(St, S_t1+(S_t1/nt)*((et @ et / Qt - 1)))
        # print(mt, at+At@et)
        # print(Ct, (St/S_t1) * (Rt-At@np.transpose(At)*Qt))

        #save
        self.m_posterior_mean.append(mt)
        self.Cst_posterior_var.append(Cst_t)
        self.C_posterior_scale.append(Ct)
        self.a_prior_mean.append(at)
        self.Rst_prior_var.append(Rst_t)
        self.R_prior_scale.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Qst_one_step_forecast_var.append(Qst_t)
        self.Q_one_step_forecast_scale.append(Qt)
        self.A_adaptive_vector.append(At)
        self.e_one_step_forecast_err.append(et)
        self.n_precision_shape.append(nt)
        self.S_precision_rate.append(St)
    
    def run(self):
        for t in range(1, self.y_len+1):
            self._one_iter(t)
        # delete initial value
        self.m_posterior_mean = self.m_posterior_mean[1:]
        self.Cst_posterior_var = self.Cst_posterior_var[1:]
        self.C_posterior_scale = self.C_posterior_scale[1:]
        self.n_precision_shape = self.n_precision_shape[1:]
        self.S_precision_rate = self.S_precision_rate[1:]

    #retrospective analysis: should I use C, R? or C_star, R_star?
    #now, use C,R
    def _retro_one_iter(self, t, k):
        i = t-k
        B_i = self.C_posterior_scale[i-1] @ np.transpose(self.DLM_model.G_sys_eq_transition[i]) @ np.linalg.inv(self.R_prior_scale[i])
        a_t_k = self.m_posterior_mean[i-1] + B_i @ (self.ra_reversed_retrospective_a[-1] - self.a_prior_mean[i])
        R_t_k = self.C_posterior_scale[i-1] + B_i @ (self.rR_reversed_retrospective_R[-1] - self.R_prior_scale[i]) @ np.transpose(B_i)

        self.ra_reversed_retrospective_a.append(a_t_k)
        self.rR_reversed_retrospective_R.append(R_t_k)

    def run_retrospective_analysis(self, t_of_given_Dt=None):
        if t_of_given_Dt is None:
            t = self.y_len
        else:
            t = t_of_given_Dt
        self.t_smoothing_given_Dt = t
        
        self.ra_reversed_retrospective_a = [self.m_posterior_mean[t-1]]
        self.rR_reversed_retrospective_R = [self.C_posterior_scale[t-1]]
        
        for k in range(1,t):
            self._retro_one_iter(t, k)

    #forecast analysis
    def _make_Rst_t_Wst_t(self, P_t_k, t_k):
        Wst_t_k = self.DLM_model.Wst_sys_eq_scale_free_covariance[t_k-1]
        Rst_t_k = P_t_k + Wst_t_k
        return Rst_t_k, Wst_t_k

    def _forecast_one_iter(self, t_k, t_of_given_Dt):
        G_t_k = self.DLM_model.G_sys_eq_transition[t_k-1]
        a_t_k = G_t_k @ self.fa_forecast_state_mean_a[-1]

        if t_k > self.y_len: #extrapolation over y_len
            G_t_1 = self.DLM_model.G_sys_eq_transition[t_of_given_Dt]
            Pst_1 = G_t_1 @ self.Cst_posterior_var[-1] @ np.transpose(G_t_1)
            _, Wt_1 = self._make_Rst_t_Wst_t(Pst_1, t_k) #W_t_1 is from the model container in this case, because we know W_st_t

            R_t_k = G_t_k @ self.fR_forecast_state_scale_R[-1] @ np.transpose(G_t_k) + Wt_1 #modified by chap6.3

            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.S_precision_rate[t_of_given_Dt-1] #modified

        else:
            P_t_k = G_t_k @ self.fR_forecast_state_scale_R[-1] @ np.transpose(G_t_k)
            R_t_k, _ = self._make_Rst_t_Wst_t(P_t_k, t_k)
        
            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.S_precision_rate[t_k-1]

        self.fa_forecast_state_mean_a.append(a_t_k)
        self.fR_forecast_state_scale_R.append(R_t_k)
        self.ff_forecast_obs_mean_f.append(f_t_k)
        self.fQ_forecast_obs_scale_Q.append(Q_t_k)

    def run_forecast_analysis(self, t_of_given_Dt, K_end_step):
        t = t_of_given_Dt
        self.fa_forecast_state_mean_a = [self.m_posterior_mean[t-1]]
        self.fR_forecast_state_scale_R = [self.C_posterior_scale[t-1]]
        self.t_forecasting_given_Dt = t_of_given_Dt
        self.K_forecasting_given_Dt = K_end_step


        for t_k in range(t_of_given_Dt + 1, K_end_step + 1):
            self._forecast_one_iter(t_k, t_of_given_Dt)

        self.fa_forecast_state_mean_a = self.fa_forecast_state_mean_a[1:]
        self.fR_forecast_state_scale_R = self.fR_forecast_state_scale_R[1:]

    def cal_forecast_state_covariance(self, t_of_given_Dt, t_plus_k, t_plus_j):
        if t_plus_k > t_plus_j:
            k, j = t_plus_k, t_plus_j #k>j
        elif t_plus_k < t_plus_j:
            k, j = t_plus_j, t_plus_k #k>j
        else:
            return self.fR_forecast_state_scale_R[t_plus_k - t_of_given_Dt - 1]

        fCt = [self.fR_forecast_state_scale_R[t_plus_j - t_of_given_Dt - 1]] #Ct(j,j)
        for i in range(j+1, k+1):
            fCt.append(self.DLM_model.G_sys_eq_transition[i] @ fCt[-1])
        return fCt[-1]

    # == getters ==
    def get_posterior_m_C(self):
        return self.m_posterior_mean, self.C_posterior_scale

    def get_prior_a_R(self):
        return self.a_prior_mean, self.R_prior_scale

    def get_one_step_forecast_f_Q(self):
        return self.f_one_step_forecast_mean, self.Q_one_step_forecast_scale
    
    def get_retrospective_a_R(self):
        a_smoothed = list(reversed(self.ra_reversed_retrospective_a))
        R_smoothed = list(reversed(self.rR_reversed_retrospective_R))
        return a_smoothed, R_smoothed

    def get_forecast_a_R(self):
        return self.fa_forecast_state_mean_a, self.fR_forecast_state_scale_R

    def get_forecast_f_Q(self):
        return self.ff_forecast_obs_mean_f, self.fQ_forecast_obs_scale_Q


class DLM_univariate_y_without_V_in_D0_with_reference_prior(DLM_univariate_y_without_V_in_D0):
    def __init__(self, y_observation, D0: DLM_model_container):
        #D0 should have F,G,Wst
        super().__init__(y_observation, D0, None, 0, None, 0)
        #delete initial None
        self.m_posterior_mean = []
        self.Cst_posterior_var = []
        self.C_posterior_scale = []
        self.n_precision_shape = []
        self.S_precision_rate = []
        
        self.K_improper_postker_order2 = []
        self.k_improper_postker_order1 = []
        self.gamma_scale = []
        self.lambda_prior_correction = []
        self.delta_post_correction = []

    def _one_iter_in_improper_period(self, t):
        Wt_inv = np.linalg.inv(self.DLM_model.Wst_sys_eq_scale_free_covariance[t-1]) # scale-free!
        Gt = self.DLM_model.G_sys_eq_transition[t-1]

        Pt = np.transpose(Gt) @ Wt_inv @ Gt + self.K_improper_postker_order2[-1]
        Pt_inv = np.linalg.inv(Pt)
        WinvGPinv = Wt_inv @ Gt @ Pt_inv
        Ht = Wt_inv - WinvGPinv @ np.transpose(Gt) @ Wt_inv
        ht = WinvGPinv @ self.k_improper_postker_order1[-1]

        #V is unknown
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        Kt = Ht + Ft @ np.transpose(Ft)
        kt = ht + Ft @ self.y_observation[t-1]

        gamma_t = self.gamma_scale[-1] + 1
        lambda_t = None
        delta_t = None
        if t == 1:
            lambda_t = 0
            delta_t = self.y_observation[0][0]**2
        else:
            lambda_t = self.delta_post_correction[-1] - np.transpose(self.k_improper_postker_order1[-1]) @ Pt_inv @ self.k_improper_postker_order1[-1]
            delta_t = lambda_t + self.y_observation[t-1][0]**2

        self.K_improper_postker_order2.append(Kt)
        self.k_improper_postker_order1.append(kt)
        self.gamma_scale.append(gamma_t)
        self.lambda_prior_correction.append(lambda_t)
        self.delta_post_correction.append(delta_t)

    def run(self):
        theta_dim = self.DLM_model.Wst_sys_eq_scale_free_covariance[0].shape[0]
        self.K_improper_postker_order2 = [np.zeros((theta_dim, theta_dim))]
        self.k_improper_postker_order1 = [np.zeros((theta_dim))]
        self.gamma_scale = [0]
        self.lambda_prior_correction = [0]
        self.delta_post_correction = [0]

        for t in range(1, self.y_len+1):
            if t > theta_dim+1:
                self._one_iter(t)
            elif t == (theta_dim+1):
                self._one_iter_in_improper_period(t)
                #for K,k
                K_t_inv = np.linalg.inv(self.K_improper_postker_order2[-1])
                k_t = self.k_improper_postker_order1[-1]
                m_t = K_t_inv @ k_t
                #for S
                n_t = self.gamma_scale[-1] - theta_dim
                assert n_t==1
                d_t = self.delta_post_correction[-1] - np.transpose(k_t) @ m_t
                S_t = d_t / n_t
                #update
                self.m_posterior_mean.append(m_t)
                self.Cst_posterior_var.append(K_t_inv)
                self.C_posterior_scale.append(S_t * K_t_inv)
                self.a_prior_mean.append(None)
                self.Rst_prior_var.append(None)
                self.R_prior_scale.append(None)
                self.f_one_step_forecast_mean.append(None)
                self.Qst_one_step_forecast_var.append(None)
                self.Q_one_step_forecast_scale.append(None)
                self.A_adaptive_vector.append(None)
                self.n_precision_shape.append(n_t)
                self.S_precision_rate.append(S_t)
            else:
                self._one_iter_in_improper_period(t)
                self.m_posterior_mean.append(None)
                self.Cst_posterior_var.append(None)
                self.C_posterior_scale.append(None)
                self.a_prior_mean.append(None)
                self.Rst_prior_var.append(None)
                self.R_prior_scale.append(None)
                self.f_one_step_forecast_mean.append(None)
                self.Qst_one_step_forecast_var.append(None)
                self.Q_one_step_forecast_scale.append(None)
                self.A_adaptive_vector.append(None)
                self.n_precision_shape.append(None)
                self.S_precision_rate.append(None)

        self.K_improper_postker_order2 = self.K_improper_postker_order2[1:]
        self.k_improper_postker_order1 = self.k_improper_postker_order1[1:]

    def run_retrospective_analysis(self, t_of_given_Dt=None):
        raise NotImplementedError("maybe..later..") # chap 4.10.4

    def run_forecast_analysis(self, t_of_given_Dt, K_end_step):
        theta_dim = self.DLM_model.Wst_sys_eq_scale_free_covariance[0].shape[0]
        if t_of_given_Dt < theta_dim+2:
            raise AttributeError("t of Dt should be in the range of proper posteriors")
        return super().run_forecast_analysis(t_of_given_Dt, K_end_step)

class DLM_univariate_y_without_V_W_in_D0(DLM_univariate_y_without_V_in_D0):
    def _checker(self, model_inst):
        self.util_inst.container_checker(model_inst, True, True, False, False, False)

    def __init__(self, y_observation, 
                model_inst_having_F_G: DLM_model_container,
                initial_m0_given_D0: np.ndarray, 
                initial_C0st_given_D0: np.ndarray, 
                n0_given_D0: float, 
                S0_given_D0: float,
                discount_factor_for_Wst: float):
                #D0 should have F,G

        self.util_inst = DLM_utility()
        self._checker(model_inst_having_F_G)

        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.y_len = len(y_observation)
        self.DLM_model = model_inst_having_F_G
        self.m0 = initial_m0_given_D0
        self.C0st = initial_C0st_given_D0
        self.n0 = n0_given_D0
        self.S0 = S0_given_D0
        self.delta_Wst = discount_factor_for_Wst

        #manipulate D0 (structurally it is bad...)
        self.DLM_model.Wst_sys_eq_scale_free_covariance = []

        #result containers
        self.m_posterior_mean = [initial_m0_given_D0]
        self.Cst_posterior_var = [initial_C0st_given_D0]
        self.C_posterior_scale = [S0_given_D0*initial_C0st_given_D0]
        self.a_prior_mean = []
        self.Rst_prior_var = []
        self.R_prior_scale = []
        self.f_one_step_forecast_mean = []
        self.Qst_one_step_forecast_var = []
        self.Q_one_step_forecast_scale = []
        self.A_adaptive_vector = []
        self.e_one_step_forecast_err = []
        self.n_precision_shape = [n0_given_D0] #shape: {n_t}/2 at t
        self.S_precision_rate = [S0_given_D0] #rate: {n_t}{S_t}/2 at t
        self.ra_reversed_retrospective_a = []
        self.rR_reversed_retrospective_R = []
        self.fa_forecast_state_mean_a = []
        self.fR_forecast_state_scale_R = []
        self.ff_forecast_obs_mean_f = []
        self.fQ_forecast_obs_scale_Q = []

    def _make_Rt_Wst_t_using_discounting_factor(self, Pst): #override here for component-discount model
        Rst_t = Pst/self.delta_Wst
        applied_Wst = Pst*(1-self.delta_Wst)/self.delta_Wst
        return Rst_t, applied_Wst

    def _one_iter(self, t):
        #conditional on V
        ##prior
        Gt = self.DLM_model.G_sys_eq_transition[t-1]
        at = Gt @ self.m_posterior_mean[-1]

        Pst = Gt @ self.Cst_posterior_var[-1] @ np.transpose(Gt)
        Rst_t, applied_Wst = self._make_Rt_Wst_t_using_discounting_factor(Pst)
        self.DLM_model.Wst_sys_eq_scale_free_covariance.append(applied_Wst) #save for forecasting

        ##one_step_forecast
        Ft = self.DLM_model.F_obs_eq_design[t-1]
        ft = np.transpose(Ft) @ at
        Qst_t = 1 + np.transpose(Ft) @ Rst_t @ Ft

        ##posterior
        et = self.y_observation[t-1] - ft
        At = Rst_t @ Ft / Qst_t
        mt = at + At @ et
        Cst_t = Rst_t - At @ np.transpose(At) * Qst_t

        #precision
        nt = self.n_precision_shape[-1] + 1
        S_t1 = self.S_precision_rate[-1]
        St = S_t1*(nt-1)/nt + et @ et / (nt * Qst_t[0][0])

        #unconditional on V
        Ct = St * Cst_t
        Rt = S_t1 * Rst_t
        Qt = S_t1 * Qst_t
        
        #save
        self.m_posterior_mean.append(mt)
        self.Cst_posterior_var.append(Cst_t)
        self.C_posterior_scale.append(Ct)
        self.a_prior_mean.append(at)
        self.Rst_prior_var.append(Rst_t)
        self.R_prior_scale.append(Rt)
        self.f_one_step_forecast_mean.append(ft)
        self.Qst_one_step_forecast_var.append(Qst_t)
        self.Q_one_step_forecast_scale.append(Qt)
        self.A_adaptive_vector.append(At)
        self.e_one_step_forecast_err.append(et)
        self.n_precision_shape.append(nt)
        self.S_precision_rate.append(St)

    def _forecast_one_iter(self, t_k, t_of_given_Dt):
        G_t_k = self.DLM_model.G_sys_eq_transition[t_k-1]
        a_t_k = G_t_k @ self.fa_forecast_state_mean_a[-1]

        if t_k > self.y_len: #extrapolation over y_len
            G_t_1 = self.DLM_model.G_sys_eq_transition[t_of_given_Dt]
            Pst_1 = G_t_1 @ self.Cst_posterior_var[-1] @ np.transpose(G_t_1)
            _, Wt_1 = self._make_Rt_Wst_t_using_discounting_factor(Pst_1)

            R_t_k = G_t_k @ self.fR_forecast_state_scale_R[-1] @ np.transpose(G_t_k) + Wt_1 #modified by chap6.3

            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.S_precision_rate[t_of_given_Dt-1] #modified
            # print(F_t_k, R_t_k, self.S_precision_rate[t_of_given_Dt-1])
            # print(np.transpose(F_t_k) @ R_t_k @ F_t_k)

        else:
            R_t_k = G_t_k @ self.fR_forecast_state_scale_R[-1] @ np.transpose(G_t_k) + self.DLM_model.Wst_sys_eq_scale_free_covariance[t_k-1]
        
            F_t_k = self.DLM_model.F_obs_eq_design[t_k-1]
            f_t_k = np.transpose(F_t_k) @ a_t_k
            Q_t_k = np.transpose(F_t_k) @ R_t_k @ F_t_k + self.S_precision_rate[t_k-1]

        self.fa_forecast_state_mean_a.append(a_t_k)
        self.fR_forecast_state_scale_R.append(R_t_k)
        self.ff_forecast_obs_mean_f.append(f_t_k)
        self.fQ_forecast_obs_scale_Q.append(Q_t_k)
        
    def run_forecast_analysis(self, t_of_given_Dt, K_end_step):
        t = t_of_given_Dt
        self.fa_forecast_state_mean_a = [self.m_posterior_mean[t-1]]
        self.fR_forecast_state_scale_R = [self.C_posterior_scale[t-1]]
        self.t_forecasting_given_Dt = t_of_given_Dt
        self.K_forecasting_given_Dt = K_end_step

        for t_k in range(t_of_given_Dt + 1, K_end_step + 1):
            self._forecast_one_iter(t_k, t_of_given_Dt)

        self.fa_forecast_state_mean_a = self.fa_forecast_state_mean_a[1:]
        self.fR_forecast_state_scale_R = self.fR_forecast_state_scale_R[1:]


class DLM_univariate_y_without_V_W_in_D0_with_component_discount_factor(DLM_univariate_y_without_V_W_in_D0):
    def __init__(self, y_observation, 
                model_inst_having_F_G: DLM_model_container,
                initial_m0_given_D0: np.ndarray, 
                initial_C0st_given_D0: np.ndarray, 
                n0_given_D0: float, 
                S0_given_D0: float,
                discount_factor_tuple: tuple[float],
                discount_component_blocks_partition: tuple[int]):
                #D0 should have F,G

        super().__init__(y_observation, 
                        model_inst_having_F_G,
                        initial_m0_given_D0,
                        initial_C0st_given_D0,
                        n0_given_D0,
                        S0_given_D0,
                        discount_factor_for_Wst = None)

        self.discount_factor_tuple = discount_factor_tuple
        self.discount_partition = discount_component_blocks_partition
        if discount_component_blocks_partition[-1] != len(initial_m0_given_D0):
            raise ValueError("check your discount_component_blocks_partition")

    def _make_Rt_Wst_t_using_discounting_factor(self, Pst): #override here for component-discount model
        Rst_t = np.copy(Pst)
        applied_Wst = np.copy(Pst)
        for i, delta in enumerate(self.discount_factor_tuple):
            start = 0
            if i!=0:
                start = self.discount_partition[i-1]
            end = self.discount_partition[i]

            block_Pst = Pst[start:end, start:end]
            Rst_t[start:end, start:end] = block_Pst/delta
            applied_Wst[start:end, start:end] = block_Pst * (1-delta)/delta
        
        return Rst_t, applied_Wst


class DLM_visualizer:
    def __init__(self, dlm_inst: DLM_univariate_y_without_V_in_D0, cred: float, is_used_t_dist: bool):
        self.dlm_inst = dlm_inst
        self.cred = cred

        # ===
        self.theta_len = len(self.dlm_inst.m0)
        self.y_len = self.dlm_inst.y_len
        self.y = self.dlm_inst.y_observation
        self.z_q = scss.norm.ppf([1-(1-cred)/2])[0]
        if is_used_t_dist:
            self.t_q = [scss.t.ppf([1-(1-cred)], df=d)[0] for d in self.dlm_inst.n_precision_shape]
        else:
            self.t_q = [self.z_q for _ in range(self.y_len)]
        

        # ===
        self.variable_names = ["param"+str(i) for i in range(self.theta_len)]
        self.graphic_use_variable_name = False

    def set_variable_names(self, name_list):
        self.variable_names = name_list
        self.graphic_use_variable_name=True

    def show_one_step_forecast(self, obs_dim_idx=0, show=True, title_str="one-step forecasting"):
        "univariate y_t case: set dim_idx=0"
        forecast_f, forecast_Q = self.dlm_inst.get_one_step_forecast_f_Q()

        plt.figure(figsize=(10, 5))
        plt.tight_layout()

        plt.scatter(range(1,self.y_len+1), [y[obs_dim_idx] for y in self.y], s=10) #blue dot: obs
        plt.plot(range(1,self.y_len+1), [f[obs_dim_idx] for f in forecast_f], color="green") #green: one-step forecast E(Y_t|D_{t-1})
        cred_interval_upper = [f[obs_dim_idx] + t*np.sqrt(q[obs_dim_idx][obs_dim_idx]) for f, q, t in zip(forecast_f, forecast_Q, self.t_q)]
        cred_interval_lower = [f[obs_dim_idx] - t*np.sqrt(q[obs_dim_idx][obs_dim_idx]) for f, q, t in zip(forecast_f, forecast_Q, self.t_q)]
        plt.plot(range(1,self.y_len+1), cred_interval_upper, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
        plt.plot(range(1,self.y_len+1), cred_interval_lower, color="grey") #one-step forecast (Y_t|D_{t-1}) 95% credible interval
        plt.title(title_str)
        if show:
            plt.show()

    def show_filtering_specific_dim(self, param_idx, obs_dim_idx, show=False):
        posterior_mt, posterior_ct = self.dlm_inst.get_posterior_m_C()
                
        plt.scatter(range(1,self.y_len+1), [y[obs_dim_idx] for y in self.y], s=10) #blue dot: obs
        plt.plot(range(1,self.y_len+1), [m[param_idx] for m in posterior_mt], color="orange") #orange: posterior E(theta_t|D_t)
        cred_interval_upper = [m[param_idx] + t*np.sqrt(c[param_idx][param_idx]) for m, c, t in zip(posterior_mt, posterior_ct, self.t_q)]
        cred_interval_lower = [m[param_idx] - t*np.sqrt(c[param_idx][param_idx]) for m, c, t in zip(posterior_mt, posterior_ct, self.t_q)]
        plt.plot(range(1,self.y_len+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
        plt.plot(range(1,self.y_len+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{t}) 95% credible interval
        plt.ylabel("filtering: "+str(self.variable_names[param_idx]))
        if show:
            plt.show()

    def show_filtering(self, figure_grid_dim, choose_param_dims:None|list=None, obs_dim_idxs:None|list=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_param_dims is None:
            choose_param_dims = [i for i in range(self.theta_len)]
        if obs_dim_idxs is None:
            obs_dim_idxs = [0 for _ in range(len(choose_param_dims))]

        plt.figure(figsize=(6*grid_column, 3*grid_row))
        
        # plt.tight_layout(pad=0.5)

        for i, (param_dim, obs_dim) in enumerate(zip(choose_param_dims, obs_dim_idxs)):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_filtering_specific_dim(param_dim, obs_dim, show=False)
        if show:
            plt.show()


    def show_smoothing_specific_dim(self, param_idx, obs_dim_idx, show=False):
        retro_a_at_T, retro_R_at_T = self.dlm_inst.get_retrospective_a_R()
        rT = self.dlm_inst.t_smoothing_given_Dt

        plt.scatter(range(1,self.y_len+1), [y[obs_dim_idx] for y in self.y], s=10) #blue dot: obs

        plt.plot(range(1,rT+1), [a[param_idx] for a in retro_a_at_T], color="red")
        cred_interval_upper = [a[param_idx] + self.z_q*np.sqrt(r[param_idx][param_idx]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
        cred_interval_lower = [a[param_idx] - self.z_q*np.sqrt(r[param_idx][param_idx]) for a, r in zip(retro_a_at_T, retro_R_at_T)]
        plt.plot(range(1,rT+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
        plt.plot(range(1,rT+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
        plt.ylabel("smoothing: "+str(self.variable_names[param_idx]))
        if show:
            plt.show()

    def show_smoothing(self, figure_grid_dim, choose_param_dims:None|list=None, obs_dim_idxs:None|list=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_param_dims is None:
            choose_param_dims = [i for i in range(self.theta_len)]
        if obs_dim_idxs is None:
            obs_dim_idxs = [0 for _ in range(len(choose_param_dims))]

        plt.figure(figsize=(6*grid_column, 3*grid_row))
        
        # plt.tight_layout(pad=0.5)

        for i, (param_dim, obs_dim) in enumerate(zip(choose_param_dims, obs_dim_idxs)):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_smoothing_specific_dim(param_dim, obs_dim, show=False)
        if show:
            plt.show()

    def show_added_smoothing(self, obs_dim_idx=0, show=True, title_str="smoothing"):
        retro_a_at_T, retro_R_at_T = self.dlm_inst.get_retrospective_a_R()
        rT = self.dlm_inst.t_smoothing_given_Dt

        sum_retro_a = [np.sum(at) for at in retro_a_at_T]
        sum_retro_R = [np.sum(Rt) for Rt in retro_R_at_T]
        
        plt.figure(figsize=(10, 5))
        plt.tight_layout()

        plt.scatter(range(1,self.y_len+1), [y[obs_dim_idx] for y in self.y], s=10) #blue dot: obs
        plt.plot(range(1,rT+1), [a for a in sum_retro_a], color="red")
        cred_interval_upper = [a + self.z_q*np.sqrt(r) for a, r in zip(sum_retro_a, sum_retro_R)]
        cred_interval_lower = [a - self.z_q*np.sqrt(r) for a, r in zip(sum_retro_a, sum_retro_R)]
        plt.plot(range(1,rT+1), cred_interval_upper, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
        plt.plot(range(1,rT+1), cred_interval_lower, color="grey") #posterior (\theta_t|D_{T}) 95% credible interval
        plt.title(title_str)
        if show:
            plt.show()

    def show_forecasting(self, show=True, print_summary=True, print_rounding=3):
        fo_mean, fo_q = self.dlm_inst.get_forecast_f_Q()
        
        ft = self.dlm_inst.t_forecasting_given_Dt
        fK = self.dlm_inst.K_forecasting_given_Dt

        plt.plot(range(ft+1, fK+1), [f[0] for f in fo_mean], color="green")
        cred_interval_upper = [f[0]+self.z_q*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]
        cred_interval_lower = [f[0]-self.z_q*np.sqrt(q[0][0]) for f, q in zip(fo_mean, fo_q)]
        plt.plot(range(ft+1, fK+1), cred_interval_upper, color="grey")
        plt.plot(range(ft+1, fK+1), cred_interval_lower, color="grey")
        if print_summary:
            for i in range(fK-ft):
                print(str(i+1)+" step ahead, ", "mean:", round(fo_mean[i][0],print_rounding), " variance:", round(fo_q[i][0][0],print_rounding), 
                    " 95% CI:(", round(cred_interval_lower[i],print_rounding),",",round(cred_interval_upper[i],print_rounding),")")
        if show:
            plt.show()


if __name__=="__main__":
    #example: from stat 223 final
    
    delta1 = 0.95
    true_model1_inst = DLM_model_container(102)
    true_model1_inst.set_F_const_design_mat(np.array([[1],[0]]))
    true_model1_inst.set_G_const_transition_mat(np.array([
        [1,1],
        [0,1]
    ]))
    true_model1_inst.set_V_const_obs_eq_covariance(np.array([[100]]))
    true_model1_inst.set_W_const_state_error_cov(100*np.array([
        [1-(delta1**2), (1-delta1)**2],
        [(1-delta1)**2, (1-delta1)**3]
    ]))
    simulator1_inst = DLM_simulator(true_model1_inst, 20230318)
    simulator1_inst.simulate_data(np.array([100, 0]), 
                                100*np.array([[1-delta1**2, (1-delta1)**2],[(1-delta1)**2, (1-delta1)**3]]))
    data1_theta, data1_y = simulator1_inst.get_theta_y()
    
    # === 2d

    model2d_d095_container = DLM_model_container(102+10)
    model2d_d095_container.set_F_const_design_mat(np.array([[1],[0]]))
    model2d_d095_container.set_G_const_transition_mat(np.array([
        [1,1],
        [0,1]
    ]))
    model2d_d095_container.set_Wst_const_state_error_scale_free_cov(np.array([
        [(1-0.95**2), (1-0.95)**2],
        [(1-0.95)**2, (1-0.95)**3]
    ]))
    model2d_d095_fit_inst = DLM_univariate_y_without_V_in_D0(
        data1_y, model2d_d095_container,
        np.array([0,0]),
        np.array([[1,0],[0,1]]),
        0.01,
        1
    )
    model2d_d095_fit_inst.run()
    model2d_d095_fit_inst.run_retrospective_analysis()
    model2d_d095_fit_inst.run_forecast_analysis(102, 112)

    model2d_d095_vis_inst = DLM_visualizer(model2d_d095_fit_inst, 0.95, True)
    model2d_d095_vis_inst.set_variable_names(["theta1", "theta2"])
    model2d_d095_vis_inst.show_one_step_forecast()
    model2d_d095_vis_inst.show_filtering((1,2))
    model2d_d095_vis_inst.show_smoothing((1,2))
    model2d_d095_vis_inst.show_one_step_forecast(show=False, title_str="")
    model2d_d095_vis_inst.show_forecasting()
    plt.show()