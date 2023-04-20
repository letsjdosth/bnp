import numpy as np

from DLM_Core import DLM_model_container, DLM_simulator, DLM_utility

class Kalman_filter_smoother:
    def __init__(self, y_observation, D0: DLM_model_container, initial_mu0_given_D0, initial_P0_given_D0):
        self.util_inst = DLM_utility()

        #input
        self.y_observation = self.util_inst.vectorize_seq(y_observation)
        self.D0 = D0
        self.mu0 = initial_mu0_given_D0
        self.P0 = initial_P0_given_D0

        #result containers
        ## filtering
        self.theta_one_step_forecast = [] #x_t^{t-1} in Shumway-Stoffer, f_t=E(\theta_t|D_{t-1}) in West
        self.P_one_step_forecast = [] # P_t^{t-1} in Shumway-Stoffer, R_t=Var(\theta_t|D_{t-1}) in West

        self.theta_on_time = [initial_mu0_given_D0] #x_t^{t} in Shumway-Stoffer, m_t=E(\theta_t|D_{t}) in West
        self.P_on_time = [initial_P0_given_D0] # P_t^{t} in Shumway-Stoffer, C_t=Var(\theta_t|D_{t}) in West

        self.innovation = []
        self.innovation_cov = []
        self.kalman_gain = []

        ## smoothing
        self.theta_smoothed_reversed = []
        self.P_smoothed_reversed = []
        self.J_smoother_gain_reversed = []

    #filtering
    def _filter_one_iter(self, t):
        # one-step forecast (prior)
        Gt = self.D0.G_sys_eq_transition[t-1]
        ut = self.D0.u_covariate[t-1]
        ut_sys_coeff = self.D0.u_coeff_state_eq_seq[t-1]
        theta_t_one_step_forecast = Gt @ self.theta_on_time[-1] + ut_sys_coeff @ ut
        Pt_one_step_forecast = Gt @ self.P_on_time[-1] @ np.transpose(Gt) + self.D0.W_sys_eq_covariance[t-1]

        # on_time (posterior)
        At = self.D0.F_obs_eq_design[t-1]
        Rt = self.D0.V_obs_eq_covariance[t-1]
        ut_obs_coeff = self.D0.u_coeff_obs_eq_seq[t-1]

        innovation_t = self.y_observation[t-1] - At @ theta_t_one_step_forecast - ut_obs_coeff @ ut
        innovation_t_cov = At @ Pt_one_step_forecast @ np.transpose(At) + Rt
        kalman_gain_t = Pt_one_step_forecast @ At @ np.linalg.inv(innovation_t_cov)

        theta_t_on_time = theta_t_one_step_forecast + kalman_gain_t @ innovation_t
        KtAt = kalman_gain_t @ At
        Pt_on_time = (np.identity(KtAt.shape[0]) - KtAt) @ Pt_one_step_forecast

        # save
        self.theta_one_step_forecast.append(theta_t_one_step_forecast)
        self.P_one_step_forecast.append(Pt_one_step_forecast)

        self.theta_on_time.append(theta_t_on_time)
        self.P_on_time.append(Pt_on_time)

        self.innovation.append(innovation_t)
        self.innovation_cov.append(innovation_t_cov)
        self.kalman_gain.append(kalman_gain_t)


    def run_filter(self):
        #check everything is set
        #update
        for t in range(1, self.D0.y_len+1):
            self._filter_one_iter(t)

        # delete initial value
        self.theta_on_time = self.theta_on_time[1:]
        self.P_on_time = self.P_on_time[1:]

    #smoothing
    def _smoother_one_iter(self, t_minus_1):
        #smoother
        Jt1 = self.P_on_time[t_minus_1-1] @ np.transpose(self.D0.G_sys_eq_transition[t_minus_1]) @ np.linalg.inv(self.P_one_step_forecast[t_minus_1])
        theta_t1_smoothed = self.theta_on_time[t_minus_1-1] + Jt1 @ (self.theta_smoothed_reversed[-1] - self.theta_one_step_forecast[t_minus_1])
        Pt1_smoothed = self.P_on_time[t_minus_1-1] + Jt1 @ (self.P_smoothed_reversed[-1] - self.P_one_step_forecast[t_minus_1]) @ np.transpose(Jt1)

        #save
        self.theta_smoothed_reversed.append(theta_t1_smoothed)
        self.P_smoothed_reversed.append(Pt1_smoothed)
        self.J_smoother_gain_reversed.append(Jt1)

    def run_smoother(self):
        #run it after running filter
        self.theta_smoothed_reversed.append(self.theta_on_time[-1])
        self.P_smoothed_reversed.append(self.P_on_time[-1])
        self.J_smoother_gain_reversed = []
        for t_minus_1 in range(self.D0.y_len-1, 0, -1):
            self._smoother_one_iter(t_minus_1)

    # == getters ==
    def get_filtered_on_time_theta_P(self):
        return self.theta_on_time, self.P_on_time

    def get_filtered_one_step_forecast_theta_P(self):
        return self.theta_one_step_forecast, self.P_one_step_forecast

    def get_smoothed_theta_P(self):
        theta_smoothed = list(reversed(self.theta_smoothed_reversed))
        P_smoothed = list(reversed(self.P_smoothed_reversed))
        return theta_smoothed, P_smoothed



if __name__=="__main__":
    import matplotlib.pyplot as plt
    test1 = True
    test2 = True

    if test1:
        test1_W = [np.array([[1]]) for _ in range(100)]
        test1_V = [np.array([[0.1]]) for _ in range(100)]
        test1_F = [np.array([[1]]) for _ in range(100)]
        test1_G = [np.array([[0.9]]) for _ in range(100)]
        test1_D0 = DLM_model_container(100)
        test1_D0.set_Ft_design_mat(test1_F)
        test1_D0.set_Gt_transition_mat(test1_G)
        test1_D0.set_Vt_obs_eq_covariance(test1_V)
        test1_D0.set_Wt_state_error_cov(test1_W)
        test1_D0.set_u_no_covariate()

        test1_sim_inst = DLM_simulator(test1_D0, 20220814)
        test1_sim_inst.simulate_data(np.array([0]), np.array([[1]]))
        test1_theta_seq, test1_y_seq = test1_sim_inst.get_theta_y()
        # print(test1_theta_seq)
        # print(test1_y_seq)

        plt.plot(range(100), test1_theta_seq)
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs
        plt.show()

        test1_filter_inst = Kalman_filter_smoother(test1_y_seq, test1_D0, np.array([0]), np.array([[1]]))
        test1_filter_inst.run_filter()
        test1_filtered_theta_on_time = test1_filter_inst.theta_on_time
        test1_filtered_theta_one_step_forecast = test1_filter_inst.theta_one_step_forecast
        # print(test1_filtered_theta_on_time)
        plt.plot(range(100), test1_theta_seq) #blue: true theta
        plt.plot(range(100), test1_filtered_theta_on_time) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), test1_filtered_theta_one_step_forecast) #green: prior E(theta_t|D_{t-1})
        plt.scatter(range(100), test1_y_seq, s=10) #blue dot: obs

        test1_filter_inst.run_smoother()
        test1_smoothed_theta,_ = test1_filter_inst.get_smoothed_theta_P()
        # print(test1_smoothed_theta)

        plt.plot(range(100), test1_smoothed_theta) #red: smoothed theta
        plt.show()

    if test2:
        test2_W = [np.array([[1, 0.8],[0.8, 1]]) for _ in range(100)]
        test2_V = [np.array([[0.1, 0],[0, 0.1]]) for _ in range(100)]
        test2_F = [np.array([[1,0],[0,1]]) for _ in range(100)]
        test2_G = [np.array([[0.9, 0], [0, 0.5]]) for _ in range(100)]
        test2_D0 = DLM_model_container(100)
        test2_D0.set_Ft_design_mat(test2_F)
        test2_D0.set_Gt_transition_mat(test2_G)
        test2_D0.set_Vt_obs_eq_covariance(test2_V)
        test2_D0.set_Wt_state_error_cov(test2_V)
        test2_D0.set_u_no_covariate()

        test2_sim_inst = DLM_simulator(test2_D0, 20220814)
        test2_sim_inst.simulate_data(np.array([0,0]), np.array([[1,0],[0,1]]))
        test2_theta_seq, test2_y_seq = test2_sim_inst.get_theta_y()

        print(test2_theta_seq)
        print(test2_y_seq)
        plt.plot(range(100), [x[0] for x in test2_theta_seq]) #blue: true theta
        plt.scatter(range(100), [x[0] for x in test2_y_seq], s=10) #blue dot: obs
        plt.show()
        # plt.plot(range(100), [x[1] for x in test2_theta_seq])
        # plt.plot(range(100), [x[1] for x in test2_y_seq])
        # plt.show()

        test2_filter_inst = Kalman_filter_smoother(test2_y_seq, test2_D0, np.array([0,0]), np.array([[1,0],[0,1]]))
        test2_filter_inst.run_filter()
        test2_filtered_theta_on_time = test2_filter_inst.theta_on_time
        test2_filtered_theta_one_step_forecast = test2_filter_inst.theta_one_step_forecast
        print(test2_filtered_theta_on_time)
        plt.plot(range(100), [x[0] for x in test2_theta_seq]) #blue: true theta
        plt.plot(range(100), [x[0] for x in test2_filtered_theta_on_time]) #orange: posterior E(theta_t|D_t)
        plt.plot(range(100), [x[0] for x in test2_filtered_theta_one_step_forecast]) #green: prior E(theta_t|D_{t-1})
        plt.scatter(range(100), [x[0] for x in test2_y_seq], s=10) #blue dot: obs

        test2_filter_inst.run_smoother()
        test2_smoothed_theta, _ = test2_filter_inst.get_smoothed_theta_P()
        plt.plot(range(100), [x[0] for x in test2_smoothed_theta]) #red: smoothed theta
        plt.show()
