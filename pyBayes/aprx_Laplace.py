import numpy as np

from optim_newton import NewtonUnconstrained

class IntegralLaplaceApprox: #need test
    #int{g(x)*exp{n*h(x)}}dx, x\in R^p, g:R^p to R, h:R^p to r
    def __init__(self, g_coeff, h_exponent, h_exponent_gradient, h_exponent_hessian):
        self.g_coeff = g_coeff
        self.h_exponent = h_exponent
        self.h_exponent_gradient = h_exponent_gradient
        self.h_exponent_hessian = h_exponent_hessian
        # self.p_dim = np.size(h_exponent)
        self.h_mode = None

    def find_mode_of_h(self, newton_initial, 
            tolerance=0.001, method = "cholesky", 
            a_slope_flatter_ratio=0.2, b_step_shorten_ratio=0.5):
        
        #for maximum
        def neg_h(eval_pt):
            return self.h_exponent(eval_pt) * (-1)  
        
        def neg_h_gradient(eval_pt):
            return self.h_exponent_gradient(eval_pt) * (-1)
        
        def neg_h_hessian(eval_pt):
            return self.h_exponent_hessian(eval_pt) * (-1)

        # fn_objective, fn_objective_gradient, fn_objective_hessian, fn_objective_domain_indicator = None        
        self.newton_inst = NewtonUnconstrained(neg_h, neg_h_gradient, neg_h_hessian)
        self.newton_inst.run_newton_with_backtracking_line_search(
            starting_pt = newton_initial, 
            tolerance = tolerance, 
            method = method,
            a_slope_flatter_ratio = a_slope_flatter_ratio, 
            b_step_shorten_ratio = b_step_shorten_ratio)
        self.h_mode = self.newton_inst.get_arg_min()

    def find_approximated_integral_value(self, M):
        from math import pi
        p_dim = self.h_mode.size
        const_term = (2*pi/M)**(p_dim/2)
        numerator = self.g_coeff(self.h_mode) * np.exp(self.h_exponent(self.h_mode) * M)
        denominator = np.linalg.det(self.h_exponent_hessian(self.h_mode)*(-1))**(0.5)
        return const_term*numerator/denominator



class DistLaplaceApprox:
    def __init__(self, density_log_kernel, 
        density_log_kernel_gradient, 
        density_log_kernel_hessian) -> None:
        
        self.log_kernel = density_log_kernel
        self.log_kernel_gradient = density_log_kernel_gradient
        self.log_kernel_hessian = density_log_kernel_hessian
        
        self.newton_inst = None
        self.laplace_approx_mean = None
        self.laplace_approx_precision = None
        self.laplace_approx_covariance = None


    def _cal_gaussian_params(self):
        self.laplace_approx_mean = self.newton_inst.get_arg_min()
        self.laplace_approx_precision = self.log_kernel_hessian(self.laplace_approx_mean) * (-1)
        self.laplace_approx_covariance = np.linalg.inv(self.laplace_approx_precision)


    def fit(self, newton_initial, 
            tolerance=0.001, method = "cholesky", 
            a_slope_flatter_ratio=0.2, b_step_shorten_ratio=0.5):
        
        #for maximum
        def neg_log_kernel(eval_pt):
            return self.log_kernel(eval_pt) * (-1)  
        
        def neg_log_kernel_gradient(eval_pt):
            return self.log_kernel_gradient(eval_pt) * (-1)
        
        def neg_log_kernel_hessian(eval_pt):
            return self.log_kernel_hessian(eval_pt) * (-1)

        # fn_objective, fn_objective_gradient, fn_objective_hessian, fn_objective_domain_indicator = None        
        self.newton_inst = NewtonUnconstrained(neg_log_kernel, neg_log_kernel_gradient, neg_log_kernel_hessian)
        self.newton_inst.run_newton_with_backtracking_line_search(
            starting_pt = newton_initial, 
            tolerance = tolerance, 
            method = method,
            a_slope_flatter_ratio = a_slope_flatter_ratio, 
            b_step_shorten_ratio = b_step_shorten_ratio)

        self._cal_gaussian_params()



    def get_mode(self):
        return self.laplace_approx_mean
    
    def get_laplace_approximated_mean(self):
        return self.laplace_approx_mean

    def get_maximizing_sequence(self):
        return self.newton_inst.get_minimizing_sequence()
    
    def get_maximizing_function_value_sequence(self):
        seq = [-x for x in self.newton_inst.get_minimizing_function_value_sequence()]
        return seq
    
    def get_mode_log_ker_value(self):
        return self.newton_inst.get_min() * (-1)
    
    def get_decrement_sequence(self):
        return self.newton_inst.get_decrement_sequence()

    
    def get_laplace_approximated_inv_variance(self) -> np.array:
        return self.laplace_approx_precision

    def get_laplace_approximated_variance(self) -> np.array:
        return self.laplace_approx_covariance


    def get_laplace_approximated_samples(self, num_sample, seed) -> np.array:
        rn_generator = np.random.default_rng(seed=seed)
        gaussian_samples = rn_generator.multivariate_normal(
            self.laplace_approx_mean, 
            self.laplace_approx_covariance,
            num_sample)
        return gaussian_samples

    def get_laplace_approximated_MCproposal_sampler(self, seed=None): #need test
        if seed is not None:
            rn_generator = np.random.default_rng(seed=seed)
        else:
            rn_generator = np.random.default_rng()
        def proposal_sampler(_):
            gaussian_samples = rn_generator.multivariate_normal(
                self.laplace_approx_mean, 
                self.laplace_approx_covariance,
                num_sample=1)
            return gaussian_samples
        return proposal_sampler

if __name__ == "__main__":
    # test an integral
    #later....(......)
    

    # # test a posterior approximation(STAT206 HW3 problem8): laplace approximation for posterior with gumbel data
    rn_generator = np.random.default_rng(seed=20220216)

    def gumbel2_inverse_cdf(y, param_alpha, param_beta):
        return (-np.log(y)/param_beta)**(-1/param_alpha)

    def gumbel2_random_number_generator(num_smpl, param_alpha, param_beta):
        unif_smpl = rn_generator.random(num_smpl)
        gumbel2_smpl = np.array([gumbel2_inverse_cdf(y, param_alpha, param_beta) for y in unif_smpl])
        return gumbel2_smpl
        
    # generate sample
    gumbel2_smpl_55 = gumbel2_random_number_generator(500, 5, 5) #<- true parameters

    # posterior kernel functions
    def unif_gumbel2_log_posterior(prior_param_vec, data):
        #log q (in the note)
        prior_alpha, prior_beta = prior_param_vec
        n = data.size
        negalpha_powered_data_sum = np.sum(data**(-prior_alpha))
        logdata_sum = np.sum(np.log(data))

        log_q_val = (n * (np.log(prior_alpha) + np.log(prior_beta))
            + (1-prior_alpha)*logdata_sum
            - prior_beta*negalpha_powered_data_sum)

        return log_q_val

    def unif_gumbel2_log_posterior_gradient(prior_param_vec, data):
        #dlog q/da, logq/db (in the note)
        prior_alpha, prior_beta = prior_param_vec
        n = data.size
        negalpha_powered_data_times_log_data_sum = np.sum(data**(-prior_alpha) * np.log(data))
        negalpha_powered_data_sum = np.sum(data**(-prior_alpha))
        logdata_sum = np.sum(np.log(data))

        dlogq_da = n/prior_alpha - logdata_sum + prior_beta * negalpha_powered_data_times_log_data_sum
        dlogq_db = n/prior_beta - negalpha_powered_data_sum

        return np.array([dlogq_da, dlogq_db])

    def unif_gumbel2_log_posterior_hessian(prior_param_vec, data):
        #log q (in the note)
        prior_alpha, prior_beta = prior_param_vec
        n = data.size
        negalpha_powered_data_times_squared_log_data_sum = np.sum((data**(-prior_alpha)) * (np.log(data)**2))
        negalpha_powered_data_times_log_data_sum = np.sum(data**(-prior_alpha) * np.log(data))

        h11 = - n/prior_alpha**2 - prior_beta * negalpha_powered_data_times_squared_log_data_sum
        h12 = negalpha_powered_data_times_log_data_sum
        h22 = -n/prior_beta**2

        return np.array([[h11, h12],[h12,h22]])

    import functools
    newton_objective = functools.partial(unif_gumbel2_log_posterior, data=gumbel2_smpl_55)
    newton_gradient = functools.partial(unif_gumbel2_log_posterior_gradient, data=gumbel2_smpl_55)
    newton_hessian = functools.partial(unif_gumbel2_log_posterior_hessian, data=gumbel2_smpl_55)

    
    laplace_inst = DistLaplaceApprox(newton_objective, newton_gradient, newton_hessian)
    laplace_inst.fit(np.array([8, 10]))


    print(laplace_inst.get_mode())
    print(laplace_inst.get_maximizing_sequence())
    print(laplace_inst.get_maximizing_function_value_sequence())
    print(laplace_inst.get_mode_log_ker_value())

    print(laplace_inst.get_laplace_approximated_inv_variance())
    print(laplace_inst.get_laplace_approximated_variance())
    print(laplace_inst.get_laplace_approximated_samples(10, seed = 20220316))