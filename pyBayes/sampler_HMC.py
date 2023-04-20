from math import log
from random import uniform, normalvariate, seed

import numpy as np

from MCMC_Core import MCMC_base


class MC_Hamiltonian(MCMC_base):
    #override
    def __init__(self, log_target_pdf, log_target_gradient, 
                M_momentum_generater_var_1dlist: list, L_leapfrog_step_num, epsilon_leap_step_size, initial: np.array, random_seed=None) -> None:
        self.dim = len(initial)
        self.log_target_pdf = log_target_pdf #arg (smpl_pt)
        self.log_target_gradient = log_target_gradient #arg (smpl_pt)
        
        self.leapfrog_step_num = L_leapfrog_step_num
        self.step_size = epsilon_leap_step_size
        if len(M_momentum_generater_var_1dlist) != len(initial):
            raise ValueError("M should have the same length as initial value array")
        self.momentum_gen_Normal_cov = M_momentum_generater_var_1dlist
        self.momentum_gen_Normal_cov_inv = [1/x for x in M_momentum_generater_var_1dlist]

        self.MC_sample  = [initial]

        self.num_total_iters = 0
        self.num_accept = 0

        self.underflowexception_count = 0
        self.pid = None

        self.random_seed = random_seed
        if random_seed is not None:
            seed(random_seed)
    
    def _start_momentum_generator(self) -> np.array:
        new_momentum = np.array([normalvariate(0, m**0.5) for m in self.momentum_gen_Normal_cov])
        return new_momentum

    def _leap_frog_step(self, start_position, start_momentum):
        end_position = np.array([x for x in start_position], dtype="float64") #x
        end_momentum = np.array([x for x in start_momentum], dtype="float64") #z
        for _ in range(self.leapfrog_step_num):
            end_momentum += (0.5*self.step_size*self.log_target_gradient(end_position))
            end_position += (self.step_size * (np.diag(self.momentum_gen_Normal_cov_inv) @ end_momentum))
            end_momentum += (0.5*self.step_size*self.log_target_gradient(end_position))
        return (end_position, end_momentum)

    def _normal_log_pdf_under_M(self, x_vec) -> float:
        #log(pdf of N(x; 0, M))
        #need only kernel part
        log_kernel = -0.5 * sum([m*(x**2) for x, m in zip(x_vec, self.momentum_gen_Normal_cov_inv)])
        return log_kernel

    def _log_r_calculator(self, start_position, start_momentum, 
                            end_position, end_momentum) -> float:
        log_r = self.log_target_pdf(end_position) + self._normal_log_pdf_under_M(end_momentum) \
            - self.log_target_pdf(start_position) - self._normal_log_pdf_under_M(start_momentum)
        return log_r

    def _MH_rejection_step(self, start_position, start_momentum, 
                            end_position, end_momentum, underflow_protection=False) -> bool:
        try:
            log_r = self._log_r_calculator(start_position, start_momentum, end_position, end_momentum)
        except ValueError("underflow") as e: #underflow error catch (May have convergence issues)
            if underflow_protection:
                self.underflowexception_count += 1
                return False
            else:
                raise e
        else:
            unif_sample = uniform(0,1)
            if log(unif_sample) < log_r:
                return True
            else:
                return False
        
    def sampler(self, **kwargs):
        start_position = self.MC_sample[-1]
        start_momentum = self._start_momentum_generator()
        end_position, end_momentum = self._leap_frog_step(start_position, start_momentum)
        accept_bool = self._MH_rejection_step(start_position, start_momentum, end_position, end_momentum)

        self.num_total_iters += 1
        if accept_bool:
            self.MC_sample.append(end_position)
            self.num_accept += 1
        else:
            self.MC_sample.append(start_position)

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        super().generate_samples(num_samples, pid, verbose, print_iter_cycle)
        print("acceptance rate:", self.num_accept/self.num_total_iters)
        print("(underflow protection:", self.underflowexception_count, ")")



if __name__ == "__main__":
    from functools import partial
    from MCMC_Core import MCMC_Diag

    def normal_log_pdf(x_vec, mu, inv_sigma):
        return -0.5 * np.log(1/np.linalg.det(inv_sigma)) - 0.5 * (np.transpose(x_vec-mu) @ inv_sigma @ (x_vec-mu))
    def normal_log_pdf_gradient(x_vec, mu, inv_sigma):
        return inv_sigma @ (mu - x_vec)

    now_mu = np.array([1, 10])
    now_sigma = np.array([[2,-1],
                          [-1,1]])
    now_inv_sigma = np.linalg.inv(now_sigma)
    now_log_target = partial(normal_log_pdf, mu=now_mu, inv_sigma=now_inv_sigma)
    now_log_target_gradient = partial(normal_log_pdf_gradient, mu=now_mu, inv_sigma=now_inv_sigma)

    print(now_log_target(np.array([0,0])))
    print(now_log_target_gradient(np.array([0,0])))

    hmc_inst = MC_Hamiltonian(now_log_target, now_log_target_gradient, [1, 1], 10, 0.1, np.array([0,0]), 20220508)
    hmc_inst.generate_samples(10000)
    
    diag_inst = MCMC_Diag()
    diag_inst.set_mc_sample_from_MCMC_instance(hmc_inst)
    diag_inst.burnin(1000)
    diag_inst.show_scatterplot(0,1)
    diag_inst.show_traceplot((1,2))
    diag_inst.show_acf(30,(1,2))