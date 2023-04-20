import time, csv
from math import log, exp
from random import seed, random

import numpy as np

from optim_decent import DescentUnconstrained
# from optim_newton 

# class rejection_sampler():
#     def __init__(self, log_target_pdf, log_proposal_pdf, proposal_sampler, random_seed):
#         self.log_target_pdf = log_target_pdf #arg (smpl)
        
#         self.log_proposal_pdf = log_proposal_pdf #arg (smpl)
#         self.proposal_sampler = proposal_sampler #function with argument (smpl)
        
#         self.MC_sample = [] #no initial

#         self.num_total_iters = 0
#         self.num_accept = 0

#         self.random_seed = random_seed
#         seed(random_seed)

#     def log

class PosteriorRejectionSampler():
    def __init__(self, log_likelihood, prior_sampler, random_seed):
        self.log_likelihood = log_likelihood
        self.prior_sampler = prior_sampler
        self.log_likelihood_supremum = None

        self.random_seed = random_seed
        seed(random_seed)

        self.MC_sample = [] #no initial

        self.num_total_iters = 0
        self.num_accept = 0

    def find_sup_log_likelihood_by_decent(self, log_likelihood_gradient, initial, tolerance=0.0001):
        def neg_log_likelihood(eval_pt):
            return np.array(self.log_likelihood(eval_pt))*(-1)
        def neg_log_likelihood_gradient(eval_pt):
            return np.array(log_likelihood_gradient(eval_pt))*(-1)
        optim_inst = DescentUnconstrained(neg_log_likelihood, neg_log_likelihood_gradient)
        optim_inst.run_gradient_descent_with_backtracking_line_search(np.array(initial), tolerance)
        supremum_log_likelihood = optim_inst.get_min() * (-1)
        print("Maximum log-likelihood value:", supremum_log_likelihood, "at", optim_inst.get_arg_min())
        self.log_likelihood_supremum = supremum_log_likelihood + tolerance

    # def find_sup_log_likelihood_by_newton(self, log_likelihood_gradient, log_likelihood_hessian, initial, tolerance=0.0001):
    #     pass

    def set_sup_log_likelihood_directly(self, sup_value):
        self.log_likelihood_supremum = sup_value

    def _log_r_calculator(self, candid):
        log_r = self.log_likelihood(candid) - self.log_likelihood_supremum
        return log_r

    def sampler(self, **kwrgs):
        if self.log_likelihood_supremum is None:
            raise AttributeError("find or set the sup_log_likelihood value first")
        
        candid_sample = self.prior_sampler()
        log_unif_sample = log(random())
        log_r = self._log_r_calculator(candid_sample)
        if log_unif_sample < log_r:
            self.MC_sample.append(candid_sample)
            self.num_accept += 1
        self.num_total_iters += 1
    
    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        start_time = time.time()
        while self.num_accept < num_samples:
            self.sampler(iter_idx = self.num_total_iters)
            
            if self.num_total_iters==100 and verbose:
                elap_time_head_iter = time.time()-start_time
                estimated_time = (num_samples/self.num_accept)*elap_time_head_iter
                print("estimated running time: ", estimated_time//60, "min ", estimated_time%60, "sec")

            if self.num_total_iters%print_iter_cycle == 0 and verbose and pid is not None:
                print("pid:",pid," iteration", self.num_total_iters, "accepted: ", self.num_accept, "/", num_samples)
            elif self.num_total_iters%print_iter_cycle == 0 and verbose and pid is None:
                print("iteration", self.num_total_iters, "accepted: ", self.num_accept, "/", num_samples)
        elap_time = time.time()-start_time
        
        if pid is not None and verbose:
            print("pid:",pid, "generating", num_samples, "/", num_samples, " samples - done! \
                \n(total number of iterations:", self.num_total_iters,")\n(elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        elif pid is None and verbose:
            print("generating", num_samples, "/", num_samples, " samples - done! \
                \n(total number of iterations:", self.num_total_iters,")\n(elapsed time for execution: ", elap_time//60,"min ", elap_time%60,"sec)")
        print("acceptance rate: ", round(self.num_accept / self.num_total_iters, 4))

    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)

    def empirical_normalized_constant(self):
        print("Warning: for this function, you had to specify the log_likelihood function with its full constants")
        return exp(self.log_likelihood_supremum) * self.num_accept/self.num_total_iters


if __name__ == "__main__":
    #ex1
    from math import inf
    def prior_sampler_ex1():
        return [random()]
    def log_likelihood_ex1(eval_pt):
        p = eval_pt[0]
        if p < 0:
            return -inf
        if p > 1:
            return -inf
        return np.array([log(10*9*8/6) + 7*log(p) + 3*log(1-p)])
    def log_likelihood_gradient_ex1(eval_pt):
        p = eval_pt[0]
        return np.array([7/p - 3/(1-p)])

    ex1_inst = PosteriorRejectionSampler(log_likelihood_ex1, prior_sampler_ex1, 20220417)
    ex1_inst.find_sup_log_likelihood_by_decent(log_likelihood_gradient_ex1, [0.1], tolerance=0.001) #first argument : gradient!!!
    # ex1_inst.set_sup_log_likelihood_directly(log(10*9*8/6) + 7*log(0.7) + 3*log(1-0.7))
    ex1_inst.generate_samples(10000, print_iter_cycle=5000)
    
    from MCMC_Core import MCMC_Diag
    ex1_diag_inst = MCMC_Diag()
    ex1_diag_inst.set_mc_sample_from_MCMC_instance(ex1_inst)
    ex1_diag_inst.show_hist((1,1))
