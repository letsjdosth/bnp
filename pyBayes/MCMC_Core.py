import time
import csv
from math import log
from random import seed, uniform

import numpy as np
import matplotlib.pyplot as plt


class MCMC_base:
    def __init__(self, initial):
        self.MC_sample = [initial]

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


class MCMC_Gibbs(MCMC_base):
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
        last = self.MC_sample[-1]
        new = self.deep_copier(last)
        #update new
        new = self.full_conditional_sampler(new)

        self.MC_sample.append(new)
    
    def full_conditional_sampler(self, last_param):
        new_sample = [x for x in last_param]
        #update new

        return new_sample


class MCMC_MH(MCMC_base):
    #override
    def __init__(self, log_target_pdf, log_proposal_pdf, proposal_sampler, initial, random_seed=None):
        self.log_target_pdf = log_target_pdf #arg (smpl_pt)
        self.log_proposal_pdf = log_proposal_pdf #arg (from_smpl, to_smpl)
        self.proposal_sampler = proposal_sampler #function with argument (from_smpl)
        
        self.initial = initial
        
        self.MC_sample = [initial]

        self.num_total_iters = 0
        self.num_accept = 0

        self.random_seed = random_seed
        if random_seed is not None:
            seed(random_seed)
        
    def _log_r_calculator(self, candid, last):
        log_r = (self.log_target_pdf(candid) - self.log_proposal_pdf(from_smpl=last, to_smpl=candid) - \
             self.log_target_pdf(last) + self.log_proposal_pdf(from_smpl=candid, to_smpl=last))
        return log_r

    def sampler(self, **kwargs):
        last = self.MC_sample[-1]
        candid = self.proposal_sampler(last)
        unif_sample = uniform(0, 1)
        log_r = self._log_r_calculator(candid, last)
        # print(log(unif_sample), log_r) #for debug
        if log(unif_sample) < log_r:
            self.MC_sample.append(candid)
            self.num_total_iters += 1
            self.num_accept += 1
        else:
            self.MC_sample.append(last)
            self.num_total_iters += 1

    def generate_samples(self, num_samples, pid=None, verbose=True, print_iter_cycle=500):
        super().generate_samples(num_samples, pid=pid, verbose=verbose, print_iter_cycle=print_iter_cycle)
        if verbose:
            print("acceptance rate: ", round(self.num_accept / self.num_total_iters, 4))


class MCMC_Diag:
    def __init__(self):
        self.MC_sample = []
        self.num_dim = None
        
        # variable-name manager
        self.variable_names = None

        # graphical parameters
        self.graphic_traceplot_mean = False
        self.graphic_traceplot_median = False
        self.graphic_hist_mean = True
        self.graphic_hist_median = True
        self.graphic_hist_95CI = True
        self.graphic_scatterplot_mean = False
        self.graphic_scatterplot_median = False

        self.graphic_use_variable_name=False
        #note: self.set_variable_names function can switch this flag
        #note: print function depends on this flag
    
    def set_mc_samples_from_list(self, mc_sample, variable_names=None):
        self.MC_sample = mc_sample
        self.num_dim = len(mc_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)


    def set_mc_sample_from_MCMC_instance(self, inst_MCMC, variable_names=None):
        self.MC_sample = inst_MCMC.MC_sample
        self.num_dim = len(inst_MCMC.MC_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)

    def set_mc_sample_from_csv(self, file_name, variable_names=None):
        with open(file_name + '.csv', 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for csv_row in reader:
                csv_row = [float(elem) for elem in csv_row]
                self.MC_sample.append(csv_row)
        self.num_dim = len(self.MC_sample[0])
        if variable_names is not None:
            self.set_variable_names(variable_names)

    def set_variable_names(self, name_list):
        self.variable_names = name_list
        self.graphic_use_variable_name=True

    def write_samples(self, filename: str):
        with open(filename + '.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for sample in self.MC_sample:
                csv_row = sample
                writer.writerow(csv_row)

    def burnin(self, num_burn_in):
        self.MC_sample = self.MC_sample[num_burn_in-1:]

    def thinning(self, lag):
        self.MC_sample = self.MC_sample[::lag]
    
    def _round_list(self, list_obj, round_digit):
        rounded = [round(x, round_digit) for x in list_obj]
        return rounded

    def get_specific_dim_samples(self, dim_idx):
        if dim_idx >= self.num_dim:
            raise ValueError("dimension index should be lower than number of dimension. note that index starts at 0")
        return [smpl[dim_idx] for smpl in self.MC_sample]
    
    def get_sample_mean(self, round=None):
        mean_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            mean_vec.append(np.mean(ith_dim_samples))
        if round is not None:
            mean_vec = self._round_list(mean_vec, round)
        return mean_vec


    def get_sample_var(self, round=None):
        var_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            var_vec.append(np.var(ith_dim_samples))
        if round is not None:
            var_vec = self._round_list(var_vec, round)
        return var_vec

    def get_sample_quantile(self, quantile_list, round=None):
        quantile_vec = []
        for i in range(self.num_dim):
            ith_dim_samples = self.get_specific_dim_samples(i)
            quantiles = [np.quantile(ith_dim_samples, q) for q in quantile_list]
            quantile_vec.append(quantiles)
        
        if round is not None:
            quantile_vec = [self._round_list(x, round) for x in quantile_vec]
        return quantile_vec

    def print_summaries(self, round = None, latex_table_format=False):
        #name/mean/var/95%CI
        mean_vec = self.get_sample_mean(round=round)
        var_vec = self.get_sample_var(round=round)
        cred95_interval_vec = self.get_sample_quantile([0.025, 0.975], round=round)


        print("param \t\t mean \t var \t 95%CI")
        if self.graphic_use_variable_name:
            for var_name, mean_val, var_val, cred95_vals in zip(self.variable_names, mean_vec, var_vec, cred95_interval_vec):
                if latex_table_format:
                    print(var_name, "&", mean_val, "&", var_val, "&", cred95_vals, "\\\\")
                else:
                    print(var_name, "\t\t", mean_val, "\t", var_val, "\t", cred95_vals)
        else:
            for i, (mean_val, var_val, cred95_vals) in enumerate(zip(mean_vec, var_vec, cred95_interval_vec)):
                if latex_table_format:
                    print(i,"th", "& ", mean_val, "&", var_val, "&", cred95_vals, "\\\\") 
                else:
                    print(i,"th", "\t\t", mean_val, "\t", var_val, "\t", cred95_vals)



    def show_traceplot_specific_dim(self, dim_idx, show=False):
        traceplot_data = self.get_specific_dim_samples(dim_idx)
        plt.plot(range(len(traceplot_data)), traceplot_data)
        if self.graphic_use_variable_name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")

        if self.graphic_traceplot_mean:
            plt.axhline(np.mean(traceplot_data), color="red", linestyle="solid", linewidth=0.8)
        if self.graphic_traceplot_median:
            plt.axhline(np.median(traceplot_data), color="red", linestyle="dashed", linewidth=0.8)

        if show:
            plt.show()

    def show_traceplot(self, figure_grid_dim, choose_dims=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_dims is None:
            choose_dims = range(self.num_dim)

        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, dim_idx in enumerate(choose_dims):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_traceplot_specific_dim(dim_idx)
        if show:
            plt.show()

    
    def show_hist_specific_dim(self, dim_idx, show=False, hist_type="bar"):
        hist_data = self.get_specific_dim_samples(dim_idx)

        plt.hist(hist_data, bins=100, histtype=hist_type, density=True)
        if self.graphic_use_variable_name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")
        
        if self.graphic_hist_mean:
            plt.axvline(np.mean(hist_data), color="red", linestyle="solid", linewidth=0.8)
        
        if self.graphic_hist_median:
            plt.axvline(np.median(hist_data), color="red", linestyle="dashed", linewidth=0.8)

        if self.graphic_hist_95CI:
            quantile_0_95 = self.get_sample_quantile([0.025, 0.975])[dim_idx]
            x_axis_pts = np.linspace(quantile_0_95[0], quantile_0_95[1], num=100)
            y_axis_pts = np.zeros(len(x_axis_pts)) + 0.01
            plt.scatter(x_axis_pts, y_axis_pts, color="red", s=10, zorder=2)

        if show:
            plt.show()

    def show_hist(self, figure_grid_dim, choose_dims=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_dims is None:
            choose_dims = range(self.num_dim)
       
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, dim_idx in enumerate(choose_dims):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_hist_specific_dim(dim_idx)
        if show:
            plt.show()

    def show_hist_superimposed(self, choose_dims=None, show=True, y_lab=None):
        if choose_dims is None:
            choose_dims = range(self.num_dim)
        self.graphic_hist_mean = False
        self.graphic_hist_median = False
        self.graphic_hist_95CI = False
       
        for _, dim_idx in enumerate(choose_dims):
            self.show_hist_specific_dim(dim_idx, hist_type="step")
        if y_lab:
            plt.ylabel(y_lab)

        self.graphic_hist_mean = True
        self.graphic_hist_median = True
        self.graphic_hist_95CI = True
        
        if show:
            plt.show()

    def get_autocorr(self, dim_idx, maxLag):
        y = self.get_specific_dim_samples(dim_idx)
        acf = []
        y_mean = np.mean(y)
        y = [elem - y_mean  for elem in y]
        n_var = sum([elem**2 for elem in y])
        for k in range(maxLag+1):
            N = len(y)-k
            n_cov_term = 0
            for i in range(N):
                n_cov_term += y[i]*y[i+k]
            acf.append(n_cov_term / n_var)
        return acf
    
    def show_acf_specific_dim(self, dim_idx, maxLag, show=False):
        grid = [i for i in range(maxLag+1)]
        acf = self.get_autocorr(dim_idx, maxLag)
        plt.ylim([-1,1])
       
        plt.bar(grid, acf, width=0.3)
        plt.axhline(0, color="black", linewidth=0.8)

        if self.graphic_use_variable_name:
            plt.ylabel(self.variable_names[dim_idx])
        else:
            plt.ylabel(str(dim_idx)+"th dim")

        if show:
            plt.show()

    def show_acf(self, maxLag, figure_grid_dim, choose_dims=None, show=True):
        grid_row = figure_grid_dim[0]
        grid_column= figure_grid_dim[1]
        if choose_dims is None:
            choose_dims = range(self.num_dim)
        
        plt.figure(figsize=(5*grid_column, 3*grid_row))
        for i, dim_idx in enumerate(choose_dims):
            plt.subplot(grid_row, grid_column, i+1)
            self.show_acf_specific_dim(dim_idx, maxLag)
        
        if show:
            plt.show()
    
    def show_scatterplot(self, dim_idx_horizontal, dim_idx_vertical, show=True):
        x = self.get_specific_dim_samples(dim_idx_horizontal)
        y = self.get_specific_dim_samples(dim_idx_vertical)
        plt.scatter(x, y)
        if self.graphic_use_variable_name:
            plt.xlabel(self.variable_names[dim_idx_horizontal])
            plt.ylabel(self.variable_names[dim_idx_vertical])
        else:
            plt.xlabel(str(dim_idx_horizontal)+"th dim")
            plt.ylabel(str(dim_idx_vertical)+"th dim")
        
        if self.graphic_scatterplot_mean:
            plt.axvline(np.mean(x), color="red", linestyle="solid", linewidth=0.8)
            plt.axhline(np.mean(y), color="red", linestyle="solid", linewidth=0.8)
                
        if self.graphic_scatterplot_median:
            plt.axvline(np.median(x), color="red", linestyle="dashed", linewidth=0.8)
            plt.axhline(np.median(y), color="red", linestyle="dashed", linewidth=0.8)
        
        if show:
            plt.show()
    
    def show_boxplot(self, choose_dims=None, show=True):
        boxplot_data = []
        if choose_dims is None:
            choose_dims = [i for i in range(self.num_dim)]

        for dim_idx in choose_dims:
            boxplot_data.append(self.get_specific_dim_samples(dim_idx))
        boxplot_data = np.transpose(np.array(boxplot_data))

        label = []
        if self.graphic_use_variable_name:
            label = [self.variable_names[i] for i in choose_dims]
        else:
            label = choose_dims

        plt.boxplot(boxplot_data, labels=label)
        plt.xticks(rotation=45)

        if show:
            plt.show()

    def show_mean_CI_plot(self, choose_dims=None, show=True):
        plot_data = []
        if choose_dims is None:
            choose_dims = [i for i in range(self.num_dim)]

        for dim_idx in choose_dims:
            plot_data.append(self.get_specific_dim_samples(dim_idx))
        plot_data = np.array(plot_data)

        label = []
        if self.graphic_use_variable_name:
            label = [self.variable_names[i] for i in choose_dims]
        else:
            label = choose_dims

        mean_vec = self.get_sample_mean()
        mean_vec_choose_dims = [mean_vec[i] for i in choose_dims]
        for i, dim_idx in enumerate(choose_dims):
            quantile = self.get_sample_quantile([0.025, 0.25, 0.5, 0.75, 0.975])[dim_idx]
            x = i+1
            plt.plot([x], [mean_vec_choose_dims[i]], 'ro')
            plt.plot([x, x], [quantile[0], quantile[4]], color='black', linestyle='-', linewidth=2, zorder=0)
            plt.plot([x-0.1, x+0.1], [quantile[1], quantile[1]], color='black', linestyle='-', linewidth=2, zorder=0)
            plt.plot([x-0.1, x+0.1], [quantile[3], quantile[3]], color='black', linestyle='-', linewidth=2, zorder=0)
            plt.plot([x-0.2, x+0.2], [quantile[2], quantile[2]], color='black', linestyle='-', linewidth=2, zorder=0)

        plt.xticks([i+1 for i in range(len(choose_dims))], rotation=45, labels=label)
        if show:
            plt.show()


    def effective_sample_size(self, dim_idx, sum_lags=30):
        n = len(self.MC_sample)
        auto_corr = self.get_autocorr(dim_idx, sum_lags)
        ess = n / (1 + 2*sum(auto_corr))
        return ess



if __name__ == "__main__":
    pass