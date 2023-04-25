from random import seed, betavariate

from pyBayes.rv_gen_dirichlet import Sampler_Dirichlet


class DirichletProcessGenerator:
    def __init__(self, set_seed) -> None:
        seed(set_seed)
        self.dir_generator = Sampler_Dirichlet(set_seed)
    
    def sampler(self, precision:float, center_dist_func, center_dist_sampler, num_grid_pt, grid_starting_pt, grid_ending_pt):
        # grid_starting_pt and grid_ending_pt are truncation points.
        
        grid = [center_dist_sampler() for _ in range(num_grid_pt)]
        grid.sort()
        if grid_starting_pt < grid[0]:
            grid = [grid_starting_pt] + grid
        else:
            raise ValueError("set better grid starting pt")
        dir_param = [center_dist_func(grid[0])]
        for i in range(1, num_grid_pt):
            dir_param.append((center_dist_func(grid[i]) - center_dist_func(grid[i-1])) * precision)
        dir_param.append((1 - center_dist_func(grid[-1])) * precision)

        increments = self.dir_generator.sampler(dir_param)
        cum_dist_val = 0
        sample_path = []
        for inc in increments: #or use np.accumulate
            cum_dist_val += inc
            sample_path.append(cum_dist_val)

        increments = increments + [0]
        if grid[-1] < grid_ending_pt:
            grid.append(grid_ending_pt)
            sample_path.append(1)
        else:
            raise ValueError("set better grid ending pt")
        
        return grid, increments, sample_path

class DirichletProcessGenerator_StickBreaking:
    def __init__(self, set_seed) -> None:
        seed(set_seed)
        self.atom_loc = None
        self.atom_weight = None

    def atom_sampler(self, precision:float, center_dist_sampler, num_atom: int):
        atom_loc = [center_dist_sampler() for _ in range(num_atom)]
        left_stick_length = 1.0
        atom_weight = []
        for _ in range(num_atom-1):
            portion = betavariate(1, precision)
            weight = portion * left_stick_length
            atom_weight.append(weight)
            left_stick_length = left_stick_length * (1 - portion)
        atom_weight.append(left_stick_length)
        return atom_loc, atom_weight

    def cumulatative_dist_func(self, atom_loc: list, atom_weight: list, trunc_lower: float, trunc_upper: float):
        "Warning: to use `plt.bar`, set `where='post'`"
        #sort
        tuple_list = []
        for loc, weight in zip(atom_loc, atom_weight):
            tuple_list.append((loc, weight))
        
        def sort_key(tup):
            return tup[0]
        tuple_list.sort(key=sort_key)
        
        sample_path = [0]
        increments = [0]
        grid = [trunc_lower]
        cum_dist_val = 0
        for tup in tuple_list:
            cum_dist_val += tup[1]
            grid.append(tup[0])
            increments.append(tup[1])
            sample_path.append(cum_dist_val)
        increments.append(0)
        sample_path.append(1)
        grid = grid + [trunc_upper]

        return grid, increments, sample_path
    
    def mean_var_functional(self, atom_loc:list, atom_weight:list):
        # n = len(atom_loc)
        mean_X = 0
        mean_X2 = 0
        for loc, w in zip(atom_loc, atom_weight):
            mean_X += (loc*w)
            mean_X2 += ((loc)**2 * w)
        var_X = mean_X2 - mean_X**2
        return mean_X, var_X, mean_X2



if __name__=="__main__":
    from random import gammavariate
    from functools import partial
    import numpy as np
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    
    std_norm_cdf = partial(norm.cdf, loc=0, scale=1)
    std_norm_sampler = partial(norm.rvs, loc=0, scale=1)

    np.random.seed(20230419)    
    
    def1_dir_inc = True #4a
    def2_stick_breaking = True #4a
    moment_test = True #4b
    mdp_test = True #4c


    if def1_dir_inc:
        inst = DirichletProcessGenerator(20230419)

        grid, increments, path = inst.sampler(1, std_norm_cdf, std_norm_sampler, 2, -5, 5)
        print(grid)
        print([round(x,3) for x in path])
        print([round(x,3) for x in increments])
        plt.step(grid, path, where='post')
        plt.scatter(grid, [0.01 for _ in range(len(grid))], c='red')
        plt.bar(grid, increments, 0.1, color='orange')
        plt.show()
        
        grid, _, path = inst.sampler(0.1, std_norm_cdf, std_norm_sampler, 500, -5, 5)
        plt.step(grid, path, where='post', c='blue', label=r'$\alpha$=0.1')
        for i in range(4):
            grid, _, path = inst.sampler(0.1, std_norm_cdf, std_norm_sampler, 500, -5, 5)
            plt.step(grid, path, where='post', c='blue')

        grid, _, path = inst.sampler(1, std_norm_cdf, std_norm_sampler, 500, -5, 5)
        plt.step(grid, path, where='post', c='orange', label=r'$\alpha$=1')
        for i in range(4):
            grid, _, path = inst.sampler(1, std_norm_cdf, std_norm_sampler, 500, -5, 5)
            plt.step(grid, path, where='post', c='orange')

        grid, _, path = inst.sampler(10, std_norm_cdf, std_norm_sampler, 500, -5, 5)
        plt.step(grid, path, where='post', c='green', label=r'$\alpha$=10')
        for i in range(4):
            grid, _, path = inst.sampler(10, std_norm_cdf, std_norm_sampler, 500, -5, 5)
            plt.step(grid, path, where='post', c='green')

        grid, _, path = inst.sampler(100, std_norm_cdf, std_norm_sampler, 500, -5, 5)
        plt.step(grid, path, where='post', c='purple', label=r'$\alpha$=100')
        for i in range(4):
            grid, _, path = inst.sampler(100, std_norm_cdf, std_norm_sampler, 500, -5, 5)
            plt.step(grid, path, where='post', c='purple')

        plt.plot(np.linspace(-3,3,200), std_norm_cdf(np.linspace(-3,3,200)), c='red', label=r'$G_0$=N(0,1)')
        plt.legend()
        plt.show()

    
    if def2_stick_breaking:
        inst = DirichletProcessGenerator_StickBreaking(20230420)
        atom_loc, atom_weight = inst.atom_sampler(1, std_norm_sampler, 10)
        print([round(x,4) for x in atom_loc])
        print([round(x,4) for x in atom_weight])
        print(sum(atom_weight))
        plt.bar(atom_loc, atom_weight, 0.02)
        plt.show()

        grid, increments, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        print([round(x, 3) for x in grid])
        print([round(x, 3) for x in increments])
        print([round(x, 3) for x in path])
        plt.step(grid, path, where='post')
        plt.scatter(grid, [0.01 for _ in range(len(grid))], c='red')
        plt.bar(grid, increments, 0.1, color='orange')
        plt.show()

        atom_loc, atom_weight = inst.atom_sampler(0.1, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='blue', label=r'$\alpha$=0.1')
        for i in range(4):
            atom_loc, atom_weight = inst.atom_sampler(0.1, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='blue')
        atom_loc, atom_weight = inst.atom_sampler(1, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='orange', label=r'$\alpha$=1')
        for i in range(4):
            atom_loc, atom_weight = inst.atom_sampler(1, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='orange')
        atom_loc, atom_weight = inst.atom_sampler(10, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='green', label=r'$\alpha$=10')
        for i in range(4):
            atom_loc, atom_weight = inst.atom_sampler(10, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='green')
        atom_loc, atom_weight = inst.atom_sampler(100, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='purple', label=r'$\alpha$=100')
        for i in range(4):
            atom_loc, atom_weight = inst.atom_sampler(100, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='purple')
        plt.plot(np.linspace(-3,3,200), std_norm_cdf(np.linspace(-3,3,200)), c='red', label=r'$G_0$=N(0,1)')
        plt.legend()
        plt.show()


    if moment_test:
        inst = DirichletProcessGenerator_StickBreaking(20230419)
        mean_vec = []
        var_vec = []
        X2_vec = []
        alpha = 10
        for i in range(1000):
            atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 1000)
            mean_val, var_val, X2_val = inst.mean_var_functional(atom_loc, atom_weight)
            mean_vec.append(mean_val)
            var_vec.append(var_val)
            X2_vec.append(X2_val)
        
        print("E[mu(G)]:", "sim:", np.mean(mean_vec), " true:", 0)
        print("Var[mu(G)]:", "sim:", np.var(mean_vec), " true:", 1/(alpha+1))
        print("E[integral x^2 dG]:", "sim:", np.mean(X2_vec), " true:", 1)
        print("E[sigma^2(G)]:", "sim:", np.mean(var_vec), " true:", alpha/(alpha+1))

        plt.hist(mean_vec, bins=100)
        plt.title("mean_functional_of_G")
        plt.show()
        plt.hist(var_vec, bins=100)
        plt.title("variance_functional_of_G")
        plt.show()


    if mdp_test:
        alpha_prior_shape = 1
        alpha_prior_rate = 1 #mean 1, var 1
        alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
        atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='blue', label=r'$\alpha\sim$gamma(1,1)')
        for i in range(4):
            alpha_prior_shape = 1
            alpha_prior_rate = 1 #mean 1, var 1
            alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
            atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='blue')

        alpha_prior_shape = 10
        alpha_prior_rate = 10 #mean 1, var 1/10
        alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
        atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='orange', label=r'$\alpha\sim$gamma(10,10)')
        for i in range(4):
            alpha_prior_shape = 10
            alpha_prior_rate = 10 #mean 1, var 1/10
            alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
            atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='orange')


        alpha_prior_shape = 100
        alpha_prior_rate = 10 #mean 10, var 1
        alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
        atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='purple', label=r'$\alpha\sim$gamma(100,10)')
        for i in range(4):
            alpha_prior_shape = 100
            alpha_prior_rate = 10 #mean 10, var 1
            alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
            atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='purple')
        plt.plot(np.linspace(-3,3,200), std_norm_cdf(np.linspace(-3,3,200)), c='red', label=r'$G_0$=N(0,1)')

        alpha_prior_shape = 10
        alpha_prior_rate = 1 #mean 10, var 10
        alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
        atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
        grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
        plt.step(grid, path, where='post', c='green', label=r'$\alpha\sim$gamma(10,1)')
        for i in range(4):
            alpha_prior_shape = 10
            alpha_prior_rate = 1 #mean 10, var 10
            alpha = gammavariate(alpha_prior_shape, 1/alpha_prior_rate)
            atom_loc, atom_weight = inst.atom_sampler(alpha, std_norm_sampler, 500)
            grid, _, path = inst.cumulatative_dist_func(atom_loc, atom_weight, -5, 5)
            plt.step(grid, path, where='post', c='green')

        plt.legend()
        plt.show()
