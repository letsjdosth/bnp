from random import seed, gammavariate
from math import exp

class GammaProcess_from_NeutralToTheRightProcess:
    "gamma process generator using NTTR"
    #no truncation: generated cdf may not be 1 at the end point of the grid
    def __init__(self, set_seed) -> None:
        seed(set_seed)
    
    def sampler(self, precision:float, baseline_cum_hazard_func, grid:list):
        baseline_cumhazard_increment = []
        
        # generate Z, cumulative hazard path
        generated_hazard_list = []
        
        increment_0 = baseline_cum_hazard_func(grid[0]) - baseline_cum_hazard_func(0)
        baseline_cumhazard_increment.append(increment_0)
        generated_hazard_0 = gammavariate(precision*increment_0, 1/precision)
        generated_hazard_list.append(generated_hazard_0)
        
        cum_hazard_sum = generated_hazard_0
        generated_cum_hazard = [generated_hazard_0]
        for i in range(1, len(grid)):
            increment_i = baseline_cum_hazard_func(grid[i]) - baseline_cum_hazard_func(grid[i-1])
            baseline_cumhazard_increment.append(increment_i)
            generated_hazard = gammavariate(precision*increment_i, 1/precision)
            generated_hazard_list.append(generated_hazard)
            cum_hazard_sum += generated_hazard
            generated_cum_hazard.append(cum_hazard_sum)
    
        # construct F, distribution function path
        generated_dist = [0]
        for r in generated_hazard_list:
            F_t_1 = generated_dist[-1]
            Ft = F_t_1 + (1 - exp(-r))*(1 - F_t_1)
            generated_dist.append(Ft)
        
        return [0]+grid, [0]+generated_cum_hazard, generated_dist

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    def exp1_baseline_cumulative_hazard(t):
        return t
    inst = GammaProcess_from_NeutralToTheRightProcess(20230421)
    input_grid = [gammavariate(1, 1) for _ in range(3000)]
    input_grid.sort()

    fig, ax = plt.subplots(1, 2)
    for x in range(10):
        grid, cum_hazard, cdf = inst.sampler(0.1, exp1_baseline_cumulative_hazard, input_grid)
        ax[0].step(grid, cum_hazard, where="post", c='blue')
        ax[1].step(grid, cdf, where="post", c='blue')
    ax[0].plot(grid, grid, c='red')
    ax[1].plot(grid, [1-exp(-r) for r in grid], c='red')
    plt.show()

    fig, ax = plt.subplots(1, 2)
    for x in range(10):
        grid, cum_hazard, cdf = inst.sampler(1, exp1_baseline_cumulative_hazard, input_grid)
        ax[0].step(grid, cum_hazard, where="post", c='blue')
        ax[1].step(grid, cdf, where="post", c='blue')
    ax[0].plot(grid, grid, c='red')
    ax[1].plot(grid, [1-exp(-r) for r in grid], c='red')
    plt.show()

    fig, ax = plt.subplots(1, 2)
    for x in range(10):
        grid, cum_hazard, cdf = inst.sampler(10, exp1_baseline_cumulative_hazard, input_grid)
        ax[0].step(grid, cum_hazard, where="post", c='blue')
        ax[1].step(grid, cdf, where="post", c='blue')
    ax[0].plot(grid, grid, c='red')
    ax[1].plot(grid, [1-exp(-r) for r in grid], c='red')
    plt.show()

    fig, ax = plt.subplots(1, 2)
    for x in range(10):
        grid, cum_hazard, cdf = inst.sampler(100, exp1_baseline_cumulative_hazard, input_grid)
        ax[0].step(grid, cum_hazard, where="post", c='blue')
        ax[1].step(grid, cdf, where="post", c='blue')
    ax[0].plot(grid, grid, c='red')
    ax[1].plot(grid, [1-exp(-r) for r in grid], c='red')
    plt.show()
