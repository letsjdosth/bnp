from random import gammavariate, seed

class Sampler_Dirichlet:
    def __init__(self, set_seed):
        seed(set_seed)

    def _parameter_support_checker(self, alpha_param):
        for a in alpha_param:
            if a<=0:
                raise ValueError("all elements of alpha should be >0")

    # def sampler(self, alpha_param: list[float]) -> list[float]: #for python 3.9 or later
    def sampler(self, alpha_param: list):
        self._parameter_support_checker(alpha_param)
        beta = 1 #any value
        gamma_samples = [gammavariate(alpha, beta) for alpha in alpha_param]
        sum_gamma_samples = sum(gamma_samples)
        dir_sample = [smpl/sum_gamma_samples for smpl in gamma_samples]
        return dir_sample
    
    def sampler_iter(self, sample_size: int, alpha_param: list):
        samples = []
        for _ in range(sample_size):
            samples.append(self.sampler(alpha_param))
        return samples

if __name__ == "__main__":
    dir_sampler_inst = Sampler_Dirichlet(20220406)
    print(dir_sampler_inst.sampler_iter(5, [1,2,3,4,5]))
    print(dir_sampler_inst.sampler_iter(5, [1,1,1,1,1]))
    