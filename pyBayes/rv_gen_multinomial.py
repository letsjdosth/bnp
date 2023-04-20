from random import seed, random

class Sampler_multinomial:
    def __init__(self, set_seed=20220406):
        seed(set_seed)

    def _parameter_support_checker(self, p_param):
        eps = 0.000000000000001
        if sum(p_param) > 1.0+eps or sum(p_param) < 1.0-eps: 
            #floting number problem
            print("p: ", p_param)
            print("sum(p): ", sum(p_param))
            raise ValueError("p should have sum 1")
        for p_i in p_param:
            if p_i<0:
                raise ValueError("p_i should be >=0")
        
    def _idx_sampler_n1(self, p_param):
        self._parameter_support_checker(p_param)
        unif_rv = random()
        
        #determine class
        p_sum = 0
        for i, p_i in enumerate(p_param):
            p_sum += p_i
            if p_sum >= unif_rv:
                return i
        return len(p_param) #floting number problem

    def sampler(self, n_param, p_param):
        sample = [0 for _ in range(len(p_param))]
        for _ in range(n_param):
            i = self._idx_sampler_n1(p_param)
            sample[i] += 1
        return sample

if __name__ == "__main__":
    inst = Sampler_multinomial()
    print(inst.sampler(100, [0.1, 0.2, 0.3, 0.4]))
    print(inst.sampler(100, [0, 0.5, 0.3, 0.2]))
    print(inst.sampler(100, [0, 0.5, 0.3, 0.2]))
