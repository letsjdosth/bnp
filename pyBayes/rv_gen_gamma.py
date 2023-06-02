from random import normalvariate, seed, gammavariate

import numpy as np
import scipy.linalg

class GammaBase:
    def __init__(self, set_seed=None):
        if set_seed is not None:
            seed(set_seed)

    def _parameter_support_checker(self, alpha_shape, beta_rate):
        if alpha_shape<=0:
            raise ValueError("alpha should be >0")
        if beta_rate<=0:
            raise ValueError("beta should be >0")

    def sampler(self, alpha_shape, beta_rate):
        pass

    def sampler_iter(self, sample_size: int, alpha_shape, beta_rate):
        samples = []
        for _ in range(sample_size):
            samples.append(self.sampler(alpha_shape, beta_rate))
        return samples

class Sampler_univariate_Gamma(GammaBase):
    def __init__(self, set_seed=None):
        super().__init__(set_seed)

    def sampler(self, alpha_shape, beta_rate):
        self._parameter_support_checker(alpha_shape, beta_rate)
        return gammavariate(alpha_shape, 1/beta_rate)
    
class Sampler_univariate_InvGamma(GammaBase):
    def __init__(self, set_seed=None):
        super().__init__(set_seed)
    
    def sampler(self, alpha_shape, beta_rate):
        self._parameter_support_checker(alpha_shape, beta_rate)
        scale = 1/beta_rate
        try:
            sample = 1/gammavariate(alpha_shape, scale)
        except ZeroDivisionError: #problematic, but...
            sample = self.sampler(alpha_shape, beta_rate)
        return sample

class Sampler_univariate_Chisq():
    # chisq(v) ~ gamma(shape=v/2, rate=1/2)
    def __init__(self, set_seed=None):
        self.gamma_sampler_inst = Sampler_univariate_Gamma(set_seed)

    def sampler_iter(self, sample_size: int, df):
        return self.gamma_sampler_inst.sampler_iter(sample_size, df/2, 0.5)

class Sampler_univariate_InvChisq():
    # inv.chisq(v) ~ inv.gamma(shape=v/2, rate=1/2)
    def __init__(self, set_seed=None):
        self.invgamma_sampler_inst = Sampler_univariate_InvGamma(set_seed)

    def sampler_iter(self, sample_size: int, df):
        return self.invgamma_sampler_inst.sampler_iter(sample_size, df/2, 0.5)

# ============================================================================

class Sampler_Wishart:
    def __init__(self, set_seed=None):
        if set_seed is not None:
            self.random_generator = np.random.default_rng(seed=set_seed)
        else:
            self.random_generator = np.random.default_rng()

    def _parameter_support_checker(self, df, V_scale, p_dim):
        # need cost
        if df <= (p_dim-1):
            raise ValueError("degrees of freedom should be > p-1")
        if not np.allclose(V_scale, V_scale.T, rtol=1e-02, atol=1e-05):
            print("V_scale: \n", V_scale)
            raise ValueError("V_scale should be symmetric")
        eigvals = np.linalg.eigvals(V_scale)
        if any([val<0 for val in eigvals]):
            raise ValueError("V_scale should be positive definite")

    def bartlett_base(self, df:int, V_Scale:np.array, p_dim) -> tuple[np.ndarray, np.ndarray]:
        chisq_sampler = Sampler_univariate_Chisq()
        bartlett_A_mat = []
        for i in range(p_dim):
            bartlett_A_ith_row = [normalvariate(0,1) for _ in range(i)]
            bartlett_A_ith_row.append(chisq_sampler.sampler_iter(1, df-i)[0]**0.5)
            bartlett_A_ith_row += [0 for _ in range(p_dim-i-1)]
            bartlett_A_mat.append(bartlett_A_ith_row)
        bartlett_A_mat = np.array(bartlett_A_mat)
        bartlett_L_mat = np.linalg.cholesky(V_Scale)
        return bartlett_A_mat, bartlett_L_mat

    def _sampler_bartlett(self, df:int, V_Scale: np.array, p_dim):
        bartlett_A_mat, bartlett_L_mat = self.bartlett_base(df, V_Scale, p_dim)
        wishart_sample = bartlett_L_mat @ bartlett_A_mat @ np.transpose(bartlett_A_mat) @ np.transpose(bartlett_L_mat)
        return wishart_sample

    def _sampler(self, df: int, V_scale: np.array, p_dim):
        # do not use it directly
        # (parameter support check is too costly, so I move it to the head of 'sampler_iter()' and run it once)
        mvn_samples = self.random_generator.multivariate_normal(np.zeros((p_dim,)), V_scale, size=df)
        wishart_sample = np.zeros(V_scale.shape)
        for mvn_sample in mvn_samples:
            wishart_sample += (np.outer(mvn_sample, mvn_sample))
        return wishart_sample

    def sampler_iter(self, sample_size: int, df: int, V_scale: np.array, mode="outer"):
        p_dim = V_scale.shape[0]
        self._parameter_support_checker(df, V_scale, p_dim)

        samples = []
        for _ in range(sample_size):
            if mode=="outer":
                samples.append(self._sampler(df, V_scale, p_dim))
            elif mode=="bartlett":
                samples.append(self._sampler_bartlett(df, V_scale, p_dim))
            else:
                raise ValueError("wishart sampler: mode should be either outer or bartlett")
        return samples

class Sampler_InvWishart:
    def __init__(self, set_seed=None):
        self.wishart_sampler = Sampler_Wishart(set_seed)

    def _sampler_bartlett(self, df:int, V_scale: np.array, p_dim):
        bartlett_A_mat, bartlett_L_mat = self.wishart_sampler.bartlett_base(df, V_scale, p_dim)
        #A: lower triangular
        #L: lower triangular
        A_inv = scipy.linalg.solve_triangular(bartlett_A_mat, np.eye(bartlett_A_mat.shape[1]), lower=True)
        L_inv = scipy.linalg.solve_triangular(bartlett_L_mat, np.eye(bartlett_L_mat.shape[1]), lower=True)
        inv_wishart_sample = np.transpose(L_inv) @ np.transpose(A_inv) @ A_inv @ L_inv
        return inv_wishart_sample

    def sampler_iter(self, sample_size: int, df:int, G_scale: np.array, mode="inv"):
        'X ~ Wishart(df,V) <=> 1/X ~inv.wishart(df,V^(-1)=G)'

        V_scale = np.linalg.inv(G_scale)
        if mode == "inv":
            wishart_samples = self.wishart_sampler.sampler_iter(sample_size, df, V_scale, mode="outer")
            samples = [np.linalg.inv(wishart_sample) for wishart_sample in wishart_samples]
        elif mode == "bartlett":
            p_dim = V_scale.shape[0]
            samples = [self._sampler_bartlett(df, V_scale, p_dim) for _ in range(sample_size)]
        else:
            raise ValueError("inverse wishart sampler: mode should be either bartlett or inv")
        return samples

class Sampler_multivariate_InvGamma:
    def __init__(self, set_seed=None):
        self.invwishart_sampler = Sampler_InvWishart(set_seed)
    
    def sampler_iter(self, sample_size: int, df:int, G_scale: np.array):
        # X~inv-gamma(w,G) <=> X~inv-wishart(2w, (1/w)G)
        return self.invwishart_sampler.sampler_iter(sample_size, 2*df, G_scale*(1/df))


if __name__=="__main__":
    chisq_inst = Sampler_univariate_Chisq(GammaBase)
    v = 2
    test_chisq_samples = chisq_inst.sampler_iter(10000, v)
    from statistics import mean, variance
    print(mean(test_chisq_samples), (v/2)/(1/2), "\n", variance(test_chisq_samples), (v/2)/(1/4))


    wishart_inst = Sampler_Wishart(set_seed=20220420)
    print(wishart_inst.sampler_iter(3, 5, np.array([[2,-1],[-1,3]])))

    inv_wishart_inst = Sampler_InvWishart(set_seed=20220420)
    inv_wishart_samples = inv_wishart_inst.sampler_iter(2, 5, np.array([[2,-1],[-1,3]]))
    print(inv_wishart_samples)
