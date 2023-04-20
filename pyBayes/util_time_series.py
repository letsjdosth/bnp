import cmath

import numpy as np

def difference_oper(sequence: list):
    before = sequence[0]
    iterator_for_seq = iter(sequence)
    next(iterator_for_seq)
    diff_sequence = []
    for now in iterator_for_seq:
        diff_sequence.append(now - before)
        before = now
    return diff_sequence

def ar_polynomial_roots(phi_samples: list[list[float]], reciprocal: bool) -> list[list[tuple[float, float]]]:
    # return [[(r, \theta), (r, \theta),...]] with increasing order w.r.t r
    def sort_key1(c):
        return c[0]
    
    ar_poly_polar_roots_at_samples = []
    for sample in phi_samples:
        coeff = [1] + [-x for x in sample]
        ar_poly = np.polynomial.polynomial.Polynomial(coeff)
        ar_poly_roots = ar_poly.roots()
        ar_poly_polar_roots = [cmath.polar(x) for x in ar_poly_roots] # r,\theta
        if reciprocal:
            ar_poly_polar_roots = [(1/r,-theta) for r,theta in ar_poly_polar_roots]
        ar_poly_polar_roots.sort(key=sort_key1, reverse=False)
        ar_poly_polar_roots_at_samples.append(ar_poly_polar_roots)
    print("# of AR roots: ", len(ar_poly_polar_roots_at_samples[0]))
    
    return ar_poly_polar_roots_at_samples

def autocorr(sequence, maxLag):
    acf = []
    y_mean = np.mean(sequence)
    y = [elem - y_mean  for elem in sequence]
    n_var = sum([elem**2 for elem in y])
    for k in range(maxLag+1):
        N = len(y)-k
        n_cov_term = 0
        for i in range(N):
            n_cov_term += y[i]*y[i+k]
        acf.append((n_cov_term / n_var)[0])
    return acf
        
if __name__=="__main__":
    test_seq = [1,2,3,4,5,7]
    print(difference_oper(test_seq))