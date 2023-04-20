import cmath

import numpy as np
import matplotlib.pyplot as plt

class ARMA:
    def __init__(self, white_noise_var, ar_coeff: list[float]|None = None, ma_coeff: list[float]|None = None, seed=None) -> None:
        self.ar_coeff = []
        self.ma_coeff = []
        self._p = 0
        self._q = 0

        if ar_coeff is not None:
            self._p = len(ar_coeff)
            self.ar_coeff = ar_coeff
        if ma_coeff is not None:
            self._q = len(ma_coeff)
            self.ma_coeff = ma_coeff

        self._v = white_noise_var
        
        if seed:
            self.np_rand_inst = np.random.default_rng(seed)
        else:
            self.np_rand_inst = np.random.default_rng()

    def set_ar_coeff_from_reciprocal_roots(self, 
                                        reciprocal_roots: list[complex|float]|list[tuple],
                                        polar_radi_angle: bool = True):
        #use 0-pi for angle.
        if polar_radi_angle:
            roots = [(1/r)*cmath.exp(-1j*theta) for r,theta in reciprocal_roots]
        else:
            roots = [1/x for x in reciprocal_roots]
        ar_coeff = np.polynomial.polynomial.polyfromroots(roots)
        intercept = ar_coeff[0]
        ar_coeff = ([-x/intercept for x in ar_coeff])[1:]
        all_real = True
        for phi in ar_coeff:
            if phi.imag != 0:
                all_real = False
        if not all_real:
            raise ValueError("use conjugate root pairs")
        else:
            ar_coeff = [x.real for x in ar_coeff]
        self.__init__(self._v, ar_coeff)

    def _ar_polynomial(self, u):
        poly_val = 1
        for i, phi_i in enumerate(self.ar_coeff):
            poly_val -= (phi_i * u**(i+1))
        return poly_val
    
    def _ma_polynomial(self, u):
        poly_val = 1
        for i, theta_i in enumerate(self.ma_coeff):
            poly_val += (theta_i * u**(i+1))
        return poly_val

    def ar_polyonmial_root(self, reciprocal: bool = True):
        def sort_key1(c):
                return c[0]
        
        coeff = [1] + [-x for x in self.ar_coeff]
        ar_poly = np.polynomial.polynomial.Polynomial(coeff)
        ar_poly_roots = ar_poly.roots()
        ar_poly_polar_roots = [cmath.polar(x) for x in ar_poly_roots] # (radius,\angle)
        if reciprocal:
            ar_poly_polar_roots = [(1/radius,-angle) for radius,angle in ar_poly_polar_roots]
        ar_poly_polar_roots.sort(key=sort_key1, reverse=False)
        return ar_poly_polar_roots

    def spectral_density(self, grid_T: int, domain_0pi: bool = True, log_density = False):
        grid = np.linspace(0, cmath.pi, num=int(grid_T/2)+1, endpoint=True)

        spec_density_const = self._v/(2*cmath.pi)
        spec_density_on_grid = []
        for w in grid:
            big_phi = self._ar_polynomial(cmath.exp(-1j*w))
            big_theta = self._ma_polynomial(cmath.exp(-1j*w))
            f_at_w = spec_density_const * abs(big_theta)**2 / abs(big_phi)**2
            spec_density_on_grid.append(f_at_w)
        
        if domain_0pi==False:
            grid = np.linspace(0, 0.5, num=int(grid_T/2)+1, endpoint=True)
        if log_density:
            spec_density_on_grid = [np.log(x) for x in spec_density_on_grid]

        return spec_density_on_grid, grid
    
    def plot_spectral_density(self, grid_T: int, domain_0pi: bool = True, log_density = False, show=True):
        fw, w = self.spectral_density(grid_T, domain_0pi=domain_0pi, log_density=log_density)
        plt.plot(w, fw)
        if show:
            plt.show()

    def generate_random_path(self, length_T, np_array=True):
        max_p_q = max(self._p, self._q)
        innovation_path = [self.np_rand_inst.normal(0, self._v) for _ in range(max_p_q)]
        y_path = [0 for _ in range(max_p_q)]

        for _ in range(max_p_q, max_p_q+length_T):
            innovation_at_t = self.np_rand_inst.normal(0, self._v)
            y_at_t = innovation_at_t + 0
            y_at_t += sum([innovation_path[-(q+1)]*theta_q for q, theta_q in enumerate(self.ma_coeff)])
            y_at_t += sum([y_path[-(p+1)]*phi_p for p, phi_p in enumerate(self.ar_coeff)])
            innovation_path.append(innovation_at_t)
            y_path.append(y_at_t)

        y_path = y_path[max_p_q:]
        innovation_path = innovation_path[max_p_q:]

        if np_array:
            y_path = np.array(y_path)
            innovation_path = np.array(innovation_path)
        return y_path, innovation_path


if __name__=="__main__":
    
    arma_inst2 = ARMA(1, seed=20230214)
    arma_inst2.set_ar_coeff_from_reciprocal_roots([(0.8, 1), (0.8, -1)], polar_radi_angle=True)
    # arma_inst2.set_ar_coeff_from_reciprocal_roots([(0.7, 2.7), (0.7, -2.7)], polar_radi_angle=True)
    print("ar2 coeff", arma_inst2.ar_coeff)
    print("ar2 polyroot", arma_inst2.ar_polyonmial_root(reciprocal=True))
    # ar2 coeff [0.8644836893890236, -0.64]
    # ar2 polyroot [(0.8000000000000002, 1.0), (0.8000000000000002, -1.0)]



    spec2, grid2 = arma_inst2.spectral_density(512, domain_0pi=False)
    # arma_inst2.plot_spectral_density(512)
    x, _ = arma_inst2.generate_random_path(512)
    # import csv
    # with open('ar2_testdata2.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile, delimiter=',',
    #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #     writer.writerow(x)
            
    plt.plot(x)
    plt.title("ar2: one realization")
    plt.show()
    

    x_fft = np.fft.rfft(x)
    periodogram_x = [f*f.conjugate()/(512*2*np.pi) for f in x_fft]
    arma_inst2.plot_spectral_density(512, domain_0pi=False, show=False)
    plt.plot(grid2, periodogram_x)
    plt.title("ar2: spectra")
    plt.show()


    x_med = np.median(x)
    x_clipped = [1 if x0<=x_med else 0 for x0 in x]
    # plt.plot(x_clipped)
    # plt.show()
    
    x_clipped_fft = np.fft.rfft(x_clipped)
    plt.plot(x_clipped_fft.real[1:])
    plt.title("x_clipped_fft_real")
    plt.show()
    plt.plot(x_clipped_fft.imag[1:])
    plt.title("x_clipped_fft_imag")
    plt.show()

    periodogram_x_clipped= [f*f.conjugate()/(512*2*np.pi) for f in x_clipped_fft]
    periodogram_x_clipped[0]=0
    # arma_inst2.plot_spectral_density(512, domain_0pi=False, show=False)
    plt.plot(grid2, periodogram_x_clipped)
    # plt.plot(range(len(periodogram_x_clipped)), periodogram_x_clipped)
    plt.title("ar2 clipped sequence: spectra")
    plt.show()
    
    arma_inst2.plot_spectral_density(512, domain_0pi=False, log_density=True, show=False)
    plt.plot(grid2, np.log(periodogram_x_clipped))
    plt.title("ar2 clipped sequence: log spectra")
    plt.show()

    # list_periodogram_x_clipped =[]
    # for i in range(10000):
    #     if i%1000==0:
    #         print("iter:", i)
    #     x, _ = arma_inst2.generate_random_path(512)
    #     x_med = np.median(x)
    #     x_clipped = [1 if x0>x_med else 0 for x0 in x]
    #     x_clipped_fft = np.fft.rfft(x_clipped)
    #     periodogram_x_clipped= [abs(f)**2/(512*2*np.pi) for f in x_clipped_fft]
    #     periodogram_x_clipped[0]=1
    #     list_periodogram_x_clipped.append(periodogram_x_clipped)
    # array_periodogram_x_clipped = np.array(list_periodogram_x_clipped)
    # avg_periodogram_x_clipped = np.mean(array_periodogram_x_clipped, axis=0)
    # arma_inst2.plot_spectral_density(512, domain_0pi=False, log_density=True, show=False)
    # plt.plot(grid2, np.log(avg_periodogram_x_clipped))
    # plt.title("ar2 clipped sequence: averaged log spectra")
    # plt.show()
