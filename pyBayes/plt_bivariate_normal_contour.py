import numpy as np
import matplotlib.pyplot as plt

def gen_contour_level_matrix_for_2d_Gaussian(meshgrid_x, meshgrid_y, mean_2dvec, cov_2dmat):
    dim = 2
    packed_grid = np.empty(meshgrid_x.shape + (2,))
    packed_grid[:, :, 0] = meshgrid_x
    packed_grid[:, :, 1] = meshgrid_y
    
    inv_cov = np.linalg.inv(cov_2dmat)

    cov_determinant = np.linalg.det(cov_2dmat)
    normal_const = 1/np.sqrt((2*np.pi)**dim * cov_determinant)
    
    log_kernel = np.einsum("ijk,kl,ijl->ij", packed_grid-mean_2dvec, inv_cov, packed_grid-mean_2dvec) * (-0.5) #2-d
    # log_kernel = np.einsum('...k,kl,...l->...', packed_grid-mean_2dvec, inv_cov, packed_grid-mean_2dvec) * (-0.5) #n-d
    
    # ## einsum ##
    # English help : https://rockt.github.io/2018/04/30/einsum
    # Korean help : https://baekyeongmin.github.io/dev/einsum/
    # ## for gaussian case ##
    # https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    #or we may just run 'for' statement with each grid values

    return np.exp(log_kernel)/normal_const

grid_x = np.linspace(-1, 1, 10)
grid_y = np.linspace(-1, 1, 10)
meshgrid_x, meshgrid_y = np.meshgrid(grid_x, grid_y)

mean = np.array([0,0])
cov = np.array([[1, 0.5], 
                [0.5, 1]])
mvn2d_value_mat = gen_contour_level_matrix_for_2d_Gaussian(meshgrid_x, meshgrid_y, mean, cov)

plt.contour(meshgrid_x, meshgrid_y, mvn2d_value_mat, levels=30)
plt.show()