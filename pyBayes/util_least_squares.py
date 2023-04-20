import numpy as np
import scipy.linalg

def OLS_by_QR(design_X, resp_Y):
    design_X = np.array(design_X)

    beta_dim = design_X.shape[1]
    #design_X = QR, Q:orthogonal(n*n), R:upper-triangular(n*p)
    Q, R = np.linalg.qr(design_X, mode='complete')
    R1 = R[0:beta_dim, 0:beta_dim] #upper-triangular(p*p)
    QtY = np.transpose(Q)@np.array(resp_Y)

    beta_fit = scipy.linalg.solve_triangular(R1, QtY[0:beta_dim], lower=False)
    sum_of_squared_error = sum(QtY[beta_dim:]**2)
    return beta_fit, sum_of_squared_error

def GLS_by_cholesky(design_X, resp_Y, cov_mat_Sigma, return_F=False):
    # cov_mat_Sigma = LLt
    L = np.linalg.cholesky(cov_mat_Sigma)
    log_det_cov_mat_Sigma = np.sum(np.log(np.diag(L)))*2
    
    Z = scipy.linalg.solve_triangular(L, resp_Y, lower=True) #Z = L^(-1)Y
    F = scipy.linalg.solve_triangular(L, design_X, lower=True) #F = L^(-1)X
    #now, Z = F*beta + error(with cov=I)
    
    beta_fit, sum_of_squared_error = OLS_by_QR(F, Z)
    if return_F:
        return beta_fit, sum_of_squared_error, log_det_cov_mat_Sigma, F
    else:
        return beta_fit, sum_of_squared_error, log_det_cov_mat_Sigma

def sym_defpos_matrix_inversion_cholesky(symmetric_pos_def_matrix) -> tuple[np.ndarray, float]:
    #caution: this function does not check the symmetry.

    L = np.linalg.cholesky(symmetric_pos_def_matrix) # S = LL' => S_inv = inv(LL') = inv(L')@inv(L) = inv(L)'@inv(L)
    log_det = np.sum(np.log(np.diag(L)))*2
    # print(L.shape)
    L_inv = scipy.linalg.solve_triangular(L, np.eye(L.shape[1]), lower=True) #L@L_inv=I
    # print(L_inv,"\n", np.linalg.inv(L))
    inv = np.transpose(L_inv) @ L_inv
    # print(inv@symmetric_pos_def_matri@x)
    return inv, log_det



if __name__=="__main__":
    # data_soc = []
    # data_long = []
    # data_lat = []
    
    # import csv
    # with open('data/soil_carbon_VA.csv', newline='') as csvfile:
    #     csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    #     next(csv_reader)
    #     for row in csv_reader:
    #         data_soc.append(float(row[2]))
    #         data_long.append(float(row[9]))
    #         data_lat.append(float(row[10]))


    # design_1st_X = np.array([[1, x, y] for x,y in zip(data_long, data_lat)])
    # design_2nd_X = np.array([[1, x, y, x**2, y**2, x*y] for x,y in zip(data_long, data_lat)])
    # resp_Y = np.array(data_soc)
    # beta, sse = OLS_by_QR(design_1st_X, resp_Y)
    # print(beta, sse)

    # from cov_functions import Matern
    # from random import normalvariate
    # matern_scale1_inst = Matern(1, 298, 0.169)
    # data_points = [[lon+normalvariate(0, 0.1), lat+normalvariate(0, 0.1)] for lon, lat in zip(data_long, data_lat)]
    # matern_cov = matern_scale1_inst.cov_matrix(data_points)
    # beta_gls, sse_gls, log_det_cov = GLS_by_cholesky(design_2nd_X, resp_Y, matern_cov)
    # print(beta_gls, sse_gls, log_det_cov)
    # print(np.linalg.slogdet(matern_cov))

    # test_mat = np.array([[3,1,1],[2,2,2],[0,0,1]])
    test_mat = np.array([[2,1,-1],[1,1,0],[-1,0,2]])
    print(np.linalg.inv(test_mat), np.log(np.linalg.det(test_mat)))
    print(sym_defpos_matrix_inversion_cholesky(test_mat))
