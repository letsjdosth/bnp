import numpy as np
import scipy.linalg

class NewtonAffineConstrainedInfeasibleStart:
    def __init__(self, fn_objective, fn_objective_gradient, fn_objective_hessian, 
                        const_affine_coeff_mat, const_affine_intercept_vec,
                        fn_objective_domain_indicator = None):
   
    
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        self.objective_hessian = fn_objective_hessian
        
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator

        self.constraint_coeff_mat = const_affine_coeff_mat
        self.constraint_intercept = const_affine_intercept_vec

        self.minimizing_sequence = []
        self.dual_variable_sequence = []
        self.value_sequence = []
        self.residual_norm_sequence = []
    
    def _Rn_domain_indicator(self, eval_pt):
        return True
    
    def _get_residuals(self, eval_pt, eval_dual_pt):
        dual_residual = self.objective_gradient(eval_pt) + self.constraint_coeff_mat @ eval_dual_pt
        primal_residual = self.constraint_coeff_mat @ eval_pt - self.constraint_intercept
        return dual_residual, primal_residual

    def _get_residual_norm(self, eval_pt, eval_dual_pt):
        dual_residual, primal_residual = self._get_residuals(eval_pt, eval_dual_pt)
        residual_norm = (np.sum(dual_residual**2) + np.sum(primal_residual**2))**0.5
        return residual_norm

    def _backtracking_line_search_with_residuals(self, eval_pt, eval_dual_pt, 
            newton_primal_direction, dual_direction, 
            a_slope_flatter_ratio, b_step_shorten_ratio):

        if a_slope_flatter_ratio <= 0 or a_slope_flatter_ratio >= 0.5:
            raise ValueError("a should be 0 < a < 0.5")
        if b_step_shorten_ratio <= 0 or b_step_shorten_ratio >= 1:
            raise ValueError("b should be 0 < a < 1")

        step_size = 1

        while True:
            primal_deviated_pt = eval_pt + step_size * newton_primal_direction
            dual_deviated_pt = eval_dual_pt + step_size * dual_direction

            residual_norm_deviated = self._get_residual_norm(primal_deviated_pt, dual_deviated_pt)
            residual_norm_eval = self._get_residual_norm(eval_pt, eval_dual_pt)

            if residual_norm_deviated <= (1 - a_slope_flatter_ratio * step_size) * residual_norm_eval:
                break
            else:
                step_size = step_size * b_step_shorten_ratio

        return step_size
    
    def _get_KKT_matrix_and_intercept(self, eval_pt):
        num_row_constraint_coeff_mat = self.constraint_coeff_mat.shape[0]
        hessian = self.objective_hessian(eval_pt)
        gradient = self.objective_gradient(eval_pt)

        KKT_matrix = np.block([
            [hessian,                   self.constraint_coeff_mat.transpose()],
            [self.constraint_coeff_mat, np.zeros((num_row_constraint_coeff_mat, num_row_constraint_coeff_mat))]
            ])
        KKT_intercept = np.block([gradient, self.constraint_coeff_mat @ eval_pt - self.constraint_intercept]).transpose() * (-1)
        return KKT_matrix, KKT_intercept


    def _descent_derection_newton_infeasible_cholesky(self, eval_pt):
        dim = eval_pt.size
        hessian = self.objective_hessian(eval_pt)
        gradient = self.objective_gradient(eval_pt)

        cholesky_lowtri_of_hessian = np.linalg.cholesky(hessian) # L ; H = L(L^T)
        cholesky_lowtri_of_hessian_inv = scipy.linalg.solve_triangular(cholesky_lowtri_of_hessian, np.identity(dim), lower=True)
        hessian_inv = cholesky_lowtri_of_hessian_inv.transpose() @ cholesky_lowtri_of_hessian_inv # H^(-1) = (L^(-1))^T L^(-1)
        
        #block elimination
        schur_complement = self.constraint_coeff_mat @ hessian_inv @ self.constraint_coeff_mat.transpose() * (-1) # S=-AH^(-1)A^T
        
        intercept1 = self.constraint_coeff_mat @ hessian_inv @ gradient - (self.constraint_coeff_mat @ eval_pt - self.constraint_intercept)
        dual_direction = scipy.linalg.solve(schur_complement, intercept1) #w; Sw = AH^(-1)g - h (now, h=Ax-b)

        intercept2 = self.constraint_coeff_mat.transpose() @ dual_direction * (-1) - gradient
        descent_direction = scipy.linalg.solve(hessian, intercept2) #x; Hx = -A^Tw - g

        return descent_direction, dual_direction
    
    
    def _descent_derection_newton_infeasible_inv(self, eval_pt):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt) # K, k
        KKT_matrix_inv = np.linalg.inv(KKT_matrix)
        sol_KKT_system = np.matmul(KKT_matrix_inv, KKT_intercept) #inverse. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()
        dual_direction = (sol_KKT_system[eval_pt.size:]).flatten()
        return descent_direction, dual_direction

    def _descent_derection_newton_infeasible_pinv(self, eval_pt):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt) # K, k
        sol_KKT_system = np.matmul(np.linalg.pinv(KKT_matrix), KKT_intercept) #pseudo. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()
        dual_direction = (sol_KKT_system[eval_pt.size:]).flatten()
        return descent_direction, dual_direction

    def _l2_norm(self, vec):
        return (np.sum(vec**2))*0.5

    def _check_feasible(self, eval_pt, ):
        eval_intercept = self.constraint_coeff_mat @ eval_pt # matmul
        if self._l2_norm(eval_intercept - self.constraint_intercept) < 0.0000001:
            is_feasible = True
        else:
            is_feasible = False
        return is_feasible

    def run_newton_with_infeasible_starting_point(self, starting_pt, dual_stating_pt, 
                                                tolerance = 0.001, 
                                                method="cholesky", 
                                                a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):

        self.minimizing_sequence = [starting_pt]
        self.dual_variable_sequence = [dual_stating_pt]
        self.value_sequence = [self.objective(starting_pt)]
        self.residual_norm_sequence = []
        num_iter = 0
        
        while True:
            eval_pt = self.minimizing_sequence[-1]
            eval_dual_pt = self.dual_variable_sequence[-1]

            #stopping criteria
            eval_points_residual_norm = self._get_residual_norm(eval_pt, eval_dual_pt)
            self.residual_norm_sequence.append(eval_points_residual_norm)
            if eval_points_residual_norm < tolerance and self._check_feasible(eval_pt):
                break

            if method == "cholesky":
                newton_primal_direction, dual_direction = self._descent_derection_newton_infeasible_cholesky(eval_pt)
            elif method == "inv":
                newton_primal_direction, dual_direction = self._descent_derection_newton_infeasible_inv(eval_pt)
            elif method == "pinv":
                newton_primal_direction, dual_direction = self._descent_derection_newton_infeasible_pinv(eval_pt)
            else:
                raise ValueError("method should be ['cholesky', 'inv', 'pinv']")
            
            descent_step_size = self._backtracking_line_search_with_residuals(
                                    eval_pt, eval_dual_pt,
                                    newton_primal_direction, dual_direction,
                                    a_slope_flatter_ratio, b_step_shorten_ratio)

            next_primal_pt = eval_pt + newton_primal_direction * descent_step_size
            next_dual_pt = eval_dual_pt + dual_direction * descent_step_size

            self.minimizing_sequence.append(next_primal_pt)
            self.dual_variable_sequence.append(next_dual_pt)
            self.value_sequence.append(self.objective(next_primal_pt))
            num_iter += 1
            

        print("iteration: ", num_iter)

    
    def get_minimizing_sequence(self):
        return self.minimizing_sequence
    
    def get_minimizing_function_value_sequence(self):
        return self.value_sequence

    def get_residual_norm_sequence(self):
        return self.residual_norm_sequence

    def get_arg_min(self):
        return self.minimizing_sequence[-1]

    def get_min(self):
        return self.objective(self.minimizing_sequence[-1])



if __name__ == "__main__":
    #test 1
    def test_objective1(vec_2dim, gamma = 2):
        val = 0.5 * (vec_2dim[0]**2 + gamma * (vec_2dim[1]**2))
        return np.array(val)

    def test_objective1_gradient(vec_2dim, gamma = 2):
        grad = (vec_2dim[0], vec_2dim[1] * gamma)
        return np.array(grad)

    def test_objective1_hessian(vec_2dim, gamma = 2):
        hessian = [[1, 0],
                   [0, gamma]]
        return np.array(hessian)

    a1 = np.array([[1, 1]])
    b1 = np.array([1])

    test_newton_feasible_inst1 = NewtonAffineConstrainedInfeasibleStart(
        test_objective1,
        test_objective1_gradient,
        test_objective1_hessian,
        const_affine_coeff_mat = a1, 
        const_affine_intercept_vec = b1)

    test_newton_feasible_inst1.run_newton_with_infeasible_starting_point(
        np.array([5, -1]), dual_stating_pt=np.array([1, 1]), 
        tolerance=0.0000001, method="cholesky")
    print(test_newton_feasible_inst1.get_minimizing_sequence())
    print(test_newton_feasible_inst1.get_residual_norm_sequence())
    print(test_newton_feasible_inst1.get_minimizing_function_value_sequence())
    