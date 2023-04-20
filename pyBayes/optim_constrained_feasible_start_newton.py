import numpy as np
import scipy.linalg

class NewtonAffineConstrainedFeasibleStart:
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
        self.decrement_sequence = []
        self.value_sequence = []
    
    def _Rn_domain_indicator(self, eval_pt):
        return True
    
    def _backtracking_line_search(self, eval_pt, descent_direction, 
            a_slope_flatter_ratio, b_step_shorten_ratio):

        if a_slope_flatter_ratio <= 0 or a_slope_flatter_ratio >= 0.5:
            raise ValueError("a should be 0 < a < 0.5")
        if b_step_shorten_ratio <= 0 or b_step_shorten_ratio >= 1:
            raise ValueError("b should be 0 < a < 1")

        step_size = 1

        while True:
            flatten_line_slope = self.objective_gradient(eval_pt) * a_slope_flatter_ratio * step_size
            deviation_vec = descent_direction * step_size

            objective_fn_value = self.objective(eval_pt + deviation_vec)
            flatten_line_value = self.objective(eval_pt) + sum(flatten_line_slope * deviation_vec)

            if objective_fn_value < flatten_line_value and self.objective_domain_indicator(eval_pt + deviation_vec):
                break
            else:
                step_size = step_size * b_step_shorten_ratio

        return step_size
    
    def _check_feasible(self, eval_pt):
        eval_intercept = self.constraint_coeff_mat @ eval_pt # matmul
        if eval_intercept == self.constraint_intercept:
            is_feasible = True
        else:
            is_feasible = False
        return is_feasible

    def _get_KKT_matrix_and_intercept(self, eval_pt):
        num_row_constraint_coeff_mat = self.constraint_coeff_mat.shape[0]
        hessian = self.objective_hessian(eval_pt)
        negative_gradient = self.objective_gradient(eval_pt) * -1

        KKT_matrix = np.block([
            [hessian,                   self.constraint_coeff_mat.transpose()],
            [self.constraint_coeff_mat, np.zeros((num_row_constraint_coeff_mat, num_row_constraint_coeff_mat))]
            ])
        KKT_intercept = np.block([negative_gradient, np.zeros((1, num_row_constraint_coeff_mat))]).transpose()
        return (KKT_matrix, KKT_intercept)


    def _descent_derection_newton_feasible_cholesky(self, eval_pt):
        dim = eval_pt.size
        hessian = self.objective_hessian(eval_pt)
        gradient = self.objective_gradient(eval_pt)

        cholesky_lowtri_of_hessian = np.linalg.cholesky(hessian) # L ; H = L(L^T)
        cholesky_lowtri_of_hessian_inv = scipy.linalg.solve_triangular(cholesky_lowtri_of_hessian, np.identity(dim), lower=True)
        hessian_inv = cholesky_lowtri_of_hessian_inv.transpose() @ cholesky_lowtri_of_hessian_inv # H^(-1) = (L^(-1))^T L^(-1)
        
        #block elimination
        schur_complement = self.constraint_coeff_mat @ hessian_inv @ self.constraint_coeff_mat.transpose() * (-1) # S=-AH^(-1)A^T
        
        intercept1 = self.constraint_coeff_mat @ hessian_inv @ gradient
        dual_direction = scipy.linalg.solve(schur_complement, intercept1) #w; Sw = AH^(-1)g - h (now, h=0)

        intercept2 = self.constraint_coeff_mat.transpose() @ dual_direction * (-1) - gradient
        descent_direction = scipy.linalg.solve(hessian, intercept2) #x; Hx = -A^Tw - g

        newton_decrement_square = sum(descent_direction * gradient) * (-1)
        return descent_direction, newton_decrement_square
    
    
    def _descent_derection_newton_feasible_inv(self, eval_pt):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt) # K, k
        KKT_matrix_inv = np.linalg.inv(KKT_matrix)
        sol_KKT_system = np.matmul(KKT_matrix_inv, KKT_intercept) #inverse. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()

        newton_decrement_square = sum(KKT_intercept * sol_KKT_system) #[neg_grad, 0], [x, w]
        return descent_direction, newton_decrement_square

    def _descent_derection_newton_feasible_pinv(self, eval_pt):
        KKT_matrix, KKT_intercept = self._get_KKT_matrix_and_intercept(eval_pt) # K, k
        sol_KKT_system = np.matmul(np.linalg.pinv(KKT_matrix), KKT_intercept) #pseudo. K^(-1)k
        descent_direction = (sol_KKT_system[0:eval_pt.size]).flatten()

        newton_decrement_square = sum(KKT_intercept * sol_KKT_system) #[neg_grad, 0], [x, w]
        return descent_direction, newton_decrement_square

    def run_newton_with_feasible_starting_point(self, starting_pt, tolerance = 0.001, 
                                                method="cholesky", 
                                                a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        #method : cholesky, inv, pinv (if hessian is singular)
        is_feasible = self._check_feasible(starting_pt)
        if not is_feasible:
            raise ValueError("not feasible starting point.")

        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        self.decrement_sequence = []
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            if method == "cholesky":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_cholesky(eval_pt)
            elif method == "inv":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_inv(eval_pt)
            elif method == "pinv":
                descent_direction, decrement_square = self._descent_derection_newton_feasible_pinv(eval_pt)
            else:
                raise ValueError("method should be ['cholesky', 'inv', 'pinv']")
            decrement = np.sqrt(decrement_square)
            self.decrement_sequence.append(decrement)

            if decrement_square < (tolerance*2):
                break

            descent_step_size = self._backtracking_line_search(eval_pt, descent_direction, 
                                    a_slope_flatter_ratio, b_step_shorten_ratio)
            next_point = eval_pt + descent_direction * descent_step_size
            self.minimizing_sequence.append(next_point)
            self.value_sequence.append(self.objective(next_point))
            num_iter += 1

        print("iteration: ", num_iter)

    
    def get_minimizing_sequence(self):
        return self.minimizing_sequence
    
    def get_minimizing_function_value_sequence(self):
        return self.value_sequence

    def get_decrement_sequence(self):
        return self.decrement_sequence

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

    test_newton_feasible_inst1 = NewtonAffineConstrainedFeasibleStart(
        test_objective1,
        test_objective1_gradient,
        test_objective1_hessian,
        const_affine_coeff_mat = a1, 
        const_affine_intercept_vec = b1)
    # print(test_newton_feasible_inst1._check_feasible(np.array([2, 1])))
    # print(test_newton_feasible_inst1._get_KKT_matrix_and_intercept(np.array([2, 1])))

    test_newton_feasible_inst1.run_newton_with_feasible_starting_point(np.array([2, -1]), 0.00000001, method="cholesky")
    print(test_newton_feasible_inst1.get_minimizing_sequence())
    print(test_newton_feasible_inst1.get_decrement_sequence())
    print(test_newton_feasible_inst1.get_minimizing_function_value_sequence())
    