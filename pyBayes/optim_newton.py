import numpy as np
import scipy.linalg


class NewtonUnconstrained:
    def __init__(self, fn_objective, fn_objective_gradient, fn_objective_hessian, fn_objective_domain_indicator = None):
        
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        self.objective_hessian = fn_objective_hessian
        
        
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator
        
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
    
    def _l2_norm(self, vec):
        return (sum(vec**2))**0.5
    
    def _descent_direction_newton_cholesky(self, eval_pt):
        hessian = self.objective_hessian(eval_pt)
        neg_gradient = self.objective_gradient(eval_pt) * (-1)
        cholesky_lowertri_of_hessian = scipy.linalg.cholesky(hessian, lower=True) # L ; H = L(L^T)
        #want: x; Hx = -g
        forward_variable = scipy.linalg.solve_triangular(cholesky_lowertri_of_hessian, neg_gradient, lower=True) # w ; Lw = -g
        newton_decrement = self._l2_norm(forward_variable)
        direction_newton_step = scipy.linalg.solve_triangular(np.transpose(cholesky_lowertri_of_hessian), forward_variable, lower=False) # x ; (L^t)x = w
        return (direction_newton_step, newton_decrement)

    # later: sparse / band hessian version

    def _descent_direction_newton_pinv(self, eval_pt):
        hessian = self.objective_hessian(eval_pt)
        neg_gradient = self.objective_gradient(eval_pt) * (-1)
        direction_newton_step = np.matmul(np.linalg.pinv(hessian), neg_gradient) #pseudo
        newton_decrement = np.matmul(np.transpose(neg_gradient), direction_newton_step)
        return (direction_newton_step, newton_decrement)

    def run_newton_with_backtracking_line_search(self, starting_pt, tolerance = 0.001, 
                                                method="cholesky", 
                                                a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        #method : cholesky, pinv (if hessian is singular)
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        self.decrement_sequence = []
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            if method == "cholesky":
                descent_direction, decrement = self._descent_direction_newton_cholesky(eval_pt)
            elif method == "pinv":
                descent_direction, decrement = self._descent_direction_newton_pinv(eval_pt)
            else:
                raise ValueError("method should be ['cholesky', 'pinv']")
            self.decrement_sequence.append(decrement)

            if (decrement**2) < (tolerance*2):
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

    test_newton_inst1 = NewtonUnconstrained(
        test_objective1, 
        test_objective1_gradient, 
        test_objective1_hessian)
    test_newton_inst1.run_newton_with_backtracking_line_search(np.array([1000, 150]), method="cholesky")
    print(test_newton_inst1.get_minimizing_sequence())
    print(test_newton_inst1.get_decrement_sequence())
    print(test_newton_inst1.get_minimizing_function_value_sequence())
    


    #test 2
    def test_objective2(vec_2dim):
        val = np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) + np.exp(-vec_2dim[0] - 0.1)
        return np.array(val)

    def test_objective2_gradient(vec_2dim):
        grad = (np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) - np.exp(-vec_2dim[0] - 0.1),
                3 * np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) - 3 * np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1))
        return np.array(grad)
    
    def test_objective2_hessian(vec_2dim):
        term1 = np.exp(vec_2dim[0] + 3*vec_2dim[1] - 0.1)
        term2 = np.exp(vec_2dim[0] - 3*vec_2dim[1] - 0.1)
        hessian = [[term1 + term2 + np.exp(-vec_2dim[0]-0.1), 3*term1 - 3*term2],
                   [3*term1 - 3*term2, 9*term1 + 9*term2]]
        return np.array(hessian)

    # print(test_objective2_hessian([6, 12]), np.linalg.eigvalsh(test_objective2_hessian([6, 12]))) # singular!

    test_newton_inst2 = NewtonUnconstrained(
        test_objective2, 
        test_objective2_gradient, 
        test_objective2_hessian)
    test_newton_inst2.run_newton_with_backtracking_line_search(np.array([6, 12]), method="pinv")
    print(test_newton_inst2.get_minimizing_sequence())
    print(test_newton_inst2.get_decrement_sequence())
    print(test_newton_inst2.get_minimizing_function_value_sequence())
    
