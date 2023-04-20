import numpy as np

class DescentUnconstrained:
    def __init__(self, fn_objective, fn_objective_gradient, fn_objective_domain_indicator = None):
        
        self.objective = fn_objective
        self.objective_gradient = fn_objective_gradient
        if fn_objective_domain_indicator is not None:
            self.objective_domain_indicator = fn_objective_domain_indicator
        else:
            self.objective_domain_indicator = self._Rn_domain_indicator

        self.minimizing_sequence = []
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
    
    def _descent_direction_gradient_descent_L2(self, eval_pt):
        return self.objective_gradient(eval_pt) * (-1)

    def _descent_direction_steepest_descent_L1(self, eval_pt):
        gradient = self.objective_gradient(eval_pt)
        max_index, max_abs_val = (None, 0)
        for i, val in enumerate(gradient):
            abs_val = abs(val)
            if abs_val > max_abs_val:
                max_index = i
                max_abs_val = abs_val
        direction_step = np.zeros(len(gradient))
        direction_step[max_index] = gradient[max_index] * (-1)
        return direction_step

    def _descent_direction_steepest_descent_quadratic_norm(self, eval_pt, inv_quadratic_norm_matrix):
        gradient = self.objective_gradient(eval_pt)
        direction_step = np.matmul(inv_quadratic_norm_matrix, gradient) * (-1)
        return direction_step

    def run_gradient_descent_with_backtracking_line_search(self, starting_pt, tolerance = 0.001,
                                                        a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            descent_direction = self._descent_direction_gradient_descent_L2(eval_pt)
            if self._l2_norm(descent_direction) < tolerance:
                break
            descent_step_size = self._backtracking_line_search(eval_pt, descent_direction, 
                                    a_slope_flatter_ratio, b_step_shorten_ratio)
            next_point = eval_pt + descent_direction * descent_step_size
            self.minimizing_sequence.append(next_point)
            self.value_sequence.append(self.objective(next_point))
            num_iter += 1

        print("iteration: ", num_iter)
    
    def run_steepest_descent_L1_with_backtracking_line_search(self, starting_pt, tolerance = 0.001,
                                                            a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            descent_direction = self._descent_direction_steepest_descent_L1(eval_pt)
            if self._l2_norm(descent_direction) < tolerance:
                break
            descent_step_size = self._backtracking_line_search(eval_pt, descent_direction, 
                                    a_slope_flatter_ratio, b_step_shorten_ratio)
            next_point = eval_pt + descent_direction * descent_step_size
            self.minimizing_sequence.append(next_point)
            self.value_sequence.append(self.objective(next_point))
            num_iter += 1

        print("iteration: ", num_iter)

    def run_steepest_descent_quadratic_norm_with_backtracking_line_search(self, starting_pt, quadratic_norm_matrix, tolerance = 0.001,
                                                                        a_slope_flatter_ratio = 0.2, b_step_shorten_ratio = 0.5):
        self.minimizing_sequence = [starting_pt]
        self.value_sequence = [self.objective(starting_pt)]
        inv_quadratic_norm_matrix = np.linalg.inv(quadratic_norm_matrix)
        num_iter = 0
        while True:
            eval_pt = self.minimizing_sequence[-1]
            descent_direction = self._descent_direction_steepest_descent_quadratic_norm(eval_pt, inv_quadratic_norm_matrix)
            if self._l2_norm(descent_direction) < tolerance:
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


    test_descent_inst = DescentUnconstrained(test_objective1, test_objective1_gradient)
    test_descent_inst.run_gradient_descent_with_backtracking_line_search(np.array([5, 12]), tolerance=0.1)
    print(test_descent_inst.get_minimizing_sequence())
    print(test_descent_inst.get_minimizing_function_value_sequence())

    test_descent_inst.run_steepest_descent_L1_with_backtracking_line_search(np.array([5, 12]), tolerance=0.1)
    print(test_descent_inst.get_minimizing_sequence())
    print(test_descent_inst.get_minimizing_function_value_sequence())


    #test 2
    def test_objective2(vec_2dim):
        val = np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) + np.exp(-vec_2dim[0] - 0.1)
        return np.array(val)

    def test_objective2_gradient(vec_2dim):
        grad = (np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) + np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1) - np.exp(-vec_2dim[0] - 0.1),
                3 * np.exp(vec_2dim[0] + 3 * vec_2dim[1] - 0.1) - 3 * np.exp(vec_2dim[0] - 3 * vec_2dim[1] - 0.1))
        return np.array(grad)


    test_descent_inst2 = DescentUnconstrained(test_objective2, test_objective2_gradient)
    test_descent_inst2.run_gradient_descent_with_backtracking_line_search(np.array([5, 12]))
    print(test_descent_inst2.get_minimizing_sequence())
    print(test_descent_inst2.get_minimizing_function_value_sequence())
    
    test_descent_inst2.run_steepest_descent_L1_with_backtracking_line_search(np.array([5, 12]), tolerance=0.1)
    print(test_descent_inst2.get_minimizing_sequence())
    print(test_descent_inst2.get_minimizing_function_value_sequence())

    norm_mat_1 = np.array([[2,0], [0,8]])
    test_descent_inst2.run_steepest_descent_quadratic_norm_with_backtracking_line_search(np.array([5, 12]), norm_mat_1, tolerance=0.1)
    print(test_descent_inst2.get_minimizing_sequence())
    print(test_descent_inst2.get_minimizing_function_value_sequence())

    norm_mat_2 = np.array([[8,0], [0,2]])
    test_descent_inst2.run_steepest_descent_quadratic_norm_with_backtracking_line_search(np.array([5, 12]), norm_mat_2, tolerance=0.1)
    print(test_descent_inst2.get_minimizing_sequence())
    print(test_descent_inst2.get_minimizing_function_value_sequence())
