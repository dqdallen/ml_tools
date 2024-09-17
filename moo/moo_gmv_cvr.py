from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import numpy as np

def conversion_obj(x, cvr, user_cnt):
    return -np.sum(cvr * x * user_cnt)

def gmv_obj(x, cvr, user_cnt, avg_gmv):
    return -np.sum(x*cvr*user_cnt*avg_gmv)

def cost_constraint(x, cvr, user_cnt, prize_amt, budget):
    return np.sum(x*cvr*prize_amt*user_cnt) - budget

def avg_cost_constraint(x, cvr, user_cnt, avg_gmv, prize_amt, avg_budget):
    total_cost = np.sum(x*cvr*prize_amt*user_cnt)
    total_cv = np.sum(x*cvr*user_cnt)
    return total_cost/total_cv - avg_budget

class CustomeProblem(ElementwiseProblem):
    def __init__(self, n_var, n_obj, n_constr, xl, xu, elementwise_evaluation=True, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.cvr = kwargs['cvr']
        self.user_cnt = kwargs['user_cnt']
        self.avg_gmv = kwargs['avg_gmv']
        self.prize_amt = kwargs['prize_amt']
        self.budget = kwargs['budget']
        self.avg_budget = kwargs['avg_budget']
        self.crowd_num = kwargs['crowd_num']
        self.rows, self.cols = self.cvr.shape
    
    def _evaluate(self, x, out, *args, **kwargs):
        x = np.reshape(x, (self.rows, self.cols))
        cv_obj = conversion_obj(x, self.cvr, self.user_cnt)
        g_obj = gmv_obj(x, self.cvr, self.user_cnt, self.avg_gmv)
        out["F"] = [cv_obj, g_obj]

        cost_c = cost_constraint(x, self.cvr, self.user_cnt, self.prize_amt, self.budget)
        avg_cost_c = cost_constraint(x, self.cvr, self.user_cnt, self.avg_gmv, self.prize_amt, self.avg_budget)
        out["G"] = [cost_c, avg_cost_c]

if __name__ == '__main__':
    n_var = 6
    n_obj = 2
    n_constr = 2
    xl = 0
    xu = 1
    crowd_num = 2
    cvr = np.array([[0.1, 0.2, 0.3], [0.05, 0.15, 0.25]])
    user_cnt = np.array([[100], [200]])
    avg_gmv = np.array([[240], [250]])
    prize_amt = np.array([[108, 208, 308], [108,208,308]])
    budget = 10800
    avg_budget = 200
    cp = CustomeProblem(n_var, n_obj, n_constr, xl, xu, cvr=cvr, user_cnt=user_cnt, avg_gmv=avg_gmv, prize_amt=prize_amt, budget=budget, avg_budget=avg_budget, crowd_num=crowd_num)
    algorithm = NSGA2()
    res = minimize(cp, algorithm, ("n_gen", 100), ("pop_size", 100), verbose=True)
    print('最优解: ', res.x)
    print('目标值：', res.F)