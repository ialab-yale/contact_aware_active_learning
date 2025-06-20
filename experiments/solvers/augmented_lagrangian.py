import jax
from functools import partial
from jax import value_and_grad, grad, jacfwd, vmap, jit, hessian
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import jax.debug as deb
import numpy as onp
from tqdm import tqdm
from matplotlib import pyplot as plt

class AugmentedLagrangeSolver(object):
    def __init__(self, x0, loss, eq_constr, ineq_constr, args=None, step_size=1e-3, c=1.0):
        self.def_args = args
        self.loss = loss 
        self._c_def = c
        self.c = c
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self.solution = x0
        self.dual_solution = {'lam' : lam, 'mu' : mu}
        self.avg_sq_grad = {}
        for _key in self.solution:
            self.avg_sq_grad.update({_key : np.zeros_like(self.solution[_key])})

        self._prev_val = None
        def lagrangian(solution, dual_solution, args, c):
            lam = dual_solution['lam']
            mu  = dual_solution['mu']
            _eq_constr   = eq_constr(solution, args)
            _ineq_constr = ineq_constr(solution, args)
            _eq_penalty     = c * 0.5 * np.sum((_eq_constr + lam/c)**2)
            _ineq_penalty   = c * 0.5 * np.sum(np.maximum(0., mu/c + _ineq_constr)**2)
            return loss(solution, args) \
                + _eq_penalty \
                + _ineq_penalty

        val_dldx = jit(value_and_grad(lagrangian))
        _dldlam = jit(grad(lagrangian, argnums=1))
        gamma=0.9
        eps=1e-8
        @jit
        def step(solution, dual_solution, avg_sq_grad, args, c):
            _val, _dldx   = val_dldx(solution, dual_solution, args, c)
            for _key in solution:
                avg_sq_grad[_key] = avg_sq_grad[_key] * gamma + np.square(_dldx[_key]) * (1. - gamma)
                solution[_key] = solution[_key] - step_size * _dldx[_key] / np.sqrt(avg_sq_grad[_key] + eps)

            dual_solution['lam'] = np.clip(dual_solution['lam'] + c*eq_constr(solution, args),-100,100)
            dual_solution['mu']  = np.clip(np.maximum(0, dual_solution['mu'] + c*ineq_constr(solution, args)),0,100)
            return solution, dual_solution, avg_sq_grad, _val, _dldx

        self.lagrangian      = lagrangian
        self.grad_lagrangian = val_dldx
        self.step = step

    def reset_solver(self, x0, loss, eq_constr, ineq_constr, args=None, c=1.0):
        self.def_args = args
        self.loss = loss 
        self._c_def = c
        self.c = c
        self.eq_constr   = eq_constr
        self.ineq_constr = ineq_constr
        _eq_constr       = eq_constr(x0, args)
        _ineq_constr     = ineq_constr(x0, args)
        lam = np.zeros(_eq_constr.shape)
        mu  = np.zeros(_ineq_constr.shape)
        self.solution = x0
        self.dual_solution = {'lam' : lam, 'mu' : mu}
        self.avg_sq_grad = {}
        for _key in self.solution:
            self.avg_sq_grad.update({_key : np.zeros_like(self.solution[_key])})

        self._prev_val = None

    def reset(self):
        for _key in self.avg_sq_grad:
            self.avg_sq_grad.update({_key : np.zeros_like(self.avg_sq_grad[_key])})
        self.c = self._c_def

    def get_solution(self):
        return self.solution
        # return self._unravel(self._flat_solution)

    def solve(self, args=None, max_iter=100000, eps = 1e-5, alpha=1.001):
        if args is None:
            args = self.def_args
        _eps = 1.0
        _prev_val   = None
        _grad_total = 1.0
        # INIT LISTS
        _grad_list = [_grad_total]
        _eps_list = [_eps]
        _loss_list = []
        _lam_dual_list = [self.dual_solution['lam']]
        _mu_dual_list = [self.dual_solution['mu']]

        for k in range(max_iter):
            # self.solution, _val, self.avg_sq_grad = self.step(self.solution, args, self.avg_sq_grad, self.c)
            self.solution, self.dual_solution, self.avg_sq_grad, _val, _dldx = self.step(self.solution, self.dual_solution, self.avg_sq_grad, args, self.c)
            _grad_total = 0.0
            for _key in _dldx:
                _grad_total = _grad_total + np.sum(_dldx[_key]**2)
            self.c = alpha*self.c
            _grad_total = np.sqrt(_grad_total)
            if _prev_val is None:
                _prev_val = _val
                # APPEND LOSS LIST
                _loss_list.append(_val)
            else:
                _eps = onp.abs(_val - _prev_val)
                _prev_val = _val
                print(f'\r{"_grad_total: "} {_grad_total}', end='', flush=True)
                print(f'', end='', flush=True)
                # APPEND TO LISTS
                _grad_list.append(_grad_total)
                _eps_list.append(_eps)
                _loss_list.append(_val)
                _lam_dual_list.append(self.dual_solution['lam'])
                _mu_dual_list.append(self.dual_solution['mu'])
            if _grad_total < eps:
                print()
                print('done in ', k, ' iterations', _grad_total)
                # RETURN LISTS
                return _grad_list, _eps_list, _loss_list, _lam_dual_list, _mu_dual_list, k + 1
        print()
        print('unsuccessful, tol: ', _grad_total)
        # RETURN LISTS
        return _grad_list, _eps_list, _loss_list, _lam_dual_list, _mu_dual_list, max_iter
    
    def find_minmax(self, ineq_constr_fun, xrange):
        feasible_set_x = set()
        for ele in range(ineq_constr_fun.shape[0]):
            if ineq_constr_fun[ele] <= 0.:
                feasible_set_x.add(float(xrange[ele]))
        return np.min(np.array(list(feasible_set_x))), np.max(np.array(list(feasible_set_x)))

if __name__=='__main__':
    '''
        Example use case 
    '''

    import time

    data = jax.random.uniform(key=jax.random.PRNGKey(10), shape=(2000,4))

    def f(params, args=None) : 
        x = params['x']
        return np.mean(x[0] - args['data'])
    def g(params, args) : 
        x = params['x']
        return np.array([x[0]**2 - 1])
    def h(params, args) : 
        x = params['x']
        return np.array([0])

    x0 = np.array([-.5])
    params = {'x' : x0}
    args={'data': data}
    opt = AugmentedLagrangeSolver(params,f,h,g,args=args,step_size=0.01)
    time_now = time.time()
    opt.solve(max_iter=1000, alpha=1.001)
    print('time: ',time.time() - time_now)
    sol = opt.get_solution()
    print(sol)