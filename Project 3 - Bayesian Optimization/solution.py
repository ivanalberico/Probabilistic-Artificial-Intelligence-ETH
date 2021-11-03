import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
import math
from scipy.special import gamma, kv

from scipy.stats import norm

domain = np.array([[0, 5]])

""" Solution """


class BO_algo():

    class GaussianProcess:
        def __init__(self,noise, mean, kernel_var, kernel_rho, kernel_smo, train_x, train_y):
            self.s = noise
            self.mean = mean
            self.kernel_var = kernel_var
            self.l = kernel_rho
            self.v = kernel_smo
            self.train_x = train_x
            self.train_y = train_y
            self.N = len(train_x)
            self.X = np.array(train_x).reshape((self.N, 1))
            self.K_AA = np.zeros((self.N, self.N))
            self.mu_A = np.zeros((self.N, 1))
            self.y_A = np.array(train_y).reshape((self.N, 1))

            for i in range(self.N):
                self.mu_A[i, 0] = self.mu_pri(self.X[i, 0])
                for j in range(self.N):
                    self.K_AA[i, j] = self.K_pri(self.X[i, 0], self.X[j, 0])
            self.I_AA = np.linalg.inv(self.K_AA + self.s ** 2 * np.eye(self.N))

        def K_pri(self, x1, x2):
            r = np.abs(x1 - x2)
            if r == 0:
                r = 1e-8
            part1 = 2 ** (1 - self.v) / gamma(self.v)
            part2 = (np.sqrt(2 * self.v) * r / self.l) ** self.v
            part3 = kv(self.v, np.sqrt(2 * self.v) * r / self.l)
            return self.kernel_var * part1 * part2 * part3

        def mu_pri(self, x):
            return self.mean

        def mu_pos(self, x):
            K_xA = np.zeros((1, self.N))
            for i in range(self.N):
                K_xA[0, i] = self.K_pri(x, self.X[i, 0])
            return self.mu_pri(x) + np.matmul(np.matmul(K_xA, self.I_AA), (self.y_A - self.mu_A))[0][0]

        def var_pos(self, x1, x2):
            K_x1A = np.zeros((1, self.N))
            for i in range(self.N):
                K_x1A[0, i] = self.K_pri(x1, self.X[i, 0])
            K_x2A = np.zeros((1, self.N))
            for i in range(self.N):
                K_x2A[0, i] = self.K_pri(x2, self.X[i, 0])
            return self.K_pri(x1, x2) - np.matmul(K_x1A, np.matmul(self.I_AA, np.transpose(K_x2A)))[0][0]

        def plot_results(self):
            x = np.linspace(0, 5, 50)

            y = []
            v = []
            for i in range(len(x)):
                y.append(self.mu_pos(x[i]))
                v.append(np.sqrt(self.var_pos(x[i], x[i])))
            y = np.array(y)
            plt.plot(x, y)
            plt.fill_between(x, y - v, y + v,
                             color='b', alpha=0.2, label='Predictive Distribution')
            plt.scatter(self.train_x, self.train_y)
            plt.show()

    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.x = np.zeros(1)
        self.y = np.zeros(1)

        self.h = 0.1
        self.s = 1

        self.beta = 2
        self.N = 0

        self.x_values = []
        self.c_values = []



    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if self.N == 0:
            x = np.array([[np.random.uniform(0, 5)]])
        else:
            x = self.optimize_acquisition_function()

        return x



    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * \
                 np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])



    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here

        """For Gaussian process, evaluate GP at x"""
        mean_f = self.GP_f.mu_pos(x)
        std_f = np.sqrt(self.GP_f.var_pos(x, x))

        mean_v = self.GP_v.mu_pos(x)
        std_v = np.sqrt(abs(self.GP_v.var_pos(x, x)))

        penalty = 3

        enable_UCB = True
        enable_EI = ~ enable_UCB

        if enable_UCB:

            if mean_v - 0.2 * std_v <= 1.2:
                return mean_f + std_f * self.beta - penalty
            else:
                return mean_f + std_f * self.beta


        if enable_EI:

            mean_sample = []

            for i in range(len(self.train_x)):
                mean_sample.append(self.GP_f.mu_pos(self.train_x[i]))

            mu_sample_opt = np.max(mean_sample)

            xi = 0.01

            imp = mean_f - mu_sample_opt - xi
            Z = imp / std_f

            EI = imp * norm.cdf(Z) + std_f * norm.pdf(Z)

            if mean_v - 0.2 * std_v <= 1.2:
                return EI - penalty
            else:
                return EI



    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        """For Gaussian process, add point x and y = c(v,f)"""
        if self.N == 0:
            self.train_x = [x]
            self.train_f = [f]
            self.train_v = [v]

        else:
            self.train_x.append(x)
            self.train_f.append(f)
            self.train_v.append(v)


        self.N += 1


        self.GP_f = self.GaussianProcess(0.15, 0, 0.5, 0.5, 2.5, self.train_x, self.train_f)
        self.GP_v = self.GaussianProcess(0.0001, 1.5, np.sqrt(2), 0.5, 2.5, self.train_x, self.train_v)






    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        index = -1
        max_f = -math.inf

        for i in range(self.N):
            if self.train_v[i] > 1.2 and self.train_f[i] > max_f:
                max_f = self.train_f[i]
                index = i
        return self.train_x[index]





""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    print("using this")
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    if x > 1:
        return 2
    else:
        return 1


def main():
    # Init problem
    agent = BO_algo()

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}')


if __name__ == "__main__":
    main()