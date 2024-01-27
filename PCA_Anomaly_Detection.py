
import numpy as np


def Algorithm_NeuralNetworkNinjas(F, N_x, bounds, max_eval):
    
    rho_ini = 3.0 #Trust Region Radius
    tol = 1e-15 #Tolerance for convergence

    def penalty_function(x, constraints):
        return F(x) + sum(max(0, constraint(x))**2 for constraint in constraints)
        #Quadratic penalty term

    x = np.random.uniform(bounds[:, 0],  bounds[:, 1], size=N_x)
    rho = rho_ini

    for iter_number in range(max_eval):
        x_old = np.copy(x) #Saving current solution to compare for convergence later

        constraints = [lambda x: x[i] - bounds[:, 0][i] for i in range(N_x)] + \
                      [lambda x: bounds[:, 1][i] - x[i] for i in range(N_x)] 

        f_penalty = lambda x: penalty_function(x, constraints) #Calling penalty function

        for i in range(N_x):
            v = np.zeros(N_x)
            v[i] = 1.0  #Creating unit vector V in all dimensions to be able to explore each coordinate axis independantly

            x_plus = x + rho * v #moves from current solution in positive direction of vector v scaled by trust region radius
            x_minus = x - rho * v  #moves from current solution in negative direction of vector v scaled by trust region radius

            if f_penalty(x_plus) < f_penalty(x_minus):
                x = x_plus
            else:
                x = x_minus #Compares performance of 2 directions and decides which direction is better

        if np.linalg.norm(x - x_old) < tol: #If euclidean distance is smaller than tolerance we can assume converged
            break

        rho *= 0.7 #reducing trust region by 30% each iteration (multiplying by 0.7 trust region)

    best_f = F(x)

    return x, best_f