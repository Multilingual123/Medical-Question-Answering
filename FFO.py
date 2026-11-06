import time
import numpy as np


# Fennec Fox Optimization (FFO)
def FFO(Search_Agents, objective, Lowerbound, Upperbound, Max_iterations):
    N, dimensions = Search_Agents.shape[0], Search_Agents.shape[1]
    X = Search_Agents
    fit = objective(Search_Agents[:])

    # Initialize best solution
    fbest = np.min(fit)
    blocation = np.argmin(fit)
    xbest = X[blocation, :]

    # Record the convergence curve
    FFO_curve = np.zeros(Max_iterations)

    ct = time.time()
    for t in range(Max_iterations):
        # Update best solution found so far
        best = np.min(fit)
        blocation = np.argmin(fit)

        if best < fbest:
            fbest = best
            xbest = X[blocation, :]

        # Fennec Fox optimization process: Exploration and Exploitation phases
        for i in range(Search_Agents.shape[0]):
            # Exploration Phase: Foxes search for food (global exploration)
            if np.random.rand() < 0.5:
                # Random position in the search space
                X_new = np.random.uniform(Lowerbound.shape[1], Upperbound.shape[1], dimensions)
            else:
                # Local search: explore around current position
                r = np.random.rand()
                X_new = X[i, :] + r * (xbest - X[i, :])

            # Boundary handling (ensure that the foxes stay within the boundaries)
            X_new = np.clip(X_new, Lowerbound, Upperbound)

            # Fitness evaluation of the new position
            fit_new = objective(X_new)

            # Update the position if the new position has better fitness
            if fit_new[i] < fit[i]:
                X[i, :] = X_new[i]
                fit[i] = fit_new[i]

            # Exploitation Phase: Exploit the best food source found
            if np.random.rand() < 0.5:
                R = 0.1 * (1 - t / Max_iterations)  # Adaptive factor for exploitation
                X_new = X[i, :] + R * (xbest - X[i, :])
                X_new = np.clip(X_new, Lowerbound, Upperbound)  # Boundary check

                # Fitness evaluation
                fit_new = objective(X_new)
                if fit_new[i] < fit[i]:
                    X[i, :] = X_new[i]
                    fit[i] = fit_new[i]

        # Save best score for the current iteration
        FFO_curve[t] = fbest
    ct = time.time() - ct
    return fbest, FFO_curve, xbest, ct
