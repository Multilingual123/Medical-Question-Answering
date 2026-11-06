import time

import numpy as np


# Barber Optimization Algorithm (BaOA)
def BOA(X, obj_func, lb, ub, T):
    N, dim = X.shape[0], X.shape[1]
    fitness = obj_func(X[:])  # Eq. (3): Evaluate initial population

    best_idx = np.argmin(fitness)
    best_score = fitness[best_idx]
    best_pos = X[best_idx].copy()

    convergence_curve = np.zeros((T, 1))
    ct = time.time()
    for t in range(T):
        for i in range(N):
            Fi = fitness[i]

            # === Eq. (4): Determine hairstyles set (better solutions) ===
            better_indices = [k for k in range(N) if fitness[k] < Fi and k != i]

            # === Phase 1: Exploration (Choosing hairstyle) ===
            if better_indices:
                k = np.random.choice(better_indices)  # Random hairstyle from better individuals
                SHS = X[k]  # Selected hairstyle

                ri = np.random.rand(dim)  # Random numbers in [0,1]
                Ii = np.random.choice([1, 2], size=dim)  # Randomly 1 or 2 for each dimension

                # === Eq. (5): Calculate new position based on selected hairstyle ===
                xP1 = X[i] + ri * (SHS - Ii * X[i])
                xP1 = np.clip(xP1, lb, ub)  # Ensure bounds
                fP1 = obj_func(xP1[:])

                # === Eq. (6): Accept new position if better ===
                if fP1[i] < Fi:
                    X[i] = xP1[i]
                    Fi = fP1

            # === Phase 2: Exploitation (Correction of hairstyle details) ===
            ri = np.random.rand(dim)
            # === Eq. (7): Small adjustments in position ===
            xP2 = X[i] + (1 - 2 * ri) * (ub - lb) / t
            xP2 = np.clip(xP2, lb, ub)
            fP2 = obj_func(xP2[:])

            # === Eq. (8): Accept new position if better ===
            if fP2[i] <= Fi:
                X[i] = xP2[i]
                Fi = fP2[i]

            fitness[i] = Fi

            if Fi < best_score:
                best_score = Fi
                best_pos = X[i].copy()

        convergence_curve[t] = best_score
    best_score = convergence_curve[T - 1][0]
    ct = time.time() - ct

    return best_score, convergence_curve, best_pos, ct
