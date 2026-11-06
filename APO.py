import numpy as np
import time


# Artificial Protozoa Optimizer (APO)
def APO(protozoa, fhd, Xmin, Xmax, iter_max):
    pop_size, dim = protozoa.shape[0], protozoa.shape[1]
    start_time = time.time()
    ps = pop_size
    npair = 1
    pf_max = 0.1

    protozoa = Xmin + np.random.rand(ps, dim) * (Xmax - Xmin)
    newprotozoa = np.zeros_like(protozoa)
    epn = np.zeros((npair, dim))

    protozoa_Fit = fhd(protozoa[:])
    bestval = np.min(protozoa_Fit)
    bestid = np.argmin(protozoa_Fit)
    bestProtozoa = protozoa[bestid]
    bestFit = bestval
    Convergence_curve = np.zeros((iter_max, 1))

    t = 0
    for iter in range(iter_max):
        sorted_indices = np.argsort(protozoa_Fit)
        protozoa = protozoa[sorted_indices]
        protozoa_Fit = protozoa_Fit[sorted_indices]

        pf = pf_max * np.random.rand()
        ri = np.random.permutation(ps)[:int(np.ceil(ps * pf))]

        for i in range(ps):
            if i in ri:
                pdr = 0.5 * (1 + np.cos((1 - i / ps) * np.pi))
                if np.random.rand() < pdr:
                    newprotozoa = Xmin + np.random.rand(dim) * (Xmax - Xmin)
                else:
                    Flag = np.random.choice([1, -1])
                    Mr = np.zeros(dim)
                    Mr[np.random.permutation(dim)[:int(np.ceil(np.random.rand() * dim))]] = 1
                    newprotozoa = protozoa[i] + Flag * np.random.rand() * (
                            Xmin + np.random.rand(dim) * (Xmax - Xmin)) * Mr
            else:
                f = np.random.rand() * (1 + np.cos(iter / iter_max * np.pi))
                Mf = np.zeros(dim)
                Mf[np.random.permutation(dim)[:int(np.ceil(dim * i / ps))]] = 1
                pah = 0.5 * (1 + np.cos(iter / iter_max * np.pi))
                if np.random.rand() < pah:
                    j = np.random.randint(ps)
                    for k in range(npair):
                        km = i if i == 0 else np.random.randint(i)
                        kp = i + np.random.randint(1, ps - i) if i < ps - 1 else i
                        wa = np.exp(-abs(protozoa_Fit[km] / (protozoa_Fit[kp] + np.finfo(float).eps)))
                        epn[k] = wa * (protozoa[km] - protozoa[kp])
                    newprotozoa[i] = protozoa[i] + f * (protozoa[j] - protozoa[i] + np.sum(epn, axis=0) / npair) * Mf
                else:
                    for k in range(npair):
                        imk = max(i - k, 0)
                        ipk = min(i + k, ps - 1)
                        wh = np.exp(-abs(protozoa_Fit[imk] / (protozoa_Fit[ipk] + np.finfo(float).eps)))
                        epn[k] = wh * (protozoa[imk] - protozoa[ipk])
                    Flag = np.random.choice([1, -1])
                    Xnear = (1 + Flag * np.random.rand(dim) * (1 - iter / iter_max)) * protozoa[i]
                    newprotozoa[i] = protozoa[i] + f * (Xnear - protozoa[i] + np.sum(epn, axis=0) / npair) * Mf

        newprotozoa = np.clip(newprotozoa, Xmin, Xmax)
        newprotozoa_Fit = fhd(newprotozoa[:])

        improvement_mask = protozoa_Fit > newprotozoa_Fit
        protozoa[improvement_mask] = newprotozoa[improvement_mask]
        protozoa_Fit[improvement_mask] = newprotozoa_Fit[improvement_mask]

        bestFit = np.min(protozoa_Fit)
        bestid = np.argmin(protozoa_Fit)
        bestProtozoa = protozoa[bestid]
        Convergence_curve[t] = bestFit
        t = t + 1
    bestFit = Convergence_curve[iter_max - 1][0]
    recordtime = time.time() - start_time

    return bestFit, Convergence_curve, bestProtozoa, recordtime
