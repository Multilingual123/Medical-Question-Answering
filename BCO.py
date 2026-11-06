import numpy as np
import time


def BCO(initialPopulation, fitness, VRmin, VRmax, maxIterations):
    populationSize, nVar = initialPopulation.shape[0], initialPopulation.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    optimalSolution = np.zeros((nVar, 1))
    optimalFitness = float('inf')

    ct = time.time()
    # Main loop
    for iter in range(maxIterations):

        # Calculate fitness values
        fitnessValues = np.apply_along_axis(fitness, 1, initialPopulation)

        # Update personal best positions
        personalBestPositions = initialPopulation.copy()
        personalBestFitnessValues = fitnessValues.copy()

        for i in range(populationSize):
            if fitnessValues[i] < personalBestFitnessValues[i]:
                personalBestPositions[i, :] = initialPopulation[i, :]
                personalBestFitnessValues[i] = fitnessValues[i]

        # Update global best position
        globalBestPositionIndex = np.argmin(fitnessValues)
        globalBestPosition = initialPopulation[globalBestPositionIndex, :]

        # Update positions of search agents
        for i in range(populationSize):
            # Border collie behavior
            if np.random.rand() < 0.5:
                # Herding behavior
                for j in range(nVar):
                    if np.random.rand() < 0.5:
                        personalBestPosition = personalBestPositions[i, :]
                    else:
                        personalBestPosition = personalBestPositions[np.random.randint(populationSize), :]
                    initialPopulation[i, j] = personalBestPosition[j] + np.random.rand() * (
                            globalBestPosition[j] - personalBestPosition[j])
            else:
                # Sheep behavior
                for j in range(nVar):
                    if np.random.rand() < 0.5:
                        neighborPosition = personalBestPositions[np.random.randint(populationSize), :]
                    else:
                        neighborPosition = globalBestPosition
                    initialPopulation[i, j] = neighborPosition[j] + np.random.rand() * (
                            neighborPosition[j] - initialPopulation[i, j])

            # Boundary checking
            initialPopulation[i, :] = np.clip(initialPopulation[i, :], lb, ub)

        # Store convergence curve (optimal solution at the current iteration)
        optimalFitness = np.min(fitnessValues)
        optimalSolutionIndex = np.argmin(fitnessValues)
        optimalSolution = initialPopulation[optimalSolutionIndex, :]

    # Return the results
    ConvergenceCurve = optimalSolution
    end_time = time.time()
    elapsed_time = end_time - ct
    return optimalFitness, ConvergenceCurve, optimalSolution, elapsed_time

