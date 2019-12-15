import pandas as pd
import numpy as np
import random
import copy
from datafile import *
from matplotlib import pyplot as plt


# variables:
# A - amount of workers
# P - amount of publications
# solution  is a list of decisions [i][j] telling if publication j of author i is taken into evaluation


def fix_objective_function(solution_matrix):
    print('todo')


def cost_objective_function(solution_matrix, k=250):
    value = 0
    cost2 = 0
    for i in range(0, A - 1):
        cost1 = 0
        for j in range(0, P - 1):
            value += solution_matrix[i][j] * w[i][j]
            cost1 += solution_matrix[i][j] * u[i][j]
            cost2 += solution_matrix[i][j] * u[i][j]
        cost1 -= 4 * udzial[i]
        if cost1 > 0:
            value -= k * cost1

    if cost2 > 0:
        value -= k * cost2

    return value


def seed_solution():
    matrix = np.random.randint(2, size=(A, P))
    value = cost_objective_function(matrix)
    return [matrix, value]


def randomly_change_n_positions(chosen_publications, n):
    new_solution = copy.deepcopy(chosen_publications)
    for k in range(n):
        i = random.randint(0, A-1)
        j = random.randint(0, P - 1)
        if new_solution[0][i][j] > 0:
            new_solution[0][i][j] = 0
        else:
            new_solution[0][i][j] = 1

    new_solution[1] = cost_objective_function(new_solution[0])
    return new_solution


def local_search(chosen_publications, max_attempts=50, neighbourhood_size=5):
    count = 0
    local_solution = copy.deepcopy(chosen_publications)
    k = 1
    while count < max_attempts:
        candidate = randomly_change_n_positions(local_solution, k)
        if candidate[1] > chosen_publications[1]:
            local_solution = copy.deepcopy(candidate)
            k = 1
            count = 0
        else:
            k += 1
            if k > neighbourhood_size:
                k = 1
            count = count + 1
    return local_solution


def variable_neighborhood_search(max_attempts=20, neighbourhood_size=5, iterations=100):
    count = 0
    new_solution = seed_solution()
    best_solution = copy.deepcopy(new_solution)
    while count < iterations:
        for i in range(1, neighbourhood_size+1):
            new_solution = randomly_change_n_positions(best_solution, i)
            new_solution = local_search(chosen_publications=new_solution, max_attempts=max_attempts,
                                        neighbourhood_size=neighbourhood_size)
            if new_solution[1] > best_solution[1]:
                best_solution = copy.deepcopy(new_solution)
                break
        count = count + 1
        print("Iteration = ", count, ">> Score ", best_solution[1])
    return best_solution


################# Run script #####################
print("Algorithm start")
solution = variable_neighborhood_search()
print('End - best solution: ' + str(solution))
