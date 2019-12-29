import numpy as np
import random
import copy
import math
from datafile import *

class Solution:
    def __init__(self, point):
        self.point = point
        self.value = cost_objective_function(point)

def fix_objective_function(solution_matrix):
    print('todo')


def gen_starting_points():
    points = []
    points.append(Solution(np.zeros((A, P), dtype=int)))
    points.append(Solution(np.ones((A, P), dtype=int)))

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
        cost2 -= 3*udzial[i]
        if cost1 > 0:
            value -= k * (1 + cost1)

    if cost2 > 0:
        value -= k * (1 + cost2)

    return value


def seed_solution():
    matrix = np.random.randint(2, size=(A, P))
    return Solution(matrix)


def randomly_change_n_positions(chosen_publications, n):
    new_point = copy.deepcopy(chosen_publications.point)
    for k in range(n):
        i = random.randint(0, A-1)
        j = random.randint(0, P - 1)
        if new_point[i][j] > 0:
            new_point[i][j] = 0
        else:
            new_point[i][j] = 1

    return Solution(new_point)


def variable_neighborhood_search(init_solution=seed_solution(), search_proportion = 1/10, max_neighborhood_radius=50):
    count = 0
    best_solution = copy.deepcopy(init_solution)
    #max number of solutions to find in neighborhood 1 - to improve?
    n = math.floor(search_proportion * A * P)
    radius = 1
    iteration = 0
    while count < 1000 * A * P:
        neighborhood = []
        for i in range(n*radius):
            neighborhood.append(randomly_change_n_positions(best_solution, radius))
            count += 1
        best_in_neighborhood = max(neighborhood, key=lambda x: x.value)
        if best_in_neighborhood.value > best_solution.value:
            best_solution = copy.deepcopy(best_in_neighborhood)
            radius = 1
        else:
            radius += 1
            if radius > max_neighborhood_radius:
                radius = 1
        print("Iteration = ", iteration, ">> Score ", best_solution.value, "count = ", count)
        iteration += 1

    return best_solution

################# Run script #####################
print("Algorithm start")
solution = variable_neighborhood_search(seed_solution())
print('End - best solution: ' + str(solution))
