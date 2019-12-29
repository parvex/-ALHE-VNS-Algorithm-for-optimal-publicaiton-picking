import numpy as np
import random
import copy
import math
from datafile import *

N = sum(czyN)


class Solution:
    def __init__(self, point, fix=False):
        self.point = point
        self.value = cost_objective_function(point)


def fix_objective_function(solution_matrix):
    # todo
    print('todo')


def greedy_pick_point_gen():
    point = np.zeros((A, P), dtype=int)
    for i in range(0, A - 1):
        param_list = [(j, w[i][j] / u[i][j]) for j in range(0, P)]
        param_list.sort(key=lambda x: x[1], reverse=True)
        u_value = u[i][param_list[0][0]]
        k = 1
        while u_value < 4 * udzial[i]:
            point[i][param_list[k - 1][0]] = 1
            u_value += u[i][param_list[k][0]]

    return point


def random_pick_point_gen():
    point = np.zeros((A, P), dtype=int)
    for i in range(0, A - 1):
        indexes = random.sample(range(0, P), P)
        u_value = u[i][indexes[0]]
        k = 1
        while u_value < 4 * udzial[i]:
            point[i][indexes[k - 1]] = 1
            u_value += u[i][indexes[k]]

    return point


def gen_starting_points():
    points = [Solution(np.zeros((A, P), dtype=int)), Solution(np.ones((A, P), dtype=int)),
              Solution(greedy_pick_point_gen())]
    for i in range(25):
        points.append(random_pick_point_gen())
    return points


def cost_objective_function(solution_matrix, k=250):
    value = 0
    cost2 = 0
    for i in range(A):
        cost1 = 0
        for j in range(P):
            value += solution_matrix[i][j] * w[i][j]
            cost1 += solution_matrix[i][j] * u[i][j]
            cost2 += solution_matrix[i][j] * u[i][j]
        cost1 -= 4 * udzial[i]

        if cost1 > 0:
            value -= k * (1 + cost1)

    cost2 -= 3 * N
    if cost2 > 0:
        value -= k * (1 + cost2)

    return value


def randomly_change_n_positions(chosen_publications, n):
    new_point = copy.deepcopy(chosen_publications.point)
    for k in range(n):
        i = random.randint(A)
        j = random.randint(P)
        if new_point[i][j] > 0:
            new_point[i][j] = 0
        else:
            new_point[i][j] = 1

    return Solution(new_point)


def variable_neighborhood_search(init_solution=random_pick_point_gen(), search_proportion=1 / 10,
                                 max_neighborhood_radius=A * P):
    count = 0
    best_solution = copy.deepcopy(init_solution)
    n = math.floor(search_proportion * A * P)
    radius = 1
    iteration = 0
    stage = 10
    stop = False
    while not stop:
        neighborhood = []
        for i in range(n * radius):
            neighborhood.append(randomly_change_n_positions(best_solution, radius))
            count += 1
            if count >= stage * A * P:
                file.write("Stage: " + str(stage) + " * bits >> best_value:" + str(best_solution.value))
                if stage == 1000:
                    stop = True
                    break
                stage *= 10

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


# Run script
def main():
    print("Algorithm start")
    points = gen_starting_points()
    for i, point in enumerate(points):
        print("Calculating point ", i)
        variable_neighborhood_search(point)
    print("Algorithm end")


if __name__ == '__main__':
    file = open("results", "w+")
    main()
