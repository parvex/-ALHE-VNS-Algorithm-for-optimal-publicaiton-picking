import numpy as np
import random
import copy
import math
from datafile import *

N = sum(czyN)
file = open("results.txt", "w+")
fix_function = False


# todo
def fix_point(point):
    pass
    return 0


class Solution:
    def __init__(self, point):
        self.point = point
        if fix_function:
            self.fixed_point = fix_point(point)
            self.value = value_function(self.fixed_point)
        else:
            self.value = value_function(point)


def greedy_pick_point_gen():
    point = np.zeros((A, P), dtype=int)
    for i in range(A):
        param_list = [((j, w[i][j] / u[i][j]) if u[i][j] != 0 else (j, 0)) for j in range(0, P)]
        param_list.sort(key=lambda x: x[1], reverse=True)
        u_value = u[i][param_list[0][0]]
        k = 1
        while u_value <= 4 * udzial[i]:
            point[i][param_list[k - 1][0]] = 1
            if k >= P-1:
                break
            u_value += u[i][param_list[k][0]]
            k += 1

    return point


def random_pick_point_gen():
    point = np.zeros((A, P), dtype=int)
    for i in range(A):
        indexes = random.sample(range(P), P)
        u_value = u[i][indexes[0]]
        k = 1
        while u_value <= 4 * udzial[i]:
            point[i][indexes[k - 1]] = 1
            u_value += u[i][indexes[k]]
            if k >= P-1:
                break
            k += 1

    return point


def gen_starting_points():
    points = [Solution(np.zeros((A, P), dtype=int)), Solution(np.ones((A, P), dtype=int)),
              Solution(greedy_pick_point_gen())]
    for i in range(25):
        points.append(Solution(random_pick_point_gen()))
    return points


def value_function(solution_matrix, k=250):
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
        i = random.randint(0, A-1)
        j = random.randint(0, P-1)
        new_point[i][j] = 1 if new_point[i][j] == 0 else 1

    return Solution(new_point)


def variable_neighborhood_search(init_solution, search_proportion=1 / 10,
                                 max_neighborhood_radius=A * P):
    best_solution = init_solution
    count = 0
    iteration = 0
    radius = 1
    stage = 10
    stop = False
    while not stop:
        neighborhood = []
        for i in range(math.floor(search_proportion * A * P * radius)):
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
            best_solution = best_in_neighborhood
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
    print("Cost function")
    file.write("---Cost function---")
    points = gen_starting_points()
    for i, point in enumerate(points):
        print("Calculating point " + str(i))
        file.write("Calculating point " + str(i))
        variable_neighborhood_search(point)

    # print("Fix function")
    # file.write("---Fix function---")
    # global fix_function
    # fix_function = True
    # for i, point in enumerate(points):
    #     file.write("Calculating point " + str(i))
    #     variable_neighborhood_search(point)
    file.close()
    print("Algorithm end")


if __name__ == "__main__":
    main()





