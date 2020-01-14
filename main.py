import numpy as np
import random
import copy
import math
import os

A = 0
P = 0
udzial = []
czyN = []
u = []
w = []
N = sum(czyN)
fix_function = False

class Data:
    def __init__(self, A, P, udzial, czyN, u, w, N):
        self.A = copy.deepcopy(A)
        self.P = copy.deepcopy(P)
        self.udzial = np.array(udzial)
        self.czyN = np.array(czyN)
        self.u = np.array(u)
        self.w = np.array(w)
        self.N = np.array(N)


def fix_point(point, data):
    fixed_point = copy.deepcopy(point)
    author_costs = np.sum(fixed_point * data.u, axis=1) - 4*data.udzial

    for i in range(data.A):
        #if author cost is greater than 0
        if author_costs[i] > 0:
            author_publications = data.w[i] / data.u[i] / fixed_point[i]
            min_ind = np.unravel_index(np.nanargmin(author_publications, axis=None), author_publications.shape)
            fixed_point[i][min_ind] = 0
            author_publications[min_ind] = float("NaN")
            # if author cost is greater than 0
            while np.sum(fixed_point[i] * data.u[i]) - 4 * data.udzial[i] > 0:
                min_ind = np.unravel_index(np.nanargmin(author_publications, axis=None), author_publications.shape)
                fixed_point[i][min_ind] = 0
                author_publications[min_ind] = float("NaN")

    #if university cost is greater than 0
    if np.sum(fixed_point * data.u) - 3*data.N > 0:
        publications = data.w / data.u / fixed_point
        min_ind = np.unravel_index(np.nanargmin(publications, axis=None), publications.shape)
        fixed_point[min_ind[0]][min_ind[1]] = 0
        publications[min_ind[0]][min_ind[1]] = float("NaN")
        while np.sum(fixed_point * data.u) - 3*data.N > 0:
            min_ind = np.unravel_index(np.nanargmin(publications, axis=None), publications.shape)
            fixed_point[min_ind[0]][min_ind[1]] = 0
            publications[min_ind[0]][min_ind[1]] = float("NaN")

    return fixed_point


class Solution:
    def __init__(self, point, data):
        if fix_function:
            self.point = fix_point(point, data)
            self.value = value_function(self.point, data)
        else:
            self.point = point
            self.value = cost_value_function(point, data)


def greedy_pick_point_gen(data):
    point = np.zeros((data.A, data.P), dtype=int)
    for i in range(data.A):
        param_list = [((j, data.w[i][j] / data.u[i][j]) if data.u[i][j] != 0 else (j, 0)) for j in range(0, data.P)]
        param_list.sort(key=lambda x: x[1], reverse=True)
        u_value = data.u[i][param_list[0][0]]
        k = 1
        while u_value <= 4 * data.udzial[i]:
            point[i][param_list[k - 1][0]] = 1
            if k >= data.P-1:
                break
            u_value += data.u[i][param_list[k][0]]
            k += 1

    return point


def random_pick_point_gen(data):
    point = np.zeros((data.A, data.P), dtype=int)
    for i in range(data.A):
        indexes = random.sample(range(data.P), data.P)
        u_value = data.u[i][indexes[0]]
        k = 1
        while u_value <= 4 * data.udzial[i]:
            point[i][indexes[k - 1]] = 1
            u_value += data.u[i][indexes[k]]
            if k >= data.P-1:
                break
            k += 1

    return point


def gen_starting_points(data):
    points = [Solution(np.zeros((data.A, data.P), dtype=int), data),
              Solution(np.ones((data.A, data.P), dtype=int), data),
              Solution(greedy_pick_point_gen(data), data)]
    for i in range(25):
        points.append(Solution(random_pick_point_gen(data), data))
    return points


def value_function(solution_matrix, data):
    value = np.sum(solution_matrix * data.w)
    return value


def cost_value_function(solution_matrix, data, k=250):
    value = np.sum(solution_matrix * data.w)
    cost_matrix = solution_matrix * data.u
    university_cost = np.sum(cost_matrix) - 3*N
    author_costs = np.sum(cost_matrix, axis=1) - 4*data.udzial

    for i in range(data.A):
        if author_costs[i] > 0:
            value -= k*(1 + author_costs[i])

    if university_cost > 0:
        value -= k * (1 + university_cost)

    return value


def randomly_change_n_positions(chosen_publications, n, data):
    new_point = copy.deepcopy(chosen_publications.point)
    for k in range(n):
        i = random.randint(0, data.A-1)
        j = random.randint(0, data.P-1)
        new_point[i][j] = 1 if new_point[i][j] == 0 else 0

    return Solution(new_point, data)


def variable_neighborhood_search(init_solution, search_proportion, max_neighborhood_radius, data, file):
    best_solution = init_solution
    count = 0
    iteration = 0
    radius = 1
    stage = 1
    stop = False
    while not stop:
        neighborhood = []
        for i in range(math.floor(search_proportion * data.A * data.P * radius)):
            neighborhood.append(randomly_change_n_positions(best_solution, radius, data))
            count += 1
            if count >= data.A * data.P * stage:
                file.write("Stage: " + str(stage) + "K >> best_value: " + str(best_solution.value) + "\n")
                file.write("Found point:\n")
                np.savetxt(file, best_solution.point.astype(int), fmt='%i')
                file.flush()
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
        print("Iteration = ", iteration, "Radius = ", radius , ">> Score ", best_solution.value, "count = ", count)
        iteration += 1

    return best_solution


def run_file(datafile):
    neighborhood_param = 1 / 10
    max_radius = 20
    file = open("results/" + datafile + ".txt", "w+")
    data = Data(A, P, udzial, czyN, u, w, N)
    points = gen_starting_points(data)

    # calculation for cost function
    print("File - " + datafile)
    file.write("File - " + datafile + "\n")
    print("Cost function")
    file.write("---Cost function---\n")
    file.flush()
    for i, point in enumerate(points):
        print("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius) + " end at: count = " + str(1000 * A * P))
        file.write("Calculating point: " + str(i) + " neighborhood param: "
                   + str(neighborhood_param) + " max radius: " + str(max_radius) + "\n"
                   + "Point:\n")
        np.savetxt(file, point.point.astype(int), fmt='%i')
        file.flush()
        solution = variable_neighborhood_search(point, neighborhood_param, max_radius, data, file)
        file.write("Found point: " + str(solution.point) + "\n")
        file.flush()

    # calculation for fix function
    global fix_function
    fix_function = True
    print("File - " + datafile)
    file.write("File - " + datafile + "\n")
    print("Fix function")
    file.write("---Fix function---\n")
    file.flush()
    for i, point in enumerate(points):
        print("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius) + " end at: count = " + str(1000 * A * P))
        file.write("Calculating point: " + str(i) + " neighborhood param: "
                   + str(neighborhood_param) + " max radius: " + str(max_radius) + "\n"
                   + "Point:\n")
        np.savetxt(file, point.point.astype(int), fmt='%i')
        file.flush()
        solution = variable_neighborhood_search(point, neighborhood_param, max_radius, data, file)
        file.write("Found point: " + str(solution.point) + "\n")
        file.flush()
    file.close()

if __name__ == "__main__":
    datafiles = os.listdir("data")
    np.seterr(divide='ignore', invalid='ignore')
    #updating global variables
    exec(open("data/" + datafiles[0]).read())
    N = sum(czyN)
    run_file(datafiles[0])

    print("PROGRAM END")
