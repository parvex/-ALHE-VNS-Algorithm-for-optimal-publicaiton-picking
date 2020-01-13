import numpy as np
import random
import copy
import math
import os

# datafile variables
A = 0
N0 = 0
N1 = 0
N2 = 0
P = 0
udzial = []
doktorant = []
pracownik = []
czyN = []
u = []
w = []
monografia = []
authorIdList = []
publicationIdList = []
##########################

N = sum(czyN)
fix_function = False


def fix_point(point):
    fixed_point = copy.deepcopy(point)
    for i in range(A):
        if author_cost(fixed_point[i], i) > 0:
            author_publications = [(j, w[i][j] / u[i][j]) for j in range(P)
                                   if u[i][j] != 0 and fixed_point[i][j] != 0]
            author_publications.sort(key=lambda x: x[1], reverse=False)
            fixed_point[i][author_publications[0][0]] = 0
            k = 1
            while author_cost(fixed_point[i], i) > 0:
                fixed_point[i][author_publications[k][0]] = 0
                k += 1

    if university_cost(fixed_point) > 0:
        publications = [((i, j), w[i][j] / u[i][j]) for i in range(A) for j in range(P)
                        if u[i][j] != 0 and fixed_point[i][j] != 0]
        publications.sort(key=lambda x: x[1], reverse=False)
        fixed_point[publications[0][0][0]][publications[0][0][1]] = 0
        k = 1
        while university_cost(fixed_point) > 0:
            fixed_point[publications[k][0][0]][publications[k][0][1]] = 0
            k += 1

    return fixed_point


def author_cost(author_publications, author_index):
    author_cost = 0
    for j in range(P):
        author_cost += author_publications[j] * u[author_index][j]
    author_cost -= 4 * udzial[author_index]
    return author_cost

def university_cost(point):
    university_cost = 0
    for i in range(A):
        for j in range(P):
            university_cost += point[i][j] * u[i][j]

    university_cost -= 3 * N

    return university_cost

class Solution:
    def __init__(self, point):
        self.point = point
        if fix_function:
            self.fixed_point = fix_point(point)
            self.value = value_function(self.fixed_point)
        else:
            self.value = cost_value_function(point)


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


def value_function(solution_matrix):
    value = 0
    for i in range(A):
        for j in range(P):
            value += solution_matrix[i][j] * w[i][j]

    return value

def cost_value_function(solution_matrix, k=250):
    value = 0
    university_cost = 0
    for i in range(A):
        author_cost = 0
        for j in range(P):
            value += solution_matrix[i][j] * w[i][j]
            author_cost += solution_matrix[i][j] * u[i][j]
            university_cost += solution_matrix[i][j] * u[i][j]
        author_cost -= 4 * udzial[i]

        if author_cost > 0:
            value -= k * (1 + author_cost)

    university_cost -= 3 * N
    if university_cost > 0:
        value -= k * (1 + university_cost)

    return value


def randomly_change_n_positions(chosen_publications, n):
    new_point = copy.deepcopy(chosen_publications.point)
    for k in range(n):
        i = random.randint(0, A-1)
        j = random.randint(0, P-1)
        new_point[i][j] = 1 if new_point[i][j] == 0 else 1

    return Solution(new_point)


def variable_neighborhood_search(init_solution, search_proportion, max_neighborhood_radius):
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
                file.write("Stage: " + str(stage) + " * bits >> best_value:" + str(best_solution.value) + "\n")
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


# Run script

datafiles = os.listdir("data")

for datafile in datafiles:
    neighborhood_param = 1/10
    max_radius = 20

    file = open("results/" + datafile + ".txt", "w+")
    # change dartafiles[2] to datafile - for testing purpose only
    exec(open("data/" + datafile.read()))
    N = sum(czyN)
    print("File - " + datafile)
    file.write("File - " + datafile + "\n")
    print("Cost function")
    file.write("---Cost function---\n")
    file.flush()
    points = gen_starting_points()
    for i, point in enumerate(points):
        print("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius))
        file.write("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius) + "\n")
        file.flush()
        variable_neighborhood_search(point, neighborhood_param, max_radius)

    #calculation for fix function
    fix_function = True
    print("File - " + datafiles[2])
    file.write("File - " + datafile + "\n")
    print("Fix function")
    file.write("---Fix function---\n")
    file.flush()
    points = gen_starting_points()
    for i, point in enumerate(points):
        print("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius))
        file.write("Calculating point: " + str(i) + " neighborhood param: "
              + str(neighborhood_param) + " max radius: " + str(max_radius) + "\n")
        file.flush()
        variable_neighborhood_search(point, neighborhood_param, max_radius)

    file.close()

file.close()
print("Algorithm end")




