# optimization.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import numpy as np
import itertools
from itertools import combinations
import math
from math import e
import queue

# import pacmanPlot
# import graphicsUtils
import util

# You may add any helper functions you would like here:
# def somethingUseful():
#     return True

best_z = math.inf

def is_integer_solution(linear_solutions):
    for i in range(0, len(linear_solutions)):
        value = linear_solutions[i]     # Get value if X{i}
        nearest_int = round(value)      # Get nearest int
        high = nearest_int + 0.000000000001     # upper range
        low = nearest_int - 0.000000000001      # lower range

        if value > high or value < low:
            # call recursive?
            return False

    return True


def make_new_constraints(linear_solutions, constraints):

    left_constraints = constraints.copy()
    right_constraints = constraints.copy()

    decimal_dict = {}
    for i in range(0, len(linear_solutions)):
        decimal_value = linear_solutions[i] % 1
        decimal_dict[i] = decimal_value

    #print("DICT: ")
    #for key, value in decimal_dict.items():
        #print(key, value)

    max_value = max(decimal_dict.values())
    max_index = 0
    for key, value in decimal_dict.items():
        high = value + 0.000000000001
        low = value - 0.000000000001
        if max_value <= high and max_value >= low:
            max_index = key

    #print(max_index)

    #print(max_value)

    constraint_value = linear_solutions[max_index]
    floor = math.floor(constraint_value)
    ceil = math.ceil(constraint_value)

    floor_bound = [0] * len(linear_solutions)
    floor_bound[max_index] = 1
    floor_tuple = (floor_bound, floor)
    left_constraints.append(floor_tuple)

    ceil_bound = [0] * len(linear_solutions)
    ceil_bound[max_index] = -1
    ceil_tuple = (ceil_bound, -ceil)
    right_constraints.append(ceil_tuple)
    '''
    for i in range(0, len(linear_solutions)):
        value = linear_solutions[i]
        nearest_int = round(value)
        high = nearest_int + 0.000000000001
        low = nearest_int - 0.000000000001

        # Check if Xi is an not an integer
        if value > high or value < low:
            # call recursive?
            floor = math.floor(value)
            ceil = math.ceil(value)
            neg_ceil = -1 * ceil
            floor_list = []
            ceil_list = []
            for j in range(0, len(linear_solutions)):
                if j == i:
                    print(value)
                    floor_list.append(1)
                    ceil_list.append(-1)
                else:
                    floor_list.append(0)
                    ceil_list.append(0)

            floor_tup = (floor_list, floor)
            left_constraints.append(floor_tup)
            ceil_tup = (ceil_list, neg_ceil)
            right_constraints.append(ceil_tup)
            break
    '''
    #for row in left_constraints:
        #print(row)

    #for row in right_constraints:
       #print(row)
    return left_constraints, right_constraints


def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    "*** YOUR CODE HERE ***"

    a_matrix = []
    b_vector = []
    x_vector = []
    for a_list, b in constraints:

        # Item is the A constraint values
        a_vector = []
        for item in a_list:
            a_vector.append(item)
        a_matrix.append(a_vector)

        b_vector.append(b)

    l = list(range(0, len(b_vector)))
    comb = combinations(l, len(a_matrix[0]))
    for row_comb in list(comb):
        a = []
        b = []
        for index in row_comb:
            a.append(a_matrix[index])
            b.append(b_vector[index])
        rank_a = np.linalg.matrix_rank(a)
        rank_a = np.linalg.matrix_rank(a)

        if rank_a == len(a):
            x = np.linalg.solve(a,b)
            x_vector.append(x)

    return x_vector

    util.raiseNotDefined()

def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    "*** YOUR CODE HERE ***"
    intersections = findIntersections(constraints)

    a_matrix = []
    b_vector = []
    x_vector = []
    valid_x = []
    for a_list, b in constraints:

        # Item is the A constraint values
        a_vector = []
        for item in a_list:
            a_vector.append(item)
        a_matrix.append(a_vector)

        b_vector.append(b)

    #print(a_matrix)

    #print(intersections[1])
    for x in intersections:
        b_value = np.dot(a_matrix, x)

        valid_int = True
        for i in range(0, len(b_value)):
            if b_value[i] > (b_vector[i] + 0.000000000001):
                valid_int = False

        if valid_int:
            valid_x.append(x)

    #print(valid_x)
    return valid_x

    util.raiseNotDefined()

def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    "*** YOUR CODE HERE ***"
    feasible_solutions = findFeasibleIntersections(constraints)
    #print(feasible_solutions)
    cost_dict = {}
    #print(cost)

    for x in feasible_solutions:
        c = np.dot(x, cost)
        cost_dict[c] = x

    if cost_dict:
        min_sol = min(cost_dict.keys())
        #print(min_sol)
        min_x = cost_dict[min_sol]
        #print(min_x)
        result = (min_x, min_sol)
       # print("Min Solution: ", min_sol)
        return result
    else:
        return False

    util.raiseNotDefined()

def wordProblemLP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    constraint = [ ((-1, -0), -20),
                   ((-0, -1), -15.5),
                   ((2.5, 2.5), 100),
                   ((0.5, 0.25), 50)]
    cost = (-7, -4)

    solution = solveLP(constraint, cost)
    x, c = solution
    return (x, -1*c)

    util.raiseNotDefined()


def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    interger values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the 
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.

    """
    "*** YOUR CODE HERE ***"
    solutions = solveLP(constraints, cost)
    #print(solutions)
    if not solutions:
        return None

    # Check first if is an integer solution
    linear_solutions = solutions[0]
    integer_solutions = is_integer_solution(linear_solutions)

    # If integer solution, then return it
    if integer_solutions:
        return solutions

    # If not integer solution
    elif not integer_solutions:
        # build new constraint matrix
        left_constraints, right_constraints = make_new_constraints(linear_solutions, constraints)

        search_queue = util.Queue()
        global best_z
        incumbent = None
        search_queue.push(left_constraints)
        search_queue.push(right_constraints)

        while not search_queue.isEmpty():
            problem = search_queue.pop()

            lp_sol = solveLP(problem, cost)
           #print(lp_sol)
            if lp_sol is False:
                continue
            elif lp_sol[1] > best_z:
                continue

            ip_sol = solveIP(problem, cost)

            if ip_sol is False:
                continue

            if ip_sol:
                z = ip_sol[1]
                if z <= best_z:
                    best_z = z
                    incumbent = ip_sol[0]

        if incumbent is None:
            return None

        return (incumbent, best_z)


        #left = solveIP(left_constraints, cost)

        #if left:
         #   return left

        #right = solveIP(right_constraints, cost)
        #if right:
         #   return right

    util.raiseNotDefined()

def wordProblemIP():
    """
    Formulate the work problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    # need constraints
    # need cost

    constraint = [  ((-1, -0, -1, -0, -1, -0), -15),
                    ((-0, -1, -0, -1, -0, -1), -30),
                    ((1.2, 0, 0, 0, 0, 0), 30),
                    ((0, 1.2, 0, 0, 0, 0), 30),
                    ((0, 0, 1.3, 0, 0, 0), 30),
                    ((0, 0, 0, 1.3, 0, 0), 30),
                    ((0, 0, 0, 0, 1.1, 0), 30),
                    ((0, 0, 0, 0, 0, 1.1), 30),
                    ((-1, 0, 0, 0, 0, 0), 0),
                    ((0, -1, 0, 0, 0, 0), 0),
                    ((0, 0, -1, 0, 0, 0), 0),
                    ((0, 0, 0, -1, 0, 0), 0),
                    ((0, 0, 0, 0, -1, 0), 0),
                    ((0, 0, 0, 0, 0, -1), 0)]

    cost = (12, 20, 4, 5, 2, 1)

    global best_z
    best_z = math.inf
    solution = solveIP(constraint, cost)

    return solution

    util.raiseNotDefined()

def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each 
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the 
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding objective
            value at that point: (optimial_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """

    M = len(W)      # M length of food weights per unit, or number of providers
    N = len(C)

    cost = []
    for row in T:
        for item in row:
            cost.append(item)

    constraints = []

    for i in range(0, len(W)):
        num_entries = int(len(cost)/M)
        for j in range(0, num_entries):
            index = 3*i + j
            listofzeros = [0] * len(cost)
            listofzeros[index] = W[i]
            newConstraintTuple = (listofzeros, truck_limit)
            constraints.append(newConstraintTuple)

    for i in range(0, len(cost)):
        listofzeros = [0] * len(cost)
        listofzeros[i] = -1
        newConstraintTuple = (listofzeros, 0)
        constraints.append(newConstraintTuple)

    for i in range(0, N):
        listofzeros = [0] * len(cost)
        for j in range(i, len(cost), N):
            listofzeros[j] = -1
        newConstraintTuple = (listofzeros, -C[i])
        constraints.append(newConstraintTuple)

    solution = solveIP(constraints, cost)
    return solution

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


if __name__ == "__main__":
    constraints = [((3, 2), 10),((1, -9), 8),((-3, 2), 40),((-3, -1), 20)]
    inter = findIntersections(constraints)
    print(inter)
    print()
    valid = findFeasibleIntersections(constraints)
    print(valid)
    print()
    print(solveLP(constraints, (3,5)))
    print()
    print(solveIP(constraints, (3,5)))
    print()
    print(wordProblemIP())
