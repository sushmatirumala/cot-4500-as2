import numpy as np


# Question 1

def nevilles_method(x_points, y_points, x):
    matrix = np.zeros((3, 3))
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    num_of_points = 3
    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            first_multiplication = (x - x_points[i - j]) * matrix[i][j - 1]
            second_multiplication = (x - x_points[i]) * matrix[i - 1][j - 1]
            denominator = x_points[i] - x_points[i - j]
            matrix[i][j] = (first_multiplication - second_multiplication) / denominator
    
    print(matrix[2, 2])
    print()

nevilles_method([3.6, 3.8, 3.9], [1.675, 1.436, 1.318], 3.7)


# Question 2

def divided_difference_table(x_points, y_points):
    size: int = np.size(x_points)
    matrix: np.array = np.zeros((size, size))
    for index, row in enumerate(matrix):
        row[0] =  y_points[index]
    for i in range(1, size):
        for j in range(1, size):
            numerator = matrix[i][j - 1] - matrix[i - 1][j - 1]
            denominator = x_points[i] - x_points[i - j]
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix

x_points = [7.2, 7.4, 7.5, 7.6]
y_points = [23.5492, 25.3913, 26.8224, 27.4589]
divided_table = divided_difference_table(x_points, y_points)

print([divided_table[1][1], divided_table[2][2], divided_table[3][3]])
print()


# Question 3

def get_approximate_result(matrix, x_points, value):
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    for index in range(1, np.size(x_points)):
        polynomial_coefficient = matrix[index][index]
        reoccuring_x_span *= (value - x_points[index - 1])
        mult_operation = polynomial_coefficient * reoccuring_x_span
        reoccuring_px_result += mult_operation
    return reoccuring_px_result

approximating_x = 7.3
final_approximation = get_approximate_result(divided_table, x_points, approximating_x)

print(final_approximation)
print()


# Question 4

def apply_div_dif(matrix):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            if (j >= len(matrix[i])) or matrix[i][j] != 0:
                continue
            left: float = matrix[i][j-1]
            diagonal_left: float = matrix[i-1][j-1]
            numerator: float = left - diagonal_left
            denominator = matrix[i][0] - matrix[i-j+1][0]
            operation = numerator / denominator
            matrix[i][j] = operation
    return matrix

def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
   
    num_of_points = len(x_points)
    matrix = np.zeros((2*num_of_points, 2*num_of_points))

    for x in range(2 * num_of_points - 1):
        matrix[x][0] = x_points[int(x/2)]
        matrix[x + 1][0] = x_points[int(x/2)]
        x += 1

    for x in range(2 * num_of_points - 1):
        matrix[x][1] = y_points[int(x/2)]
        matrix[x + 1][1] = y_points[int(x/2)]
        x += 1

    for x in range(num_of_points):
        matrix[(x*2) + 1][2] = slopes[int(x)]

    matrix = apply_div_dif(matrix)
    return matrix

np.set_printoptions(precision=7, suppress=True, linewidth=100)
print(hermite_interpolation())
print()


# Question 5

def cubic_spline(x, y):
    n = 3
    h = np.zeros(n)
    for i in range(0, n):
        h[i] = x_points[i+1] - x_points[i]

    A = np.zeros((n+1, n+1))
    A[0][0] = 1
    A[n][n] = 1
    for i in range(1, n):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]

    print(A)
    print()

    b = np.zeros(n+1)
    for i in range(1, n):
        b[i] = ((3.0/h[i]) * (y_points[i+1]-y_points[i])) - ((3.0/h[i-1]) * (y_points[i]-y_points[i-1]))

    print(b)
    print()

    x = np.linalg.solve(A, b)

    print(x)
    print()

x_points = [2, 5, 8, 10]
y_points = [3, 5, 7, 9]

cubic_spline(x_points, y_points)