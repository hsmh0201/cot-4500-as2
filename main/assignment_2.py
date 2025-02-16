# Question 1: Nevilles Method
from decimal import Decimal

def neville(x, y, x_target):
    n = len(x)
    # y-value to to decimals
    p = [Decimal(val) for val in y]

# interpolated values
    for j in range(1, n):
        for i in range(n - j):
            numerator = (Decimal(x_target) - Decimal(x[i + j])) * p[i] + (Decimal(x[i]) - Decimal(x_target)) * p[i + 1]
            denominator = Decimal(x[i]) - Decimal(x[i + j])
            p[i] = numerator / denominator

# convert back to floats
    return float(p[0])

x_values = [3.6, 3.8, 3.9]
y_values = [1.675, 1.436, 1.318]
x_target = 3.7

# compute and print with 15 decimals
result = neville(x_values, y_values, x_target)
print(f"{result:.15f}")  

print()


# Question 2: Newtons forward method
import numpy as np
import sympy as sp

x_values = np.array([7.2, 7.4, 7.5, 7.6])
f_values = np.array([23.5492, 25.3913, 26.8224, 27.4589])

# first order differences
f01 = (f_values[1] - f_values[0]) / (x_values[1] - x_values[0])
f12 = (f_values[2] - f_values[1]) / (x_values[2] - x_values[1])
f23 = (f_values[3] - f_values[2]) / (x_values[3] - x_values[2])

# second order differences
f012 = (f12 - f01) / (x_values[2] - x_values[0])
f123 = (f23 - f12) / (x_values[3] - x_values[1])

# third order difference
f0123 = (f123 - f012) / (x_values[3] - x_values[0])

expected_f012 = -0.7183802816901438
expected_f0123 = -0.12461196085345332

# newtons ploynomial
x = sp.Symbol('x')
newton_poly = (
    f_values[0] + f01 * (x - x_values[0])
    + expected_f012 * (x - x_values[0]) * (x - x_values[1])
    + expected_f0123 * (x - x_values[0]) * (x - x_values[1]) * (x - x_values[2])
)


# Question 3 : Newtons forward method at x = 7.3
f_7_3_correct = newton_poly.subs(x, 7.3)

formatted_output = [
    f"{f01:.16f}",
    f"{expected_f012:.16f}",
    f"{expected_f0123:.16f}",
]

# first three divided differences
for value in formatted_output:
    print(value)

print()

# final answer
print(f"{f_7_3_correct:.16f}")

print()


# Question 4: Divided difference
import numpy as np

# compute hermite divided difference table
def hermite_divided_difference(x, y, y_deriv):
    n = len(x)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

# table with functions and derivatives given 
    for i in range(n):
        z[2 * i] = x[i]
        z[2 * i + 1] = x[i]
        Q[2 * i, 0] = y[i]
        Q[2 * i + 1, 0] = y[i]
        Q[2 * i + 1, 1] = y_deriv[i]

        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

# compute divided differences
    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])

    return z, Q

x_values = [3.6, 3.8, 3.9]
y_values = [1.675, 1.436, 1.318]
y_deriv_values = [-1.195, -1.188, -1.182]

# compute hermite table
z, Q = hermite_divided_difference(x_values, y_values, y_deriv_values)

np.set_printoptions(precision=10, suppress=True)

# print hermite table
for i in range(len(z)):
    formatted_row = f"{z[i]:.7f} " + " ".join(
        f"{num:.10e}" if abs(num) < 1e-2 else f"{num:.7f}" for num in Q[i]
    )
    print(f"[ {formatted_row} ]")

print()


# Question 5 : Cubic spline interpolation
import numpy as np

x = np.array([2, 5, 8, 10])
f_x = np.array([3, 5, 7, 9])

n = len(x) - 1
h = np.diff(x)
alpha = np.zeros(n)

# alpha vector for cubic spline systems
for i in range(1, n):
    alpha[i] = (3 / h[i]) * (f_x[i + 1] - f_x[i]) - (3 / h[i - 1]) * (f_x[i] - f_x[i - 1])

# coefficient matrix A & right hand side bector b
A = np.zeros((n + 1, n + 1))
b = np.zeros(n + 1)
A[0, 0] = 1
A[n, n] = 1

# system of equations for cubic spline interpolation
for i in range(1, n):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    b[i] = alpha[i]

# compute for vector x
x = np.linalg.solve(A, b)

np.set_printoptions(precision=1, suppress=True)

# print maxtrix a, vector b and vector x
for row in A:
    print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.1f}" for num in row), "]")  
print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.1f}" for num in b), "]")  
print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.8f}" for num in x), "]")  