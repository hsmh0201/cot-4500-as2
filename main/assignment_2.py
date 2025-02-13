#Question 1
def neville(x, y, x_target):
    n = len(x)
    p = y.copy()

    for j in range(1, n):
        for i in range(n - j):
            p[i] = ((x_target - x[i + j]) * p[i] + (x[i] - x_target) * p[i + 1]) / (x[i] - x[i + j])

    return p[0]

x_values = [3.6, 3.8, 3.9]
y_values = [1.675, 1.436, 1.318]
x_target = 3.7

result = neville(x_values, y_values, x_target)
print(f"{x_target}: {results:.6f}")