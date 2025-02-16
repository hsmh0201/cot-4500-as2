from decimal import Decimal
def neville(x, y, x_target):
    n = len(x)
    p = [Decimal(val) for val in y]

    for j in range(1, n):
        for i in range(n - j):
            numerator = (Decimal(x_target) - Decimal(x[i + j])) * p[i] + (Decimal(x[i]) - Decimal(x_target)) * p[i + 1]
            denominator = Decimal(x[i]) - Decimal(x[i + j])
            p[i] = numerator / denominator

    return float(p[0])

x_values = [3.6, 3.8, 3.9]
y_values = [1.675, 1.436, 1.318]
x_target = 3.7

result = neville(x_values, y_values, x_target)
print(f"{result:.15f}")