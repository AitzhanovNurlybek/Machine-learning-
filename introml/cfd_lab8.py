import numpy as np
import matplotlib.pyplot as plt

# =======================
# Параметры сетки
# =======================
Nx = 100
Ny = 200
dx = 1 / Nx
dy = 2 / Ny

eps = 1e-5
max_iter = 7000
omega = 1.7  # параметр SOR

X = [i * dx for i in range(Nx)]
Y = [j * dy for j in range(Ny)]

# =======================
# Граничные условия
# =======================
def apply_bc(U):
    for i in range(Nx):
        x = i * dx
        if 0 <= x <= 0.5:
            U[0, i] = 1
    U[:, 0] = 0
    U[:, Nx - 1] = 0
    return U


# =======================
# Метод Якоби
# =======================
def solve_jacobi():
    U = np.zeros((Ny, Nx))
    U = apply_bc(U)
    iteration = 0

    while True:
        U_new = U.copy()
        error = 0

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                U_new[i, j] = 0.25 * (
                    U[i + 1, j] + U[i - 1, j] +
                    U[i, j + 1] + U[i, j - 1]
                )
                error = max(error, abs(U_new[i, j] - U[i, j]))

        U = apply_bc(U_new.copy())
        iteration += 1

        if iteration % 800 == 0:
            plt.figure(figsize=(6, 10))
            plt.contourf(X, Y, U, cmap='viridis')
            plt.title(f"Jacobi — iteration = {iteration}")
            plt.xlabel("x"); plt.ylabel("y"); plt.colorbar(label="u")
            plt.show()

        if error < eps or iteration >= max_iter:
            break

    return U, iteration, error


# =======================
# Метод Гаусса–Зейделя
# =======================
def solve_gauss_seidel():
    U = np.zeros((Ny, Nx))
    U = apply_bc(U)
    iteration = 0

    while True:
        error = 0

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                U_old = U[i, j]
                U[i, j] = 0.25 * (
                    U[i + 1, j] + U[i - 1, j] +
                    U[i, j + 1] + U[i, j - 1]
                )
                error = max(error, abs(U[i, j] - U_old))

        U = apply_bc(U)
        iteration += 1

        if iteration % 800 == 0:
            plt.figure(figsize=(6, 10))
            plt.contourf(X, Y, U, cmap='viridis')
            plt.title(f"Gauss–Seidel — iteration = {iteration}")
            plt.xlabel("x"); plt.ylabel("y"); plt.colorbar(label="u")
            plt.show()

        if error < eps or iteration >= max_iter:
            break

    return U, iteration, error


# =======================
# Метод SOR (Relaxation)
# =======================
def solve_sor():
    U = np.zeros((Ny, Nx))
    U = apply_bc(U)
    iteration = 0

    while True:
        error = 0

        for i in range(1, Ny - 1):
            for j in range(1, Nx - 1):
                U_old = U[i, j]

                U_star = 0.25 * (
                    U[i + 1, j] + U[i - 1, j] +
                    U[i, j + 1] + U[i, j - 1]
                )

                U[i, j] = (1 - omega) * U_old + omega * U_star
                error = max(error, abs(U[i, j] - U_old))

        U = apply_bc(U)
        iteration += 1

        if iteration % 400 == 0:
            plt.figure(figsize=(6, 10))
            plt.contourf(X, Y, U, cmap='viridis')
            plt.title(f"SOR (ω={omega}) — iteration = {iteration}")
            plt.xlabel("x"); plt.ylabel("y"); plt.colorbar(label="u")
            plt.show()

        if error < eps or iteration >= max_iter:
            break

    return U, iteration, error


# =======================
# Запуск всех методов
# =======================
U_jacobi, it_jacobi, err_jacobi = solve_jacobi()
print(f"Jacobi: Converged in {it_jacobi} iterations (error = {err_jacobi:.2e})")

U_gs, it_gs, err_gs = solve_gauss_seidel()
print(f"Gauss–Seidel: Converged in {it_gs} iterations (error = {err_gs:.2e})")

U_sor, it_sor, err_sor = solve_sor()
print(f"SOR: Converged in {it_sor} iterations (error = {err_sor:.2e})")


# =======================
# Итоговое сравнение
# =======================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, U_jacobi, cmap='viridis')
plt.title(f"Jacobi\nIterations = {it_jacobi}")
plt.colorbar(label="u")
plt.subplot(1, 3, 2)
plt.contourf(X, Y, U_gs, cmap='viridis')
plt.title(f"Gauss–Seidel\nIterations = {it_gs}")
plt.colorbar(label="u")

plt.subplot(1, 3, 3)
plt.contourf(X, Y, U_sor, cmap='viridis')
plt.title(f"SOR (ω={omega})\nIterations = {it_sor}")
plt.colorbar(label="u")

plt.tight_layout()
plt.show()