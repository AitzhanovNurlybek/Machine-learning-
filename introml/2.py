import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded
import re

class PDESolver:
    """Решатель для эллиптических, параболических и гиперболических УЧП"""
    
    def __init__(self):
        self.sym_sign = {}
        self.num_re = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
        self.term_re = re.compile(r'^([+-]?\d*(?:\.\d+)?)\*?([A-Za-z]\w*)$')
    
    def classify_2d(self, A, B, C):
        """Классификация УЧП второго порядка"""
        D = B*B - 4*A*C
        if np.isclose(D, 0.0):
            return "PARABOLIC", D
        elif D < 0:
            return "ELLIPTIC", D
        else:
            return "HYPERBOLIC", D
    
    def ask_sign(self, sym):
        """Запрос знака символической переменной"""
        if sym in self.sym_sign:
            return self.sym_sign[sym]
        
        while True:
            s = input(f"Знак переменной {sym} (>/<=/): ").strip()
            if s in (">", "+"):
                v = 1.0
            elif s in ("<", "-"):
                v = -1.0
            elif s in ("=", "0"):
                v = 0.0
            else:
                continue
            self.sym_sign[sym] = v
            return v
    
    def read_coef(self, name):
        """Чтение коэффициента с поддержкой символических переменных"""
        s = input(f"{name}: ").strip().replace(",", "")
        if self.num_re.fullmatch(s):
            return float(s)
        
        m = self.term_re.fullmatch(s.replace(" ", ""))
        if m:
            coef_str, sym = m.groups()
            coef = 1.0 if coef_str in ("", "+") else (-1.0 if coef_str == "-" else float(coef_str))
            sgn = self.ask_sign(sym)
            return coef * (1.0 if sgn > 0 else -1.0 if sgn < 0 else 0.0)
        raise ValueError("Неверный ввод")
    
    def solve_elliptic(self, nx=50, ny=50, Lx=1.0, Ly=1.0):
        """
        Решение эллиптического уравнения (уравнение Лапласа/Пуассона)
        ∇²u = f(x,y) с граничными условиями Дирихле
        """
        print("\nРешение эллиптического уравнения (Лапласа/Пуассона)")
        print("∇²u = f(x,y)")
        
        dx = Lx / (nx - 1)
        dy = Ly / (ny - 1)
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        X, Y = np.meshgrid(x, y)
        
        # Источниковый член
        f_type = input("Тип источника (0-нет, 1-константа, 2-sin): ").strip()
        if f_type == "1":
            f_val = float(input("Значение f: "))
            f = np.ones((ny, nx)) * f_val
        elif f_type == "2":
            f = np.sin(np.pi * X) * np.sin(np.pi * Y)
        else:
            f = np.zeros((ny, nx))
        
        # Граничные условия
        u = np.zeros((ny, nx))
        print("Граничные условия:")
        u[0, :] = float(input("u(x, 0) = "))  # нижняя граница
        u[-1, :] = float(input("u(x, Ly) = "))  # верхняя граница
        u[:, 0] = float(input("u(0, y) = "))  # левая граница
        u[:, -1] = float(input("u(Lx, y) = "))  # правая граница
        
        # Итерационное решение методом Гаусса-Зейделя
        max_iter = int(input("Макс. итераций (по умолчанию 1000): ") or "1000")
        tol = float(input("Точность (по умолчанию 1e-5): ") or "1e-5")
        
        for iteration in range(max_iter):
            u_old = u.copy()
            
            for i in range(1, ny-1):
                for j in range(1, nx-1):
                    u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + 
                                      u[i, j+1] + u[i, j-1] - 
                                      dx*dx * f[i, j])
            
            # Проверка сходимости
            if np.max(np.abs(u - u_old)) < tol:
                print(f"Сходимость достигнута за {iteration+1} итераций")
                break
        
        # Визуализация
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, u, cmap='viridis')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('u(x,y)')
        ax.set_title('Решение эллиптического уравнения')
        plt.show()
        
        return x, y, u
    
    def solve_parabolic(self, nx=50, nt=100, L=1.0, T=0.1, alpha=0.01):
        """
        Решение параболического уравнения (уравнение теплопроводности)
        ∂u/∂t = α∇²u
        """
        print("\nРешение параболического уравнения (теплопроводности)")
        print("∂u/∂t = α∇²u")
        
        dx = L / (nx - 1)
        dt = T / nt
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T, nt)
        
        # Коэффициент диффузии
        alpha = float(input(f"Коэффициент диффузии α (по умолчанию {alpha}): ") or str(alpha))
        
        # Проверка устойчивости
        r = alpha * dt / (dx * dx)
        if r > 0.5:
            print(f"Предупреждение: схема может быть неустойчива (r={r:.3f} > 0.5)")
            
        # Начальное условие
        print("Начальное условие u(x,0):")
        init_type = input("Тип (1-гаусс, 2-ступенька, 3-sin): ").strip()
        u = np.zeros((nt, nx))
        
        if init_type == "1":
            u[0, :] = np.exp(-50 * (x - 0.5)**2)
        elif init_type == "2":
            u[0, nx//4:3*nx//4] = 1.0
        else:
            u[0, :] = np.sin(np.pi * x)
        
        # Граничные условия
        bc_type = input("Граничные условия (1-Дирихле, 2-Неймана): ").strip()
        if bc_type == "1":
            u[:, 0] = float(input("u(0, t) = "))
            u[:, -1] = float(input("u(L, t) = "))
        
        # Решение явной схемой
        for n in range(nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = u[n, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            # Граничные условия Неймана
            if bc_type == "2":
                u[n+1, 0] = u[n+1, 1]
                u[n+1, -1] = u[n+1, -2]
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # График в разные моменты времени
        for i in range(0, nt, nt//5):
            ax1.plot(x, u[i, :], label=f't={t[i]:.3f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x,t)')
        ax1.set_title('Эволюция решения')
        ax1.legend()
        ax1.grid(True)
        
        # Контурный график
        T_grid, X_grid = np.meshgrid(t, x)
        cs = ax2.contourf(T_grid, X_grid, u.T, levels=20, cmap='hot')
        ax2.set_xlabel('Время t')
        ax2.set_ylabel('Координата x')
        ax2.set_title('Пространственно-временная эволюция')
        plt.colorbar(cs, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        return x, t, u
    
    def solve_hyperbolic(self, nx=100, nt=200, L=1.0, T=1.0, c=1.0):
        """
        Решение гиперболического уравнения (волновое уравнение)
        ∂²u/∂t² = c²∇²u
        """
        print("\nРешение гиперболического уравнения (волнового)")
        print("∂²u/∂t² = c²∇²u")
        
        dx = L / (nx - 1)
        dt = T / nt
        x = np.linspace(0, L, nx)
        t = np.linspace(0, T, nt)
        
        # Скорость волны
        c = float(input(f"Скорость волны c (по умолчанию {c}): ") or str(c))
        
        # Число Куранта
        cfl = c * dt / dx
        print(f"Число Куранта CFL = {cfl:.3f}")
        if cfl > 1:
            print("Предупреждение: схема может быть неустойчива (CFL > 1)")
        
        # Инициализация
        u = np.zeros((nt, nx))
        
        # Начальные условия
        print("Начальное смещение u(x,0):")
        init_type = input("Тип (1-гаусс, 2-sin, 3-треугольник): ").strip()
        
        if init_type == "1":
            u[0, :] = np.exp(-100 * (x - 0.5)**2)
        elif init_type == "2":
            u[0, :] = np.sin(2 * np.pi * x)
        else:
            u[0, :] = np.where(np.abs(x - 0.5) < 0.1, 1 - 10*np.abs(x - 0.5), 0)
        
        # Начальная скорость
        vel_type = input("Начальная скорость (0-нет, 1-есть): ").strip()
        if vel_type == "0":
            u[1, :] = u[0, :]  # ∂u/∂t = 0
        else:
            u[1, :] = u[0, :] + 0.1 * np.sin(np.pi * x) * dt
        
        # Граничные условия
        bc_type = input("Граничные условия (1-фиксированные, 2-свободные): ").strip()
        
        # Решение конечно-разностной схемой
        r = (c * dt / dx) ** 2
        
        for n in range(1, nt-1):
            for i in range(1, nx-1):
                u[n+1, i] = 2*u[n, i] - u[n-1, i] + r * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
            
            # Граничные условия
            if bc_type == "1":
                u[n+1, 0] = 0
                u[n+1, -1] = 0
            else:  # свободные концы
                u[n+1, 0] = u[n+1, 1]
                u[n+1, -1] = u[n+1, -2]
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Снимки в разные моменты времени
        for i in range(0, nt, nt//6):
            ax1.plot(x, u[i, :], label=f't={t[i]:.2f}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x,t)')
        ax1.set_title('Распространение волны')
        ax1.legend()
        ax1.grid(True)
        
        # Пространственно-временная диаграмма
        T_grid, X_grid = np.meshgrid(t, x)
        cs = ax2.contourf(T_grid, X_grid, u.T, levels=20, cmap='seismic')
        ax2.set_xlabel('Время t')
        ax2.set_ylabel('Координата x')
        ax2.set_title('Пространственно-временная эволюция')
        plt.colorbar(cs, ax=ax2)
        
        plt.tight_layout()
        plt.show()
        
        return x, t, u

def main():
    solver = PDESolver()
    
    while True:
        print("\n" + "="*50)
        print("РЕШАТЕЛЬ ДИФФЕРЕНЦИАЛЬНЫХ УРАВНЕНИЙ В ЧАСТНЫХ ПРОИЗВОДНЫХ")
        print("="*50)
        print("1. Классификация УЧП")
        print("2. Решение эллиптического уравнения")
        print("3. Решение параболического уравнения")
        print("4. Решение гиперболического уравнения")
        print("5. Автоматическое решение по коэффициентам")
        print("0. Выход")
        
        choice = input("\nВыберите опцию: ").strip()
        
        if choice == "0":
            break
            
        elif choice == "1":
            print("\nКлассификация УЧП вида: A*uxx + B*uxy + C*uyy + ... = 0")
            try:
                A = solver.read_coef("A (коэфф. при uxx)")
                B = solver.read_coef("B (коэфф. при uxy)")
                C = solver.read_coef("C (коэфф. при uyy)")
                
                pde_type, D = solver.classify_2d(A, B, C)
                print(f"\nДискриминант D = B² - 4AC = {D:.4f}")
                print(f"Тип уравнения: {pde_type}")
                
                if pde_type == "ELLIPTIC":
                    print("Это эллиптическое уравнение (как уравнение Лапласа)")
                elif pde_type == "PARABOLIC":
                    print("Это параболическое уравнение (как уравнение теплопроводности)")
                else:
                    print("Это гиперболическое уравнение (как волновое уравнение)")
                    
            except Exception as e:
                print(f"Ошибка: {e}")
                
        elif choice == "2":
            solver.solve_elliptic()
            
        elif choice == "3":
            solver.solve_parabolic()
            
        elif choice == "4":
            solver.solve_hyperbolic()
            
        elif choice == "5":
            print("\nАвтоматическое решение на основе классификации")
            try:
                A = solver.read_coef("A (коэфф. при uxx)")
                B = solver.read_coef("B (коэфф. при uxy)")
                C = solver.read_coef("C (коэфф. при uyy)")
                
                pde_type, D = solver.classify_2d(A, B, C)
                print(f"\nОпределен тип: {pde_type}")
                print("Запуск соответствующего решателя...")
                
                if pde_type == "ELLIPTIC":
                    solver.solve_elliptic()
                elif pde_type == "PARABOLIC":
                    solver.solve_parabolic()
                else:  # HYPERBOLIC
                    solver.solve_hyperbolic()
                    
            except Exception as e:
                print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()