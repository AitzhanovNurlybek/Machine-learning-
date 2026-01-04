




import numpy as np
import matplotlib.pyplot as plt
import os

def thomas(a, b, c, d):
    
    n = len(b)
    if n == 0:
        return np.array([])
    cp = np.zeros(n-1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    denom = b[-1] - a[-1] * cp[-1]
    dp[-1] = (d[-1] - a[-1] * dp[-1-1]) / denom
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
        
    return x

def solve_burgers(Re, L=4.0, Nx=201, dt=5e-4, t_max=1.0, snapshots=[0.01,0.1,0.5,1.0]):
    #u_t+uu_x=1/Reu_xx
    x = np.linspace(0.0, L, Nx)
    dx = x[1] - x[0]
    Nt = int(np.ceil(t_max / dt))
    # переведём моменты снимков в номера шагов
    snap_steps = {int(np.round(s / dt)): s for s in snapshots}
    # начальные условия
    u = np.zeros(Nx)
    u[0] = 1.0  # Dirichlet на левом конце в любое время
    results = {}
    # параметр r = dt/(Re*dx^2)
    r = (dt / (Re * dx * dx))
    for n in range(1, Nt + 1):
        un = u.copy()
        # явная аппроксимация  (u * u_x) at time n
        conv = np.zeros_like(u)
        conv[1:] = un[1:] * (un[1:] - un[:-1]) / dx
        # собираем трёхдиагональную систему для внутренних узлов 
        M = Nx - 2
        if M <= 0:
            break
        a = np.zeros(M-1)  
        b = np.zeros(M)    
        c = np.zeros(M-1) 
        d = np.zeros(M)   
        for j in range(1, Nx-1):
            idx = j - 1
            b[idx] = 1.0 + 2.0 * r
            if idx > 0:
                a[idx-1] = -r
            if idx < M-1:
                c[idx] = -r
            d[idx] = un[j] - dt * conv[j]
        # вклад от левого Дирихле:
        d[0] += r * 1.0
        b[-1] = 1.0 + r
        if M - 1 > 0:
            c[-1] = 0.0
        # решаем
        u_interior = thomas(a, b, c, d)
        u[1:-1] = u_interior
        # применяем BC
        u[0] = 1.0
        u[-1] = u[-2]  # Neumann
        # сохраняем если попадаем в снимок
        if n in snap_steps:
            t = round(n * dt, 6)
            results[t] = u.copy()
    return x, results

def plot_results(all_results, outdir="plots"):
  
    os.makedirs(outdir, exist_ok=True)
    # по одному рисунку на Re
    for Re, (x, res) in all_results.items():
        plt.figure(figsize=(9,4.5))
        times_sorted = sorted(res.keys())
        for t in times_sorted:
            plt.plot(x, res[t], label=f"t={t}")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"Burgers' solution (semi-implicit) — Re={Re}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        fname = os.path.join(outdir, f"burgers_Re{Re}.png")
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print("Saved", fname)
        plt.show()
    # сравнение всех Re на последнем доступном времени
    plt.figure(figsize=(9,4.5))
    for Re, (x, res) in all_results.items():
        if len(res) == 0:
            continue
        t_avail = sorted(res.keys())[-1]
        plt.plot(x, res[t_avail], label=f"Re={Re}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title("Comparison at t ~ last snapshot")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    fname = os.path.join(outdir, "burgers_compare.png")
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    print("Saved", fname)
    plt.show()

def main():
    Nx = 201        # число узлов по x
    dt = 5e-4       # шаг по времени
    t_max = 1.0
    snapshots = [0.01, 0.1, 0.5, 1.0]
    Res = [1, 50, 1000]
    all_results = {}
    for Re in Res:
        print("Running Re =", Re)
        x, res = solve_burgers(Re, L=4.0, Nx=Nx, dt=dt, t_max=t_max, snapshots=snapshots)
        all_results[Re] = (x, res)
    plot_results(all_results, outdir="plots")

if __name__ == "__main__":
    main()
