import numpy as np
import re

def c2d(A, B, C):
    D = B*B - 4*A*C
    t = "PARABOLIC" if np.isclose(D, 0.0) else ("ELLIPTIC" if D < 0 else "HYPERBOLIC")
    return t, D

def cnd(A):
    A = np.array(A, float)
    A = 0.5 * (A + A.T)  # symmetrize
    e = np.linalg.eigvalsh(A)  # symmetric eigenvalues
    z = np.sum(np.isclose(e, 0.0))
    p = np.sum(e > 0)
    n = np.sum(e < 0)
    m = len(e)
    if z > 0:
        t = "PARABOLIC"
    elif p == m or n == m:
        t = "ELLIPTIC"
    elif (p == m-1 and n == 1) or (n == m-1 and p == 1):
        t = "HYPERBOLIC"
    else:
        t = "INDEFINITE"
    return t, e

sym_sign = {}
num_re = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
term_re = re.compile(r'^([+-]?\d*(?:\.\d+)?)\*?([A-Za-z]\w*)$')

def ask_sign(sym):
    if sym in sym_sign:
        return sym_sign[sym]
    while True:
        s = input(f"sign({sym}) (>,<,=): ").strip()
        if s in (">", "+"):
            v = 1.0
        elif s in ("<", "-"):
            v = -1.0
        elif s in ("=", "0"):
            v = 0.0
        else:
            continue
        sym_sign[sym] = v
        return v

def read_coef(name):
    s = input(f"{name}: ").strip().replace(",", "")
    if num_re.fullmatch(s):
        return float(s)
    m = term_re.fullmatch(s.replace(" ", ""))
    if m:
        coef_str, sym = m.groups()
        coef = 1.0 if coef_str in ("", "+") else (-1.0 if coef_str == "-" else float(coef_str))
        sgn = ask_sign(sym)
        return coef * (1.0 if sgn > 0 else -1.0 if sgn < 0 else 0.0)
    raise ValueError("bad input")

def main():
    while True:
        mode = input("mode (2d/nd/stop): ").strip().lower()
        if mode == "stop":
            break
        if mode == "2d":
            try:
                A = read_coef("A")
                B = read_coef("B")
                C = read_coef("C")
            except Exception:
                print("err")
                continue
            t, D = c2d(A, B, C)
            print(f"D={D} -> {t}")
        elif mode == "nd":
            n = int(input("n: "))
            rows = [list(map(float, input(f"row{i+1}: ").split())) for i in range(n)]
            t, e = cnd(rows)
            print("eig=", e, "->", t)

if __name__ == "__main__":
    main()
