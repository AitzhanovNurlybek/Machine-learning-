import numpy as np
import re

def classify_2d(a, b, c):
    disc = b**2 - 4*a*c
    if np.isclose(disc, 0.0):
        kind = "Parabolic"
    elif disc < 0:
        kind = "Elliptic"
    else:
        kind = "Hyperbolic"
    return kind, disc

def classify_nd(matrix):
    arr = np.array(matrix, float)
    arr = 0.5 * (arr + arr.T)   # symmetrization
    eigs = np.linalg.eigvalsh(arr)

    zeros = np.sum(np.isclose(eigs, 0.0))
    pos = np.sum(eigs > 0)
    neg = np.sum(eigs < 0)
    n = len(eigs)

    if zeros > 0:
        kind = "Parabolic"
    elif pos == n or neg == n:
        kind = "Elliptic"
    elif (pos == n-1 and neg == 1) or (neg == n-1 and pos == 1):
        kind = "Hyperbolic"
    else:
        kind = "Indefinite"
    return kind, eigs

# regex patterns
sign_cache = {}
num_pattern = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
term_pattern = re.compile(r'^([+-]?\d*(?:\.\d+)?)\*?([A-Za-z]\w*)$')

def ask_for_sign(var):
    if var in sign_cache:
        return sign_cache[var]
    while True:
        choice = input(f"sign of {var} (+, -, 0): ").strip()
        if choice == "+":
            val = 1.0
        elif choice == "-":
            val = -1.0
        elif choice == "0":
            val = 0.0
        else:
            continue
        sign_cache[var] = val
        return val

def sanitize(expr: str) -> str:
    return re.sub(r'[^0-9A-Za-z+\-.*]', '', expr)

def read_value(label):
    expr = input(f"{label} = ").strip()
    expr = sanitize(expr)

    if num_pattern.fullmatch(expr):
        return float(expr)

    match = term_pattern.fullmatch(expr)
    if match:
        coef_str, sym = match.groups()
        coef = 1.0 if coef_str in ("", "+") else (-1.0 if coef_str == "-" else float(coef_str))
        sgn = ask_for_sign(sym)
        return coef * sgn
    raise ValueError("Invalid input")

def main():
    while True:
        choice = input("choose (2/n/exit): ").strip().lower()
        if choice == "exit":
            break

        elif choice == "2":
            try:
                a = read_value("A")
                b = read_value("B")
                c = read_value("C")
            except Exception:
                print("invalid input")
                continue
            kind, disc = classify_2d(a, b, c)
            print(f"Discriminant = {disc} → {kind}")

        elif choice == "n":
            size = int(input("matrix size n = "))
            print("Enter rows separated by ';', numbers by space")
            raw = input("matrix: ").strip()
            rows = [list(map(float, row.split())) for row in raw.split(";")]
            kind, eigs = classify_nd(rows)
            print("Eigenvalues:", eigs, "→", kind)

if __name__ == "__main__":
    main()
