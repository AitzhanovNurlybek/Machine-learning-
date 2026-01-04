import numpy as np
import matplotlib.pyplot as plt

# Параметрическое уравнение сердца
t = np.linspace(0, 2 * np.pi, 300)
x = 16 * np.sin(t)**3
y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

# Нормировка
x = x / np.max(np.abs(x))
y = y / np.max(np.abs(y))

# Текст
text_full = "Диана"

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
ax.set_aspect('equal')
ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.6, 1.2)
ax.axis('off')

# Пустой scatter для сердца
scat = ax.scatter([], [], s=20, c='red')

# Текст, который будет постепенно появляться
text_obj = ax.text(0, -1.4, "", ha='center', va='center',
                   fontsize=28, color='red', fontweight='bold')

plt.show()

# 1. Появление сердца
for i in range(1, len(t) + 1):
    coords = np.column_stack((x[:i], y[:i]))
    scat.set_offsets(coords)
    fig.canvas.draw()
    plt.pause(0.02)

# 2. Появление текста "Диана" — по одной букве
current = ""
for letter in text_full:
    current += letter
    text_obj.set_text(current)
    fig.canvas.draw()
    plt.pause(0.3)  # скорость появления букв

plt.ioff()
plt.show()
