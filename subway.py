# minimize cost = a * length + (1-a) * time to downtown
import numpy as np
# variables
a = 1 # changable
downtown = np.array([0, 0])
suburbs = np.array([
    [1, 2],
    [2, 5],
    [-1, 3],
])
main_length = 5

for suburb in suburbs:
    best_cost = float("inf")
    best_star = None

    for y in np.linspace(0, main_length, main_length * 100):
        star = np.array([0, y])
        length = np.linalg.norm(suburb - star)
        time = abs(y) + length
        cost = a * length + (1-a) * time
        if cost < best_cost:
            best_cost = cost
            best_star = star

    print("Best star point:", best_star)
    print("Minimum cost", best_cost)





