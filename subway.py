# minimize cost = a * length + (1-a) * time to downtown
import numpy as np
# variables
a = 0.1 # changable
downtown = np.array([0, 0])
suburb = np.array([1, 2])
best_cost = float("inf")
best_star = None

for y in np.linspace(0, 2, 500):
    star = np.array([0, y])
    length = np.linalg.norm(suburb - star)
    time = abs(y) + length
    cost = a * length + (1-a) * time
    if cost < best_cost:
        best_cost = cost
        best_star = star

print("Best star point:", best_star)
print("Minimum cost", best_cost)





