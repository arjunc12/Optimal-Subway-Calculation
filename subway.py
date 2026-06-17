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
best_stars = np.zeros((len(suburbs), 2))


for i, suburb in enumerate(suburbs):
    best_cost = float("inf")
    best_star = None

    for y in np.linspace(0, main_length, main_length * 100):
        star = np.array([0, y])
        length = np.linalg.norm(suburb - star)
        time = abs(y) + length # should this be main_length
        cost = a * length + (1-a) * time
        if cost < best_cost:
            best_cost = cost
            best_star = star

    print("Best star point:", best_star)
    print("Minimum cost", best_cost)
    best_stars[i] = best_star

sort_indices = np.argsort(best_stars[:, 1])
suburbs = suburbs[sort_indices]
best_stars = best_stars[sort_indices]


# track the cost to find the best network
best_overall_cost = float("inf")
best_L = 0
best_network_index = -1

for k in range (len(suburbs)):
    L = best_stars[k][1] # mainline length to this star point

    current_network_cost = a * L

    # if suburb's star is less than or equal to the mainline length, anchors
    if i <= k:
        star_y = best_stars[i][1]
        length = np.linalg.norm(suburb - np.array([0, star_y]))
        time = star_y + length
        current_network_cost += a * length + (1-a) * time
    # if suburb's star is higher, it pulls down to top of mainline
    else:
        length = np.linalg.norm(suburb - np.array([0, L]))
        current_network_cost += a * length + (1-a) * time

    print ("Mainline Length:", L)
    print ("Network Cost:", current_network_cost)

    if current_network_cost < best_overall_cost:
        best_overall_cost = current_network_cost
        best_L = L
        best_layout_index = k

print("Optimal Mainline Length: ", best_L)
print(f"Minimum Overall Network Cost: ", best_overall_cost)
print(f"Optimal Layout Number: ", best_layout_index + 1)


    


