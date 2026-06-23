# minimize cost = a * length + (1-a) * time to downtown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""functions for:
- function that takes all suburbs, all star points, and a main line length and outputs the optimal connection points
- a function that can take the necessary inputs to produce a drawing"""

# variables

# function to get a star point given a suburb location and alpha ()
def get_star_point(suburb_x, suburb_y, a):
    if a == 0:
        return 0.0
    
    numerator = np.abs(suburb_x) * np.sqrt(-a * ((a - 1) ** 2) * (a - 2))
    denominator = -(a ** 2) + (2 * a)
    f_s = suburb_y - (numerator / denominator)

    # check if f_s > 0 and returns the right values
    return np.maximum(0.0, f_s)



a = 0.5
main_length = 5

downtown = np.array([0, 0])

df = pd.DataFrame(
    {
        "suburb_x": [1, 2, -1],
        "suburb_y": [2, 5, 3],
    }
)

star_x_list = []
star_y_list = []

for idx, row in df.iterrows():
    star_x_list.append(0.0) # change if downtown changes
    star_y_list.append(get_star_point(row["suburb_x"], row["suburb_y"], a))
    
    
# save best star points back into the DataFrame
df["star_x"] = star_x_list
df["star_y"] = star_y_list

# sort, so that star points in order
df = df.sort_values(by = "star_y").reset_index(drop=True)

print(df)



# check multiple points along mainline
mainline_candidates = np.linspace(0, main_length, 6)


""# set up the multi-plot grid structure
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()  # flatten into 1D array to match 0-5 loop index ""


# track to find the best network
best_overall_cost = float("inf")
best_L = 0
best_ax_idx = 0

for i, L in enumerate(mainline_candidates):
    current_network_cost = a * L
    ax = axes[i]
    ax.grid(True, linestyle="--", alpha=0.5)

    # unchanging components
    ax.scatter(0, 0, color="red", s=150, marker="*", zorder=5)  # downtown
    ax.plot([0, 0], [0, L], color="black", linewidth=4, zorder=2)  # mainline
    ax.plot([0, 0], [L, main_length], color="black", linewidth=2, linestyle=":", alpha=0.3)  # possible mainline

    for sub_x, sub_y, optimal_star_y in zip(df["suburb_x"], df["suburb_y"], df["star_y"]):
        suburb = np.array([sub_x, sub_y])

        # Evaluate layout behavior based on current L
        if optimal_star_y <= L: # if suburb's star is less than or equal to the mainline length, anchors (connects to mainline at star)
            actual_connect_y = optimal_star_y
        else: # if suburb's star is higher, it pulls down to top of mainline (connects to end of mainline)
            actual_connect_y = L

        length = np.linalg.norm(suburb - np.array([0, actual_connect_y]))
        time = actual_connect_y + length
        current_network_cost += a * length + (1 - a) * time

        # plot suburb node and its link straight to the axis
        ax.scatter(sub_x, sub_y, color="blue", s=80, edgecolors="black", zorder=4)
        ax.plot([sub_x, 0], [sub_y, actual_connect_y], color="orange", linewidth=2, linestyle="--", zorder=3)
        ax.scatter(0, actual_connect_y, color="orange", s=40, zorder=4)
        ax.scatter(0, optimal_star_y, color="limegreen", s=100, marker="*", edgecolors="darkgreen", zorder=4, label="Ideal Star Point" if i==0 else "")

    # set temporary title with the live calculated cost
    ax.set_title(f"Network {i+1}: L = {L:.1f}\nCost: {current_network_cost:.2f}", fontsize=10)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, main_length + 0.5)
    if i == 0:
        ax.legend(loc="upper left", fontsize=8)

    # keep track of which network performs the best
    if current_network_cost < best_overall_cost:
        best_overall_cost = current_network_cost
        best_L = L
        best_ax_idx = i

# change title to show which one is the best
axes[best_ax_idx].set_title(
    axes[best_ax_idx].get_title() + " (BEST)", 
    weight="bold", 
    color="darkgreen"
)

print(f"Best Mainline Length: {best_L:.1f}")
print(f"Minimum Overall Network Cost: {best_overall_cost:.4f}")

plt.suptitle("All 6 Network Layouts", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig("network_layout.png", dpi=300)
plt.show()






# checking for best network among many many networks
mainline_candidates = np.linspace(0, main_length, main_length * 100)

# track to find the best network
best_overall_cost = float("inf")
best_L = 0
for L in mainline_candidates:
    current_network_cost = a * L

    for sub_x, sub_y, optimal_star_y in zip(df["suburb_x"], df["suburb_y"], df["star_y"]):
        suburb = np.array([sub_x, sub_y])

        # Evaluate layout behavior based on current L
        if optimal_star_y <= L: # if suburb's star is less than or equal to the mainline length, anchors (connects to mainline at star)
            actual_connect_y = optimal_star_y
        else: # if suburb's star is higher, it pulls down to top of mainline (connects to end of mainline)
            actual_connect_y = L

        length = np.linalg.norm(suburb - np.array([0, actual_connect_y]))
        time = actual_connect_y + length
        current_network_cost += a * length + (1 - a) * time

    # keep track of which network performs the best
    if current_network_cost < best_overall_cost:
        best_overall_cost = current_network_cost
        best_L = L

print(f"Best Mainline Length: {best_L:.1f}")
print(f"Minimum Overall Network Cost: {best_overall_cost:.4f}")

