import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""functions for:
- function that takes all suburbs, all star points, and a main line length and outputs the optimal connection points
- a function that can take the necessary inputs to produce a drawing"""


# function to get a star point given a suburb location and alpha
def get_star_point(suburb_x, suburb_y, a):
    if a == 0:
        return 0.0
    
    numerator = np.abs(suburb_x) * np.sqrt(-a * ((a - 1) ** 2) * (a - 2))
    denominator = -(a ** 2) + (2 * a)
    f_s = suburb_y - (numerator / denominator)

    # check if f_s > 0 and returns the right values
    return np.maximum(0.0, f_s)

# function that takes all suburbs, all star points, and a main line length and outputs the optimal connection points
def get_optimal_connection(df, main_length, a):
    # f_star_y is ideal star point with no main_length
    total_network_cost = a * main_length
    actual_connections = []
    for sub_x, sub_y, optimal_star_y in zip(df["suburb_x"], df["suburb_y"], df["star_y"]):
        suburb = np.array([sub_x, sub_y])

        # Evaluate layout behavior based on current L
        if optimal_star_y <= main_length: # if suburb's star is less than or equal to the mainline length, anchors (connects to mainline at star)
            actual_connect_y = optimal_star_y
        else: # if suburb's star is higher, it pulls down to top of mainline (connects to end of mainline)
            actual_connect_y = float(main_length)

        actual_connections.append(actual_connect_y)

        length = np.linalg.norm(suburb - np.array([0, actual_connect_y]))
        time = actual_connect_y + length
        total_network_cost += a * length + (1 - a) * time

    return actual_connections, total_network_cost

def draw_networks(df, a, best_L, main_length):
    
    plot_candidates = np.linspace(0, main_length, 6)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # flatten into 1D array to match 0-5 loop index ""

    for i, L in enumerate(plot_candidates):
        ax = axes[i]
        ax.grid(True, linestyle="--", alpha=0.5)

        actual_connections, display_cost = get_optimal_connection(df, L, a)

        ax.scatter(0, 0, color="red", s=150, marker="*", zorder=5, label="Downtown")  # downtown
        ax.plot([0, 0], [0, L], color="black", linewidth=4, zorder=2, label="Mainline")  # mainline
        ax.plot([0, 0], [L, main_length], color="black", linewidth=2, linestyle=":", alpha=0.3)  # possible mainline

        for (sub_x, sub_y, optimal_star_y), actual_connect_y in zip(zip(df["suburb_x"], df["suburb_y"], df["star_y"]), actual_connections):
            # suburb node
            ax.scatter(sub_x, sub_y, color="blue", s=80, edgecolors="black", zorder=4)
            # link
            ax.plot([sub_x, 0], [sub_y, actual_connect_y], color="orange", linewidth=2, linestyle="--", zorder=3,)
            # conection point
            ax.scatter(0, actual_connect_y, color="orange", s=40, zorder=4)
            # ideal star points
            ax.scatter( 0, optimal_star_y,  color="limegreen", s=100,  marker="*",  edgecolors="darkgreen", zorder=4,label="Ideal Star Point" if i == 0 else "",
            )

         # Tag visual winner based on structural match parameters
        is_optimal = " (OPTIMAL CHOICE)" if np.isclose(L, best_L, atol=0.51) else ""

        ax.set_title(f"Network {i+1}: L = {L:.1f}{is_optimal}\nCost: {display_cost:.2f}", fontsize=10, weight="bold" if is_optimal else "normal", color="darkgreen" if is_optimal else "black",)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, main_length + 0.5)

        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Visualizing 6 Representative Samples", fontsize=14, weight="bold",)
    plt.tight_layout()
    plt.savefig("network_layout.png", dpi=300)
    plt.show()









a = 0.5
main_length = 5

downtown = np.array([0, 0])

df = pd.DataFrame(
    {
        "suburb_x": [1, 2, -1],
        "suburb_y": [2, 5, 3],
    }
)

star_y_list = []

for x, y in zip(df["suburb_x"], df["suburb_y"]):
    star_y_list.append(get_star_point(x, y, a))
    
    
# save best star points back into the DataFrame
df["star_x"] = 0.0
df["star_y"] = star_y_list

# sort, so that star points in order
df = df.sort_values(by = "star_y").reset_index(drop=True)

print(df)

# checking for best network among many many networks
mainline_candidates = np.linspace(0, main_length, main_length * 100)

# track to find the best network
best_overall_cost = float("inf")
best_L = None
best_df = None

for L in mainline_candidates:
    actual_connections, total_network_cost = get_optimal_connection(df, L, a)
    
    if total_network_cost < best_overall_cost:
        best_overall_cost = total_network_cost
        best_L = L
        best_df = actual_connections

print(f"Best Mainline Length: {best_L:.1f}")
print(f"Minimum Overall Network Cost: {best_overall_cost:.4f}")
print(best_df)
draw_networks(df, a, best_L, main_length)


"""
# check multiple points along mainline
mainline_candidates = np.linspace(0, main_length, 6)


""# set up the multi-plot grid structure
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()  # flatten into 1D array to match 0-5 loop index ""


# track to find the best network
best_overall_cost = float("inf")
best_L = 0
best_ax_idx = 0

for L in mainline_candidates:
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


"""

