import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

1. #compare get_star_point_analytical with get_star_point_brute_force in a separate script 2. journal club! (define problem, algorithm, over view of all journals, summary, similar and difference)


# function to get a star point given a suburb location and alpha
def get_star_point_analytical(suburb_x, suburb_y, a):
    if a == 0:
        return 0.0
    
    numerator = np.abs(suburb_x) * np.sqrt(-a * ((a - 1) ** 2) * (a - 2))
    denominator = -(a ** 2) + (2 * a)
    f_s = suburb_y - (numerator / denominator)

    # check if f_s > 0 and returns the right values
    return np.maximum(0.0, f_s)

# function to get a star point using brute force method (pythag)
def get_star_point_brute_force(suburb_x, suburb_y, a, main_length):
    suburb = np.array([suburb_x, suburb_y])
    best_cost = float("inf")
    best_star_y = 0

    for y in np.linspace(0, main_length, main_length * 100):
        star = np.array([0, y])
        length = np.linalg.norm(suburb - star)
        time = abs(y) + length
        cost = a * length + (1-a) * time
        if cost < best_cost:
            best_cost = cost
            best_star_y = y

    return best_star_y
    
# function to compare the effectiveness of analytical vs brute
def compare(df, a, main_length):
    bf_y = []
    an_y = []
    for x, y in zip(df["suburb_x"], df["suburb_y"]):
        bf_y.append(float(get_star_point_brute_force(x, y, a, main_length)))
        an_y.append(float(get_star_point_analytical(x, y, a)))

    return bf_y, an_y

# function to calculate first derivative of C*'(L) at given mainline length L
def compute_c_prime(L, df, a):
    c_prime_sum = a
    for _, row in df.iterrows():
        f_star = row["star_y"]
        # only suburbs where f*(s) > L contribute to the derivative
        if f_star > L:
            sub_x = row["suburb_x"]
            sub_y = row["suburb_y"]
            denom = np.sqrt(sub_x**2 + (L - sub_y)**2) # denominator
            if denom > 0:
                c_prime_s = (L - sub_y) / denom + (1-a) 
            else:
                c_prime_s = (1-a)
            c_prime_sum += c_prime_s

    return c_prime_sum

def find_best_L_analytical(df, a, epsilon=1e-6):
    # absolute maximum upper bound based on furthest suburb
    max_search_bound = float(df["suburb_y"].max())

    # intervals bounded by f*(s) points because at those points, the cost is different because not all of the suburbs connect to the end of mainline, but go to their star point
    critical_points = [0.0]
    for f_star in df["star_y"]:
        if 0 < f_star < max_search_bound:
            critical_points.append(f_star)

    critical_points = sorted(list(set(critical_points)))

    # baseline scenario
    best_L = 0.0 # Lm = 0.0
    # gets the cost for baseline
    _, best_cost = get_optimal_connection(df, best_L, a) # MIN = C*(0)

    # checks boundary points b/c derivative of cost function changes dy
    for pt in critical_points:
        _, cost = get_optimal_connection(df, pt, a)
        if cost < best_cost:
            best_cost = cost
            best_L = pt

    # binary search within each continuous interval
    for i in range(len(critical_points)-1):
        LO = critical_points[i]
        HI = critical_points[i+1]

        # if signs are identical, no root exists in this interval
        if compute_c_prime(LO, df, a) * compute_c_prime(HI, df, a) > 0:
            continue
        while (HI-LO) > epsilon:
            MID = (LO + HI)/2.0
            deriv = compute_c_prime(MID, df, a)
            if deriv < 0:
                LO = MID
            else:
                HI = MID

        candidate_L = (LO + HI) / 2.0
        _, cost = get_optimal_connection(df, candidate_L, a) #C*(LO+HI/2)
        if cost < best_cost: # (MIN)
            best_cost = cost
            best_L = candidate_L

    return best_L, best_cost  

# function that takes all suburbs, all star points, and a main line length and outputs the optimal connection points
def get_optimal_connection(df, main_length, a): # C*(L)
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

def draw_networks(df, a, best_L, main_length, graph_num = 6):
    
    plot_candidates = np.linspace(0, main_length, graph_num)

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

bf_y, an_y = compare(df, a, main_length)
print("Brute force star points:", bf_y)
print("Analytical star points:", an_y)

star_y_list = [get_star_point_analytical(x, y, a) for x, y in zip(df["suburb_x"], df["suburb_y"])]

# save best star points back into the DataFrame
df["star_x"] = 0.0
df["star_y"] = star_y_list

# sort, so that star points in order
df = df.sort_values(by = "star_y").reset_index(drop=True)
print(df)

best_L, min_cost = find_best_L_analytical(df, a)
print(f"\n[Calculus Search] Best Mainline Length: {best_L:.6f}")
print(f"[Calculus Search] Minimum Overall Network Cost: {min_cost:.6f}")

draw_networks(df, a, best_L, main_length)


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

print(f"\n[Brute Force] Best Mainline Length: {best_L:.6f}")
print(f"[Brute Force] Minimum Overall Network Cost: {best_overall_cost:.6f}")


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

