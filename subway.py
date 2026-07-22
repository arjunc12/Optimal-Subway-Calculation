import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import time

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
            if denom > 0 or denom < 0: 
                c_prime_s = (L - sub_y) / denom + (1-a) 
            else: # if denomenator is 0, should it just be (1-a)
                c_prime_s = (1-a)
            c_prime_sum += c_prime_s

    return c_prime_sum

# function to find the best main_length using binary search
def find_best_L_analytical(df, a, epsilon=1e-6):
    # absolute maximum upper bound based on furthest suburb
    max_search_bound = float(df["suburb_y"].max())

    # intervals bounded by f*(s) points because at those points, the cost is different because not all of the suburbs connect to the end of mainline, but go to their star point
    critical_points = [0.0]
    for f_star in df["star_y"]:
        if 0 < f_star < max_search_bound:
            critical_points.append(f_star)
    critical_points.append(max_search_bound)
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
    
    double_star_points = []

    for i in range(len(critical_points) - 1):
        LO = critical_points[i]
        HI = critical_points[i+1]

        # 1. Establish baseline tracking for the current interval boundaries
        _, cost_lo = get_optimal_connection(df, LO, a)
        _, cost_hi = get_optimal_connection(df, HI, a)
        
        if cost_lo < cost_hi:
            interval_best_L = LO
            interval_best_cost = cost_lo
        else:
            interval_best_L = HI
            interval_best_cost = cost_hi

        # 2. If the derivative crosses zero, search for a deeper valley inside the interval
        if compute_c_prime(LO, df, a) * compute_c_prime(HI, df, a) <= 0:
            while (HI - LO) > epsilon:
                MID = (LO + HI) / 2.0
                deriv = compute_c_prime(MID, df, a)
                if deriv < 0:
                    LO = MID
                else:
                    HI = MID

            candidate_L = (LO + HI) / 2.0
            _, candidate_cost = get_optimal_connection(df, candidate_L, a)
            
            # If the middle valley beats the boundary edges, update the interval winner
            if candidate_cost < interval_best_cost:
                interval_best_L = candidate_L
                interval_best_cost = candidate_cost

        # 3. Append the final calculated Double Star Point for this specific interval
        double_star_points.append((interval_best_L, interval_best_cost))

        # 4. Check if this interval's best point beats the overall Global Minimum (MIN)
        if interval_best_cost < best_cost:
            best_cost = interval_best_cost
            best_L = interval_best_L

    # Return the global targets as well as the array containing all interval best points
    return best_L, best_cost, double_star_points

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

def draw_steps(df, a, colors = ["green", "orange", "purple", "#ff7f0e", "#17becf"]):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    suburbs = df.copy()
    suburbs["star_y"] = [get_star_point_analytical(x, y, a) for x, y in zip(suburbs["suburb_x"], suburbs["suburb_y"])]
    suburbs = suburbs.sort_values(by="star_y").reset_index(drop=True)

    star_1 = suburbs["star_y"].iloc[0]
    star_2 = suburbs["star_y"].iloc[1]
    star_3 = suburbs["star_y"].iloc[2]

    panel_titles = [
        "1. Start with downtown and suburbs",
        "2. Find a suburb's star point",
        "3. Identify all ideal star points",
        "4. Try everything up to first star point",
        "5. Try up to second star point",
        "6. Repeat for all star points"
    ]

    for i in range(6):
        ax = axes[i]
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, main_length + 0.5)

        # downtown at (0, 0)
        ax.scatter(0, 0, color="black", s=300, zorder=6, label = "Downtown")

        # panel 1: just downtown and suburbs
        if i == 0:
            for idx, row in suburbs.iterrows():
                sub_color = colors[idx % len(colors)]
                ax.scatter(row["suburb_x"], row["suburb_y"], color=sub_color, s=90, edgecolors="black", zorder=4, label = "Suburb " + str(idx + 1))
                ax.legend(loc="lower left", fontsize=8, markerscale=0.7, labelspacing=1)

        # panel 2: finding one star point
        elif i == 1:
            ax.plot([0, 0], [0, float(suburbs["suburb_y"].max())], color="black", linewidth=3, zorder=2)

            target_idx = len(suburbs)-1
            target_sub = suburbs.iloc[target_idx]
            sub_color = colors[target_idx % len(colors)]

            # draws line, suburb, and star
            ax.scatter(target_sub["suburb_x"], target_sub["suburb_y"], color=sub_color, s=90, edgecolors="black", zorder=4)
            ax.scatter(0, target_sub["star_y"], color=sub_color, marker="*", s=120, edgecolors="black", zorder=4, label = "Ideal Star Point")
            ax.plot([target_sub["suburb_x"], 0], [target_sub["suburb_y"], target_sub["star_y"]], color=sub_color, linestyle="-", zorder=3)

            # draws checked lines
            label = "Trials"
            for check_y in np.linspace(0, float(suburbs["suburb_y"].max()), 5):
                if not np.isclose(check_y, target_sub["star_y"]):
                    ax.plot([target_sub["suburb_x"], 0], [target_sub["suburb_y"], check_y], color=sub_color, linestyle="--", alpha=0.4, zorder=3, label=label)
                label = None
            ax.legend(loc="lower left", fontsize=8, markerscale=0.7, labelspacing = 1)   

        # panel 3: show all suburbs with their star points
        elif i == 2:
            ax.plot([0, 0], [0, float(suburbs["suburb_y"].max())], color="black", linewidth=3, zorder=2)
            for idx, row in suburbs.iterrows():
                sub_color = colors[idx % len(colors)]
                ax.scatter(row["suburb_x"], row["suburb_y"], color=sub_color, s=90, edgecolors="black", zorder=4)
                ax.scatter(0, row["star_y"], color=sub_color, marker="*", s=120, edgecolors="black", zorder=4)
                ax.plot([row["suburb_x"], 0], [row["suburb_y"], row["star_y"]], color=sub_color, linewidth=1.5, alpha=0.6)
            

        else:
            # panel 4: up to first star
            if i == 3:
                trial_lengths = np.linspace(0.1, star_1, 4)
            # panel 4: up to second star
            elif i == 4:
                trial_lengths = np.linspace(star_1, star_2, 4)
            # panel 5: up to third star
            else:
                trial_lengths = np.linspace(star_2, star_3, 4)

            for trial_L in trial_lengths:
                ax.plot([0, 0], [0, trial_L], color="black", linewidth=3, alpha=0.4, zorder=2)
                trial_conns, _ = get_optimal_connection(suburbs, trial_L, a)
                ax.scatter(0, trial_L, color="black", marker="_", s=150, linewidths=2, alpha=0.6, zorder=3)

                for idx, row in suburbs.iterrows():
                    sub_color = colors[idx % len(colors)]
                    conn_y = trial_conns[idx]
                    line_style = "-" if np.isclose(conn_y, row["star_y"]) else "--"
                    
                    ax.plot([row["suburb_x"], 0], [row["suburb_y"], conn_y], 
                            color=sub_color, linestyle=line_style, linewidth=1.5, alpha=0.25, zorder=3)
                    
            final_L = trial_lengths[-1]
            ax.plot([0, 0], [0, final_L], color="black", linewidth=5, zorder=3)
            final_conns, _ = get_optimal_connection(suburbs, final_L, a)
            
            for idx, row in suburbs.iterrows():
                sub_color = colors[idx % len(colors)]
                conn_y = final_conns[idx]
                
                ax.scatter(row["suburb_x"], row["suburb_y"], color=sub_color, s=90, edgecolors="black", zorder=4)
                
                line_style = "-" if np.isclose(conn_y, row["star_y"]) else "--"
                
                ax.plot([row["suburb_x"], 0], [row["suburb_y"], conn_y], color=sub_color, linestyle=line_style, linewidth=2.5, zorder=4)
                ax.scatter(0, conn_y, color=sub_color, marker="*", s=120, edgecolors="black", zorder=4)

        ax.text(0.05, 0.93, panel_titles[i], transform=ax.transAxes, fontsize=11, weight='bold',
                bbox=dict(facecolor='#E8DAEF', edgecolor='none', pad=6, alpha=0.9))
        
        if i == 4:
            ax.text(0.6, 0.25, "First suburb is anchored\nto its Star point", transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor='#E8DAEF', edgecolor='none', pad=5, alpha=0.9))
            
        fig.text(0.5, 0.02, "Take the best of best of all networks!", transform=fig.transFigure, fontsize=13, weight='bold',
             color="black", ha="center", bbox=dict(facecolor='#FEDBB5', edgecolor='none', pad=8))
    plt.suptitle("Visualizing Algorithm Steps", fontsize=14, weight="bold",)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig("optimal_subway_algorithm_generated.png", dpi=300)
    plt.show()

# function that draws what the network looks like at different main_lengths
def draw_networks(df, a, best_L, main_length, graph_num = 6, colors = ["#1f77b4", "#9467bd", "#e377c2", "#ff7f0e", "#17becf"]):
    
    plot_candidates = np.linspace(0, main_length, graph_num)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()  # flatten into 1D array to match 0-5 loop index ""

    for i, L in enumerate(plot_candidates):
        ax = axes[i]
        ax.grid(True, linestyle="--", alpha=0.5)

        actual_connections, display_cost = get_optimal_connection(df, L, a)
        ax.scatter(0, 0, color="black", s=300, zorder=5, label="Downtown")  # downtown
        ax.plot([0, 0], [0, L], color="black", linewidth=4, zorder=2, label="Mainline")  # mainline
        ax.plot([0, 0], [L, main_length], color="black", linewidth=2, linestyle=":", alpha=0.3)  # possible mainline

        for idx, (row, actual_connect_y) in enumerate(zip(df.iterrows(), actual_connections)):
            _, data = row
            sub_x, sub_y, optimal_star_y = data["suburb_x"], data["suburb_y"], data["star_y"]
            sub_color = colors[idx % len(colors)]

            # suburb node (uses matching unique color)
            ax.scatter(sub_x, sub_y, color=sub_color, s=90, edgecolors="black", zorder=4)
            # link line to the axis
            ax.plot([sub_x, 0], [sub_y, actual_connect_y], color="orange", linewidth=2, linestyle="--", zorder=3)
            # connection drop-point along the mainline
            ax.scatter(0, actual_connect_y, color="orange", s=40, zorder=4)
            # ideal Star Point (Now matches the same unique color as its suburb node!)
            ax.scatter(0, optimal_star_y, color=sub_color, s=120, marker="*", edgecolors="black", zorder=4,
                       label="Ideal Star Point" if (i == 0 and idx == 0) else "")
            
         # tag visual winner based on structural match parameters
        is_optimal = " (OPTIMAL CHOICE)" if np.isclose(L, best_L, atol=0.51) else ""

        ax.set_title(f"Network {i+1}: L = {L:.1f}{is_optimal}\nCost: {display_cost:.2f}", fontsize=10, weight="bold" if is_optimal else "normal", color="darkgreen" if is_optimal else "black",)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-0.5, main_length + 0.5)

        if i == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=0.7,labelspacing=0.8)

    plt.suptitle("Visualizing 6 Representative Samples", fontsize=14, weight="bold",)
    plt.tight_layout()
    plt.savefig("network_layout.png", dpi=300)
    plt.show()

# function that plots graph of cost vs main_length, pointing out minimum cost, ploting star points and double star points
def plot_cost_optimization(df, a):
    # gets the star points and sorts them
    df["star_y"] = [
        get_star_point_analytical(x, y, a)
        for x, y in zip(df["suburb_x"], df["suburb_y"])
    ]
    df = df.sort_values(by="star_y").reset_index(drop=True)

    # determines the maximum main line length based on suburb
    max_plot_length = float(df["suburb_y"].max())

    # finding optimal length and lowest cost
    best_L_calc, min_cost_calc, double_star_points = find_best_L_analytical(df, a)

    # gets continuous points along main length and calculates costs
    L_spectrum = np.linspace(0, max_plot_length, 1000)
    costs = [get_optimal_connection(df, L, a)[1] for L in L_spectrum]

    # intialize plot
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    # plot the continuous cost curve
    plt.plot(
        L_spectrum,
        costs,
        color="#1f77b4",
        linewidth=2.5,
        label="Total Network Cost $C^*(L)$",
    )

    # highlight individual suburb "Star Points"
    for idx, row in df.iterrows():
        f_star = row["star_y"]
        _, f_star_cost = get_optimal_connection(df, f_star, a)
        plt.scatter(
            f_star,
            f_star_cost,
            color="orange",
            edgecolor="black",
            s=70,
            zorder=4,
            label="Individual Suburb Star Point" if idx == 0 else "",
        )

    for i, (interval_L, interval_cost) in enumerate(double_star_points):
        plt.scatter(
            interval_L,
            interval_cost,
            color="gold",
            marker="*",
            s=220,
            edgecolor="darkorange",
            zorder=4,
            label="Interval Best (Double Star)" if i == 0 else "",
        )

    # highlight the minimum found via Calculus + Binary Search
    plt.scatter(
        best_L_calc,
        min_cost_calc,
        color="red",
        marker="*",
        s=250,
        edgecolor="black",
        zorder=5,
        label=f"Calculus Min Cost ($L={best_L_calc:.4f}$)",
    )

    # minimum coordinate flag
    plt.annotate(
        f"Minimum Cost: {min_cost_calc:.4f}\nOptimal $L$: {best_L_calc:.4f}",
        xy=(best_L_calc, min_cost_calc),
        xytext=(best_L_calc + (max_plot_length * 0.06), min_cost_calc + 0.4),
        arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=6),
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
    )

    plt.title(
        "Network Optimization: Mainline Length ($L$) vs. Total Cost",
        fontsize=13,
        weight="bold",
    )
    plt.xlabel("Mainline Length ($L$)", fontsize=11)
    plt.ylabel("Total Network Cost", fontsize=11)
    plt.xlim(-0.1, max_plot_length + 0.2)
    plt.ylim(min(costs) - 0.5, max(costs) + 0.5)
    plt.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("cost_vs_length_optimization.png", dpi=300)
    plt.show()

# function that returns
def run_timing_comparison(df_base, a, main_length):

    # brute force star points
    start_bf = time.perf_counter()
    bf_stars = [
        get_star_point_brute_force(x, y, a, main_length)
        for x, y in zip(df_base["suburb_x"], df_base["suburb_y"])
    ]
    df_bf = df_base.copy()
    df_bf["star_y"] = bf_stars

    # brute force mainline search
    mainline_candidates = np.linspace(0, main_length, 100) # SHOULD IT BE SET STEPS
    best_cost_bf = float("inf")
    best_L_bf = 0.0
    for L in mainline_candidates:
        _, total_network_cost = get_optimal_connection(df_bf, L, a)
        if total_network_cost < best_cost_bf:
            best_cost_bf = total_network_cost
            best_L_bf = L

    end_bf = time.perf_counter()
    total_bf_time = end_bf - start_bf


    start_an = time.perf_counter()
    an_stars = [
        get_star_point_analytical(x, y, a)
        for x, y in zip(df_base["suburb_x"], df_base["suburb_y"])
    ]
    df_an = df_base.copy()
    df_an["star_y"] = an_stars
    df_an = df_an.sort_values(by="star_y").reset_index(drop=True)
    best_L_an, _, _ = find_best_L_analytical(df_an, a)
    end_an = time.perf_counter()
    total_an_time = end_an - start_an

    return {
        "total_bf_time": total_bf_time,
        "total_an_time": total_an_time,
        "best_L_bf": best_L_bf,
        "best_L_an": best_L_an,
        "df_an": df_an,
        "df_bf": df_bf,
    }


    

def analytical_vs_brute(min_suburbs = 5, max_suburbs = 100, step = 30, a = 0.5, main_length = 1000): # SHOULD MAIN_LENGTH BE SET OR JUST BE THE LARGEST STAR POINT
    suburb_counts = []
    analytical_times = []
    brute_force_times = []
    csv_records = []

    for num_suburbs in range(min_suburbs, max_suburbs + 1, step): # NUMBER OF SUBURBS ISN"T RANDOM
        # generate random coordinates
        np.random.seed(42 + num_suburbs)
        x_coords = np.random.uniform(-100, 100, num_suburbs)
        y_coords = np.random.uniform(0, 1000, num_suburbs)

        df_base = pd.DataFrame(
        {
            "suburb_x": x_coords,
            "suburb_y": y_coords,
        }
        )

        # gets the run time for a subway with num_suburbs amount of suburbs
        metrics = run_timing_comparison(df_base, a, main_length)

        suburb_counts.append(num_suburbs)
        brute_force_times.append(metrics["total_bf_time"])
        analytical_times.append(metrics["total_an_time"])
        df_an = metrics["df_an"]
        df_bf = metrics["df_bf"]

        for idx in range(num_suburbs):
            csv_records.append(
                {
                    "total_suburbs": num_suburbs,
                    "suburb_index": idx,
                    "suburb_x": df_base.loc[idx, "suburb_x"],
                    "suburb_y": df_base.loc[idx, "suburb_y"],
                    "analytical_star_y": df_an.loc[idx, "star_y"]
                    if idx in df_an.index
                    else None,
                    "brute_force_star_y": df_bf.loc[idx, "star_y"],
                    "total_bf_run_time_sec": metrics["total_bf_time"],
                    "total_an_run_time_sec": metrics["total_an_time"],
                    "optimal_L_found_analytical": metrics["best_L_an"],
                    "optimal_L_found_brute_force": metrics["best_L_bf"],
                }
            )

    # csv
    export_df = pd.DataFrame(csv_records)
    csv_filename = "network_performance.csv"
    export_df.to_csv(csv_filename, index = False)

    # analytical vs suburbs amount
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.plot(
        suburb_counts,
        analytical_times,
        color="#1f77b4",
        marker="s",
        linewidth=2,
        label="Pure Analytical Path",
    )
    plt.title(
        "Analytical Method: Processing Time vs. Number of Suburbs",
        fontsize=13,
        weight="bold",
    )
    plt.xlabel("Number of Suburbs ($N$)", fontsize=11)
    plt.ylabel("Execution Time (Seconds)", fontsize=11)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("analytical_vs_suburbs.png", dpi=300)
    plt.close()  # closes the figure container to start fresh

    # analytical vs brute force
    x = np.arange(len(suburb_counts))  # label locations based on loop steps
    width = 0.35  # width of each individual bar

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create side-by-side bars
    ax.bar(
        x - width / 2, analytical_times, width, label="Analytical", color= "#e377c2"
    )
    ax.bar(
        x + width / 2, brute_force_times, width, label="Brute Force", color= "#17becf"
    )

    # add labels and formatting
    ax.set_xlabel("Number of Suburbs")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Runtime Comparison: Analytical vs. Brute Force Method")
    ax.set_xticks(x)
    ax.set_xticklabels(suburb_counts)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("analytical_vs_brute.png", dpi=300)
    plt.show()


def generate_network_tradeoff(df_base, alpha_spectrum = [0.1, 0.3, 0.5, 0.7, 0.9]):
    wiring_costs = []
    conduction_delays = []

    networks = []

    for a in alpha_spectrum:
        df = df_base.copy()
        df["star_y"] = [get_star_point_analytical(x, y, a) for x, y in zip(df["suburb_x"], df["suburb_y"])]
        df = df.sort_values(by="star_y").reset_index(drop=True)
        best_L, best_cost, _ = find_best_L_analytical(df, a)
        actual_connections, _ = get_optimal_connection(df, best_L, a)

        total_wire = best_L
        total_delay = 0.0

        for sub_x, sub_y, actual_connect_y in zip(df["suburb_x"], df["suburb_y"], actual_connections):
            suburb = np.array([sub_x, sub_y])
            link_length = np.linalg.norm(suburb - np.array([0, actual_connect_y]))
            total_wire += link_length
            total_delay += (actual_connect_y + link_length)

        wiring_costs.append(total_wire)
        conduction_delays.append(total_delay)

        networks.append(
            {
                "alpha": a,
                "best_L": best_L,
                "df": df,
                "connections": actual_connections,
                "wire": total_wire,
                "delay": total_delay,
                "total_cost": best_cost,
            }
        )
    
    fig_main, ax_main = plt.subplots(figsize=(10, 7))
    ax_main.grid(True, linestyle="--", alpha=0.5)

    # Plot Pareto line
    ax_main.plot(
        wiring_costs,
        conduction_delays,
        color="purple",
        linestyle="-",
        linewidth=2.5,
        alpha=0.7,
        zorder=1,
    )

    # scatter points with custom color mapping matching alpha values
    ax_main.scatter(
        wiring_costs,
        conduction_delays,
        color="#1f77b4",
        s=120,
        edgecolor="black",
        zorder=3,
        label="Optimized Networks",
    )
    
    # Annotate points on the curve with simple alpha tags
    for a, x, y in zip(alpha_spectrum, wiring_costs, conduction_delays):
        ax_main.annotate(
            f" a={a:.1f}",
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            weight="bold",
        )

    ax_main.set_title(
        "Pareto Frontier: Network Cost Component Trade-offs",
        fontsize=14,
        weight="bold",
    )
    ax_main.set_xlabel(
        "Total Wiring Infrastructure Cost (Network Physical Length)",
        fontsize=11,
    )
    ax_main.set_ylabel(
        "Total Conduction Delay (Cumulative Travel Path Time)", fontsize=11
    )

    plt.savefig("pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig_main)

    colors = ["#1f77b4", "#9467bd", "#e377c2", "#ff7f0e", "#17becf"]

    for idx, net in enumerate(networks):
        fig_sub, ax_sub = plt.subplots(figsize = (6, 5))

        ax_sub.plot([0, 0], [0, net["best_L"]], color="black", linewidth=3)
        ax_sub.scatter(0, 0, color="red", s=40, marker="*")  # downtown origin

        for s_idx, (sub_x, sub_y, actual_connect_y) in enumerate(
            zip(net["df"]["suburb_x"], net["df"]["suburb_y"], net["connections"])
        ):
            ax_sub.scatter(
                sub_x,
                sub_y,
                color=colors[s_idx % len(colors)],
                s=25,
                edgecolors="black",
                zorder=4,
            )
            ax_sub.plot(
                [sub_x, 0],
                [sub_y, actual_connect_y],
                color="orange",
                linewidth=1.5,
                linestyle="--",
                zorder=3,
            )

        # Style side panel sub-windows
        ax_sub.set_xlim(-3, 3)
        ax_sub.set_ylim(-0.5, float(df_base["suburb_y"].max()) + 0.5)
        ax_sub.axis("off")  # removes borders for a clean presentation look

        # add side label layout text next to every network panel
        info_string = (
        f"Alpha ($\\alpha$) = {net['alpha']:.1f}\n"
        f"Wire Length: {net['wire']:.1f}\n"
        f"Travel Delay: {net['delay']:.1f}\n"
        f"Total Cost: {net['total_cost']:.1f}"
        )
        # places text to the left side of each mini panel window
        ax_sub.text(
            1.75,
            float(df_base["suburb_y"].max()) / 2,
            info_string,
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5, boxstyle="round,pad=0.3"),
        )

        file_name = f"network_alpha_{net['alpha']:.1f}.png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig_sub)



a = 0.5
main_length = 5

downtown = np.array([0, 0])

df = pd.DataFrame(
    {
        "suburb_x": [1, 2, -1],
        "suburb_y": [2, 5, 3],
    }
)

draw_steps(df, a)

star_y_list = [get_star_point_analytical(x, y, a) for x, y in zip(df["suburb_x"], df["suburb_y"])]

# save best star points back into the DataFrame
df["star_x"] = 0.0
df["star_y"] = star_y_list

# sort, so that star points in order
df = df.sort_values(by = "star_y").reset_index(drop=True)
print(df)

best_L, min_cost, double_star_points = find_best_L_analytical(df, a)
print(f"\n[Calculus Search] Best Mainline Length: {best_L:.6f}")
print(f"[Calculus Search] Minimum Overall Network Cost: {min_cost:.6f}")

draw_networks(df, a, best_L, main_length)
"""
generate_network_tradeoff(df)
plot_cost_optimization(df, a)
metrics = run_timing_comparison(df, a, main_length)
print("Time for brute force method:", metrics["total_bf_time"])
print("Time for analytical method:", metrics["total_an_time"])
bf_y, an_y = compare(df, a, main_length)
print("Brute force star points:", bf_y)
print("Analytical star points:", an_y)
print()




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
analytical_vs_brute()



"""