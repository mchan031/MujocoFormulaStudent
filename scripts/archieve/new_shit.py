import os
import json
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\munki\OneDrive\Documents\GitHub\MujocoFormulaStudent\report"


label_map = {
    "1_no_waypoint_constant_track": "Exp1\n(E2E-Fixed)",
    "2_relative_waypoint_constant_track": "Exp2\n(Rel-Fixed)",
    "3_distance_waypoint_constant_track": "Exp3\n(Dist-Fixed)",
    "4_no_waypoint_random_track": "Eexp4\n(E2E-Rand)",
    "5_relative_waypoint_random_track": "Exp5\n(Rel-Rand)",
    "6_distance_waypoint_random_track": "Exp6\n(Dist-Rand)"
}

experiments = {}
results_summary = []

for exp_name in sorted(os.listdir(BASE_DIR)):
    exp_path = os.path.join(BASE_DIR, exp_name)

    if not os.path.isdir(exp_path):
        continue

    best_run = None
    best_completion = -1

    # loop through runs
    for run_name in os.listdir(exp_path):
        run_path = os.path.join(exp_path, run_name)
        json_path = os.path.join(run_path, "results.json")

        if not os.path.isfile(json_path):
            continue

        with open(json_path, "r") as f:
            data = json.load(f)

        completion = data["overall"]["lap_completion_rate"]

        if completion > best_completion:
            best_completion = completion
            best_run = data

    if best_run is not None:
        overall = best_run["overall"]

        results_summary.append({
            "exp": exp_name,
            "completion": overall["lap_completion_rate"] * 100,
            "crash": overall["crash_rate"] * 100,
            "speed": overall["mean_speed_ms"]
        })

# --- Prepare data ---
# labels = [r["exp"] for r in results_summary]
labels = [label_map.get(r["exp"], r["exp"]) for r in results_summary]
print(labels)
completion = [r["completion"] for r in results_summary]
crash = [r["crash"] for r in results_summary]
speed = [r["speed"] for r in results_summary]

x = range(len(labels))

# # --- Plot ---
# plt.figure()

# bar_width = 0.25

# plt.bar([i - bar_width for i in x], completion, width=bar_width)
# plt.bar(x, crash, width=bar_width)
# plt.bar([i + bar_width for i in x], speed, width=bar_width)

# plt.xticks(x, labels, rotation=45)
# plt.xlabel("Experiments")
# plt.ylabel("Value")
# plt.title("Comparison of Completion, Crash Rate, and Speed")

# plt.legend(["Completion (%)", "Crash (%)", "Speed (m/s)"])

# plt.tight_layout()
# plt.show()

plt.figure()

bar_width = 0.35
x = range(len(labels))

plt.bar([i - bar_width/2 for i in x], completion, width=bar_width)
plt.bar([i + bar_width/2 for i in x], crash, width=bar_width)

plt.xticks(x, labels)
plt.ylabel("Percentage (%)")
plt.title("Completion vs Crash Rate Across Experiments")
plt.legend(["Completion", "Crash"])
# --- Add speed labels on top ---
for i in x:
    y = max(completion[i], crash[i])  # place above taller bar
    plt.text(i, y + 2, f"{speed[i]:.2f} m/s", ha='center', fontsize=9)

plt.ylim(0, 110)
plt.tight_layout()
plt.show()

plt.figure()

bar_width = 0.35
x = range(len(labels))

# plt.bar([i - bar_width/2 for i in x], completion, width=bar_width)
# plt.bar([i + bar_width/2 for i in x], crash, width=bar_width)

# plt.xticks(x, labels)
# plt.ylabel("Percentage (%)")
# plt.title("Completion vs Crash Rate Across Experiments")
# plt.legend(["Completion", "Crash"])
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()

plt.figure()

plt.bar(labels, speed)

plt.ylabel("Speed (m/s)")
plt.title("Mean Speed Across Experiments")

plt.tight_layout()
plt.show()