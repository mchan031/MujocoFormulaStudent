import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Experiment summary (best RL run per experiment)
experiments = [
    {
        'name': '1_no_waypoint_constant_track',
        'lap_completion_rate': 0.72,
        'crash_rate': 0.4,
        'mean_speed_ms': 6.32
    },
    {
        'name': '2_relative_waypoint_constant_track',
        'lap_completion_rate': 0.64,
        'crash_rate': 0.48,
        'mean_speed_ms': 6.26
    },
    {
        'name': '3_distance_waypoint_constant_track',
        'lap_completion_rate': 0.24,
        'crash_rate': 0.96,
        'mean_speed_ms': 6.7
    },
    {
        'name': '4_no_waypoint_random_track',
        'lap_completion_rate': 0.52,
        'crash_rate': 0.32,
        'mean_speed_ms': 4.92
    },
    {
        'name': '5_relative_waypoint_random_track',
        'lap_completion_rate': 0.64,
        'crash_rate': 0.48,
        'mean_speed_ms': 6.26
    },
    {
        'name': '6_distance_waypoint_random_track',
        'lap_completion_rate': 0.56,
        'crash_rate': 0.92,
        'mean_speed_ms': 7.53
    },
]

names = [exp['name'] for exp in experiments]
lap_completion = [exp['lap_completion_rate'] for exp in experiments]
crash_rate = [exp['crash_rate'] for exp in experiments]
speed = [exp['mean_speed_ms'] for exp in experiments]

x = range(len(experiments))


# --- Split into 3 plots ---
fig, axs = plt.subplots(3, 1, figsize=(12, 12))

# Lap Completion Rate
axs[0].bar(x, lap_completion, color='tab:blue')
axs[0].set_xticks(x)
axs[0].set_xticklabels(names, rotation=15, ha='right')
axs[0].set_ylabel('Lap Completion Rate')
axs[0].set_title('Lap Completion Rate by Experiment')

# Crash Rate
axs[1].bar(x, crash_rate, color='tab:orange')
axs[1].set_xticks(x)
axs[1].set_xticklabels(names, rotation=15, ha='right')
axs[1].set_ylabel('Crash Rate')
axs[1].set_title('Crash Rate by Experiment')

# Mean Speed
axs[2].bar(x, speed, color='tab:green')
axs[2].set_xticks(x)
axs[2].set_xticklabels(names, rotation=15, ha='right')
axs[2].set_ylabel('Mean Speed (m/s)')
axs[2].set_title('Mean Speed by Experiment')

plt.tight_layout()
plt.show()

# --- Scatter plot: Lap Completion Rate vs Crash Rate ---
plt.figure(figsize=(8, 6))
plt.scatter(lap_completion, crash_rate, color='purple')
for i, name in enumerate(names):
    plt.text(lap_completion[i]+0.01, crash_rate[i], name, fontsize=9, va='center')
plt.xlabel('Lap Completion Rate')
plt.ylabel('Crash Rate')
plt.title('Trade-off: Lap Completion Rate vs Crash Rate')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
