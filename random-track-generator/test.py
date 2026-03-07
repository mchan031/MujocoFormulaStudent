from random_track_generator import generate_track
import matplotlib.pyplot as plt


track = generate_track(
    n_points=60,       # Voronoi points
    n_regions=20,      # Regions to select
    min_bound=0.,      # Minimum x/y bound [m]
    max_bound=150.,    # Maximum x/y bound [m]
    mode="extend",     # Generation mode
    seed=10            # Optional: for reproducibility
)

cones_left, cones_right = track.as_tuple()

# print("Left cones:\n", cones_left)
# track.cones_left
# print(len(track.cones_left))
print((track.centreline))

plt.plot(track.centreline[:, 0], track.centreline[:, 1], 'b-', label='Centreline')
plt.scatter(track.cones_left[:, 0], track.cones_left[:, 1], c='blue', s=10, label='Left cones')
plt.scatter(track.cones_right[:, 0], track.cones_right[:, 1], c='yellow', s=10, label='Right cones')
plt.plot(track.cones_left[0, 0], track.cones_left[0, 1], 'bo', label='Left cones')
plt.plot(track.cones_right[0, 0], track.cones_right[0, 1], 'yo', label='Right cones')
plt.plot(track.centreline[0, 0], track.centreline[0, 1], 'go', label='Start')
plt.axis('equal')
plt.legend()
plt.show()
