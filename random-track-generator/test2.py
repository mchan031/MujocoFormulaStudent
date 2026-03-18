import matplotlib.pyplot as plt
import numpy as np
from random_track_generator import generate_track
from random_track_generator import SimType
import os

def plot_track_with_frames(track, n_checkpoints=10, arrow_scale=3.0):
    """
    Plot the track with Frenet frames at checkpoints.
    
    Args:
        track: Track object with centreline
        n_checkpoints: Number of checkpoints
        arrow_scale: Scale factor for frame arrows
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot cones
    ax.scatter(track.cones_left[:, 0], track.cones_left[:, 1], 
               c='blue', s=10, label='Left cones', alpha=0.5)
    ax.scatter(track.cones_right[:, 0], track.cones_right[:, 1], 
               c='red', s=10, label='Right cones', alpha=0.5)
    
    # Plot centreline
    if track.centreline is not None:
        ax.plot(track.centreline[:, 0], track.centreline[:, 1], 
                'g-', linewidth=2, label='Centreline', alpha=0.7)
    
    # Get checkpoints
    checkpoints = track.get_checkpoints(n_checkpoints)
    # print(f"{checkpoints = }")
    # Plot checkpoints and frames
    for cp in checkpoints:
        point = cp['point']
        tangent = cp['tangent']
        normal = cp['normal']
        
        # Plot checkpoint point
        ax.plot(point[0], point[1], 'ko', markersize=8)
        ax.annotate(f'CP{cp["index"]}', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Plot tangent vector (direction of travel)
        ax.arrow(point[0], point[1], 
                tangent[0] * arrow_scale, tangent[1] * arrow_scale,
                head_width=0.3, head_length=0.3, fc='orange', ec='orange', 
                alpha=0.8, label='Tangent' if cp['index'] == 1 else '')
        
        # Plot normal vector (perpendicular, points left)
        ax.arrow(point[0], point[1], 
                normal[0] * arrow_scale, normal[1] * arrow_scale,
                head_width=0.3, head_length=0.3, fc='purple', ec='purple', 
                alpha=0.8, label='Normal (left)' if cp['index'] == 1 else '')
        
        print(f"{normal = } at {point = }")
    
    # ax.plot(track.cones_left[0, 0], track.cones_left[0, 1], 'bo', label='Left cones')
    # ax.plot(track.cones_right[0, 0], track.cones_right[0, 1], 'yo', label='Right cones')
    # Rotate 90 degrees CCW: (x, y) -> (-y, x)
    cones_orange_big = [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]]
    # cones_orange_big = [[-2.2, 4.7], [2.2, 4.7], [-2.2, 7.3], [2.2, 7.3]]
    # smtg = [[4.7, 2.2], [4.7, -2.2], [7.3, 2.2], [7.3, -2.2]]
    rotated = np.array(cones_orange_big)
    # rotated = np.column_stack([-coords[:, 1], coords[:, 0]])
    ax.scatter(rotated[:, 0], rotated[:, 1], 
               c='red', s=100, marker='^', label='Orange big cones')
    # plt.plot()

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Track with {n_checkpoints} Checkpoints and Frenet Frames')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax, checkpoints


def get_perpendicular_line(checkpoint, length=10.0, n_points=100):
    """
    Get points along the line perpendicular to the track at a checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary from get_checkpoints()
        length: Total length of the perpendicular line (extends both directions)
        n_points: Number of points to generate along the line
    
    Returns:
        line_points: Array of points along the perpendicular line
    """
    point = checkpoint['point']
    normal = checkpoint['normal']  # This points left
    
    # Generate points along the perpendicular line
    # The line extends in both +normal and -normal directions
    t = np.linspace(-length/2, length/2, n_points)
    
    # Points = centre_point + t * normal
    line_points = point + np.outer(t, normal)
    
    return line_points

def plot_perpendicular_lines(track, n_checkpoints=10, line_length=15.0):
    """
    Plot the perpendicular lines at each checkpoint.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot track elements
    ax.scatter(track.cones_left[:, 0], track.cones_left[:, 1], 
               c='blue', s=10, label='Left cones', alpha=0.5)
    ax.scatter(track.cones_right[:, 0], track.cones_right[:, 1], 
               c='red', s=10, label='Right cones', alpha=0.5)
    
    # if track.centreline is not None:
    #     ax.plot(track.centreline[:, 0], track.centreline[:, 1], 
    #             'g-', linewidth=2, label='Centreline', alpha=0.7)
    
    # Get checkpoints
    checkpoints = track.get_checkpoints(n_checkpoints)
    
    # Plot perpendicular lines
    for cp in checkpoints:
        point = cp['point']
        normal = cp['normal']
        
        # Calculate line endpoints
        line_start = point - normal * (line_length/2)
        line_end = point + normal * (line_length/2)
        
        # Plot perpendicular line
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 
                'orange', linewidth=10, alpha=0.9)
        
        # Plot checkpoint point
        ax.plot(point[0], point[1], 'ko', markersize=6)
        ax.annotate(f'CP{cp["index"]}', (point[0], point[1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    
    # q, p = 350, 450
    # ax.plot(track.centreline[q:p, 0], track.centreline[q:p, 1], 'go', label='Start')
    cones_orange_big = [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]]
    # cones_orange_big = [[-2.2, 4.7], [2.2, 4.7], [-2.2, 7.3], [2.2, 7.3]]
    # smtg = [[4.7, 2.2], [4.7, -2.2], [7.3, 2.2], [7.3, -2.2]]
    rotated = np.array(cones_orange_big)
    # rotated = np.column_stack([-coords[:, 1], coords[:, 0]])
    ax.scatter(rotated[:, 0], rotated[:, 1], 
               c='red', s=100, marker='^', label='Orange big cones')
    # plt.plot()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Perpendicular Lines at {n_checkpoints} Checkpoints')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


# Generate track
# track = generate_track(n_points=50, n_regions=5, 
#                        min_bound=0, max_bound=100, 
#                        mode='EXPAND', seed=42)


track = generate_track(
    n_points=60,       # Voronoi points
    n_regions=15,      # Regions to select
    min_bound=0.,      # Minimum x/y bound [m]
    max_bound=150.,    # Maximum x/y bound [m]
    mode="expand",     # Generation mode
    seed=239            # Optional: for reproducibility
)

print(track.orange_cones)

n_checkpoints = 11
# Method 1: Plot with Frenet frames
fig1, ax1, checkpoints = plot_track_with_frames(track, n_checkpoints=n_checkpoints, arrow_scale=5.0)
# print(f"{checkpoints = }")
# checkpoints = track.get_checkpoints(n_checkpoints)

# Method 2: Plot perpendicular lines
# fig2, ax2 = plot_perpendicular_lines(track, n_checkpoints=n_checkpoints, line_length=5.0)
# track.save("mujoco_tracks", SimType.MUJOCO,  include_checkpoints=True)

track_dir = "mujoco_tracks/track_6"
os.makedirs(track_dir, exist_ok=True)

track.save(track_dir, SimType.MUJOCO, include_checkpoints=True, n_checkpoints=n_checkpoints)
track.save(track_dir, SimType.FSDS, include_checkpoints=True, n_checkpoints=n_checkpoints)

# print(f"{checkpoints = }")
# # Access checkpoint data
# for cp in checkpoints:
#     print(f"Checkpoint {cp['index']}: Distance along track: {cp['distance']:.2f} m")
#     # print(f"  Position: ({cp['point'][0]:.2f}, {cp['point'][1]:.2f})")
#     # print(f"  Distance along track: {cp['distance']:.2f} m")
#     # print(f"  Tangent vector: [{cp['tangent'][0]:.3f}, {cp['tangent'][1]:.3f}]")
#     # print(f"  Normal vector: [{cp['normal'][0]:.3f}, {cp['normal'][1]:.3f}]")
#     # print()

# # Get perpendicular line at checkpoint 5
# if len(checkpoints) >= 5:
#     line_points = get_perpendicular_line(checkpoints[4], length=15.0)
#     print(f"Perpendicular line at CP5 has {len(line_points)} points")