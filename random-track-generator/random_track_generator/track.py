import gpxpy
import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
from xml.dom import minidom
# from .mujoco_utils import create_mujoco_with_checkpoints, _create_mujoco_xml
# from .mujoco_new import _create_mujoco_xml
import os
import json

class Mode(Enum):
    """ 
    Possible modes for how Voronoi regions are selected.
    
    1. Expand:
        Find closest nodes around starting node.
        Results in roundish track shapes.
    
    2. Extend:
        Find nodes closest to line extending from starting node.
        Results in elongated track shapes.
        
    3. Random:
        Select all regions randomly.
        Results in large track shapes.
    """
    EXPAND = 1
    EXTEND = 2
    RANDOM = 3

class SimType(Enum):
    """ Selection between output format for different simulators.

    1. FSSIM:
        Output FSSIM compatible .yaml file.
    2. FSDS:
        Output FSDS compatible .csv file 
    """
    FSSIM = 1
    FSDS = 2
    GPX = 3
    MUJOCO = 4

@dataclass
class Track:
    """ Track dataclass

    Attributes:
         cones_left (np.ndarray): Left cones of track.
         cones_right (np.ndarray): Right cones of track.
    """
    cones_left: np.ndarray
    cones_right: np.ndarray
    centreline: np.ndarray | None = None
    
    orange_cones = [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]]

    def as_tuple(self):
        """ Returns cones as tuple of left and right cones.
        """
        return self.cones_left, self.cones_right

    def save(self, location: str | Path, sim_type: SimType | str, *,
             lat_offset: float = 0.0, lon_offset: float = 0.0, z_offset: float = 0.0,
             include_checkpoints: bool = False, n_checkpoints: int=10):
        
        """ Saves track in specified format for use in different simulators.

        Args:
            location: Location to save track to.
            sim_type: Format to save track in. Must be either "fssim", "fsds" or "gpx".
            lat_offset: Latitude offset for GPX output format, in degrees.
            lon_offset: Longitude offset for GPX output format, in degrees.
            z_offset: Altitude offset for GPX output format, in meters.
        """
        sim_type = SimType[sim_type.upper()] if isinstance(sim_type, str) else SimType(sim_type)
        path = Path(location)

        if sim_type == SimType.FSSIM:
            with open(path / "random_track.yaml", 'w') as f:
                yaml.dump({
                    'cones_left': self.cones_left.tolist(),
                    'cones_right': self.cones_right.tolist(),
                    'cones_orange': [],
                    'cones_orange_big': [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]],
                    'starting_pose_cg': [0., 0., 0.],
                    'tk_device': [[6., 3.], [6., -3.]],
                }, f)

        elif sim_type == SimType.FSDS:
            out = path / "random_track.csv"
            with open(out, 'w') as f:
                for cone in self.cones_left:
                    f.write(f"blue,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                for cone in self.cones_right:
                    f.write(f"yellow,{cone[0]},{cone[1]},0,0.01,0.01,0\n")
                f.write("big_orange,-2.2,4.7,0,0.01,0.01,0\n")
                f.write("big_orange,2.2,4.7,0,0.01,0.01,0\n")
                f.write("big_orange,-2.2,7.3,0,0.01,0.01,0\n")
                f.write("big_orange,2.2,7.3,0,0.01,0.01,0\n")

        elif sim_type == SimType.GPX:
            gpx = gpxpy.gpx.GPX()
            gpx.tracks.append(gpxpy.gpx.GPXTrack())

            deg_per_m_lat = np.degrees(1 / 6378100)
            deg_per_m_lon = np.degrees(1 / 6378100) / np.cos(np.radians(lat_offset))

            for cone in np.vstack([self.cones_left, self.cones_right]):
                lat = lat_offset + cone[1] * deg_per_m_lat
                lon = lon_offset + cone[0] * deg_per_m_lon
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=z_offset))

            (path / "random_track.gpx").write_text(gpx.to_xml())    
            
        elif sim_type == SimType.MUJOCO:
            # Optionally compute checkpoints
            # checkpoints = None
            # print(self.centreline)
            # print(type(self.centreline))
            # raise
            # if include_checkpoints:
            #     cp_out_file = path / "checkpoints.json"
            #     self.save_checkpoints(cp_out_file, n_checkpoints=n_checkpoints)

            xml_str = _create_mujoco_xml(
                self.cones_left,
                self.cones_right,
                cones_orange_big=self.orange_cones,
            )
            
            distances = point_to_segment_distance(self.centreline, self.orange_cones[0], self.orange_cones[1])
            closest_idx = np.argmin(distances)
            centreline_rolled = np.roll(self.centreline, -closest_idx, axis=0)
            
            centreline_out_file = path / "centreline.csv"
            np.savetxt(centreline_out_file, centreline_rolled, delimiter=',', 
               header='x,y', comments='')
            
            os.makedirs(path, exist_ok=True)
            out_file = path / "track.xml"
            with open(out_file, 'w') as f:
                f.write(xml_str)

            print(f"MuJoCo track exported to: {out_file}")

    def get_frenet_frame(self, point_index):
        """
        Calculate Frenet-Serret frame at a given point on the centreline.
        
        Args:
            point_index: Index of the point on the centreline
            
        Returns:
            tangent: Unit tangent vector
            normal: Unit normal vector (points left/right)
        """
        if self.centreline is None:
            raise ValueError("Track has no centreline data")
        
        # Get point and neighbours for derivative calculation
        n_points = len(self.centreline)
        
        # Use central differences for interior points, forward/backward for edges
        if point_index == 0:
            # Forward difference
            dx = self.centreline[1, 0] - self.centreline[0, 0]
            dy = self.centreline[1, 1] - self.centreline[0, 1]
        elif point_index == n_points - 1:
            # Backward difference
            dx = self.centreline[-1, 0] - self.centreline[-2, 0]
            dy = self.centreline[-1, 1] - self.centreline[-2, 1]
        else:
            # Central difference
            dx = self.centreline[point_index + 1, 0] - self.centreline[point_index - 1, 0]
            dy = self.centreline[point_index + 1, 1] - self.centreline[point_index - 1, 1]
        
        # Tangent vector (derivative)
        tangent = np.array([dx, dy])
        # Normalize
        tangent = tangent / np.linalg.norm(tangent)
        
        # Normal vector (rotate tangent by +90 degrees for left normal)
        # For a 2D curve, normal is perpendicular to tangent
        normal = np.array([tangent[1], -tangent[0]])  # Points left (counter-clockwise)
        
        return tangent, normal
    
    def get_checkpoints(self, n_checkpoints=10):
        """
        Split the centreline into equal-length segments and get Frenet frames at each checkpoint.
        
        Args:
            n_checkpoints: Number of checkpoints
            
        Returns:
            checkpoints: List of dictionaries containing checkpoint info
        """
        if self.centreline is None:
            raise ValueError("Track has no centreline data")
        
        
        # self.centreline = np.roll(self.centreline, 10, axis=0)
        distances = point_to_segment_distance(self.centreline, self.orange_cones[0], self.orange_cones[1])
        closest_idx = np.argmin(distances)
        self.centreline = np.roll(self.centreline, -closest_idx, axis=0)
                
        # Calculate cumulative arc length along centreline
        distances = np.zeros(len(self.centreline))
        for i in range(1, len(self.centreline)):
            distances[i] = distances[i-1] + np.linalg.norm(
                self.centreline[i] - self.centreline[i-1]
            )
        
        total_length = distances[-1]
        
        print(f"First Point: {self.centreline[0]}")
        
        # Desired distances for checkpoints (excluding start point, including end point)
        checkpoint_distances = np.linspace(0, total_length, n_checkpoints+1)[:-1]
        
        checkpoints = []
        
        for i, target_dist in enumerate(checkpoint_distances):
            
            # if i == len(checkpoint_distances) - 1:
            #     break
            # Find the closest point on centreline to target distance
            idx = np.argmin(np.abs(distances - target_dist)) 
            
            # Get point coordinates
            point = self.centreline[idx]
            
            # Get Frenet frame
            tangent, normal = self.get_frenet_frame(idx)
            
            checkpoints.append({
                'index': i,
                'distance': target_dist,
                'point': point,
                'tangent': tangent,
                'normal': normal,
                'centreline_idx': idx
            })
        
        
        return checkpoints
    # Add to your Track class
    def save_checkpoints(self, filepath: str | Path, n_checkpoints: int = 10):
        """
        Save checkpoints to a JSON file.
        
        Args:
            filepath: Path where to save the checkpoints JSON
            n_checkpoints: Number of checkpoints to generate
        """
        # Generate checkpoints
        checkpoints = self.get_checkpoints(n_checkpoints=n_checkpoints)
        
        # Convert numpy arrays to lists for JSON serialization
        checkpoints_serializable = []
        for cp in checkpoints:
            checkpoints_serializable.append({
                'index': cp['index'],
                'distance': float(cp['distance']),  # Convert to float
                'point': cp['point'].tolist(),
                'tangent': cp['tangent'].tolist(),
                'normal': cp['normal'].tolist(),
                'centreline_idx': int(cp['centreline_idx'])
            })
        
        # Calculate track length from centreline
        track_length = 0.0
        if self.centreline is not None:
            for i in range(1, len(self.centreline)):
                track_length += np.linalg.norm(self.centreline[i] - self.centreline[i-1])
        
        # Save to file
        data = {
            'checkpoints': checkpoints_serializable,
            'track_length': float(track_length),
            'n_checkpoints': len(checkpoints),
            'metadata': {
                'cones_left': self.cones_left.shape[0] if self.cones_left is not None else 0,
                'cones_right': self.cones_right.shape[0] if self.cones_right is not None else 0,
            }
        }
        
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Checkpoints saved to {filepath}")
        print(f"  Track length: {track_length:.2f} m")
        print(f"  Number of checkpoints: {len(checkpoints)}")
        
        return checkpoints
    
    
def _create_mujoco_xml(cones_left, cones_right, cones_orange_big):
    """
    Create MuJoCo XML for dynamic track cones that can be pushed/knocked over.
    """
    # Create root element
    mujoco = ET.Element("mujoco", model="random_track")
    
    # Add compiler options
    compiler = ET.SubElement(mujoco, "compiler", 
                             angle="radian", 
                             coordinate="local",
                             autolimits="true")
    
    # Add default settings
    default = ET.SubElement(mujoco, "default")
    
    # Default cone properties
    # cone_default = ET.SubElement(default, "default", cls="cone")
    cone_default = ET.SubElement(default, "default", **{"class": "cone"})

    ET.SubElement(cone_default, "joint", 
                  type="free",  # This is key - allows free motion
                  damping="0.1",  # Add some damping to prevent endless rolling
                  frictionloss="0.05")  # Friction in joints
    
    ET.SubElement(cone_default, "geom",
                  condim="4",  # Use 4 for friction cones, 6 for full friction
                  friction="0.5 0.005 0.0001",  # Sliding, torsional, rolling friction
                  solref="0.02 1.0",  # Contact solver parameters
                  solimp="0.99 0.99 0.001")  # Contact impedance
    
    # Add assets (colors)
    asset = ET.SubElement(mujoco, "asset")
    
    # Define colors for cones
    ET.SubElement(asset, "material", 
                  name="blue_cone",
                  rgba="0.2 0.4 1.0 1.0",
                  specular="0.5",
                  shininess="0.2")
    
    ET.SubElement(asset, "material",
                  name="yellow_cone",
                  rgba="1.0 0.8 0.0 1.0",
                  specular="0.5",
                  shininess="0.2")
    
    ET.SubElement(asset, "material",
                  name="orange_cone",
                  rgba="1.0 0.5 0.0 1.0",
                  specular="0.5",
                  shininess="0.2")
    
    # World body containing all cones
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Add left cones (blue) - now with joints
    for i, cone_pos in enumerate(cones_left):
        # Create body with FREE joint (allows 6DOF motion)
        body = ET.SubElement(worldbody, "body",
                            name=f"cone_left_{i:04d}",
                            pos=f"{cone_pos[0]} {cone_pos[1]} 0.325")
        
        # Add free joint for physics
        ET.SubElement(body, "joint", 
                     name=f"cone_left_joint_{i:04d}",
                     type="free",
                     damping="0.1",
                     frictionloss="0.05")
        
        # Visual geometry
        ET.SubElement(body, "geom",
                     name=f"cone_left_vis_{i:04d}",
                     type="cylinder",
                     size="0.228 0.325",
                     material="blue_cone",
                     mass="1.0",  # Lightweight but not too light
                     density="100",  # Alternative to mass
                     condim="4",
                     friction="0.5 0.005 0.0001",
                     solref="0.02 1.0",
                     solimp="0.99 0.99 0.001")
        
        # # Optional: Add a second geom for better rolling (sphere at bottom)
        # ET.SubElement(body, "geom",
        #              name=f"cone_left_bottom_{i:04d}",
        #              type="sphere",
        #              size="0.14",  # Slightly smaller than cylinder radius
        #              pos="0 0 -0.1",  # At bottom of cone
        #              material="blue_cone",
        #              mass="0.05",
        #              condim="4",
        #              friction="0.5 0.005 0.0001")
    
    # Add right cones (yellow) - with joints
    for i, cone_pos in enumerate(cones_right):
        body = ET.SubElement(worldbody, "body",
                            name=f"cone_right_{i:04d}",
                            pos=f"{cone_pos[0]} {cone_pos[1]} 0.325")
        
        ET.SubElement(body, "joint", 
                     name=f"cone_right_joint_{i:04d}",
                     type="free",
                     damping="0.1",
                     frictionloss="0.05")
        
        ET.SubElement(body, "geom",
                     name=f"cone_right_vis_{i:04d}",
                     type="cylinder",
                     size="0.228 0.325",
                     material="yellow_cone",
                     mass="1.0",
                     condim="4",
                     friction="0.5 0.005 0.0001")
        
        # # Bottom sphere for better tipping physics
        # ET.SubElement(body, "geom",
        #              name=f"cone_right_bottom_{i:04d}",
        #              type="sphere",
        #              size="0.14",
        #              pos="0 0 -0.1",
        #              material="yellow_cone",
        #              mass="0.05")
    
    # Add big orange cones (start/finish area) - heavier
    if cones_orange_big is None:
        cones_orange_big = [[-2.2, 4.7], [2.2, 4.7], [-2.2, 7.3], [2.2, 7.3]]
    
    for i, cone_pos in enumerate(cones_orange_big):
        body = ET.SubElement(worldbody, "body",
                            name=f"cone_orange_big_{i:04d}",
                            pos=f"{cone_pos[0]} {cone_pos[1]} 0.505")
        
        ET.SubElement(body, "joint", 
                     name=f"cone_orange_big_joint_{i:04d}",
                     type="free",
                     damping="0.2",
                     frictionloss="0.1")
        
        ET.SubElement(body, "geom",
                     name=f"cone_orange_big_vis_{i:04d}",
                     type="cylinder",
                     size="0.3 0.505",
                     material="orange_cone",
                     mass="2.0",  # Heavier
                     condim="4",
                     friction="0.5 0.005 0.0001")
        
        # # Bottom sphere
        # ET.SubElement(body, "geom",
        #              name=f"cone_orange_big_bottom_{i:04d}",
        #              type="sphere",
        #              size="0.28",
        #              pos="0 0 -0.2",
        #              material="orange_cone",
        #              mass="0.2")
    
    # Convert to pretty XML string
    rough_string = ET.tostring(mujoco, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def point_to_segment_distance(points, p1, p2):
    """
    points: (N,2) numpy array
    p1, p2: endpoints of segment
    returns: distance of each point to the segment
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    seg = p2 - p1
    seg_len_sq = np.dot(seg, seg)

    # projection parameter t
    t = np.dot(points - p1, seg) / seg_len_sq

    # clamp to segment
    t = np.clip(t, 0, 1)

    # closest points on segment
    projection = p1 + t[:, None] * seg

    # distance
    return np.linalg.norm(points - projection, axis=1)