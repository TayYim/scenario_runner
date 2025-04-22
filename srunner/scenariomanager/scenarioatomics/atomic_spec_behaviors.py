#! python3
# -*- encoding: utf-8 -*-

import math
import carla
import sys
import os
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
import py_trees
from srunner.tools.scenario_helper import (
    transform_world_vector_to_local,
    adaptive_savgol_filter,
    adaptive_gradient,
)
import os
import csv
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib for non-blocking operation
plt.ion()  # Turn on interactive mode

# Add SPEC root directory to Python path for importing compute_see_carla
spec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if spec_path not in sys.path:
    sys.path.append(spec_path)

# Import compute_see_carla and compute_dsee_carla from hsr_calculation
from src.data_process.hsr_calculation import compute_see_carla, compute_dsee_carla
from src.utils.commonroad_handler import CommonRoadHandler


def find_leftmost_lane(waypoint):
    """Find the leftmost lane of the same road that is a driving lane."""
    curr_waypoint = waypoint
    max_iterations = 10  # Safety limit to prevent infinite loops
    iterations = 0
    
    try:
        while iterations < max_iterations:
            left_waypoint = curr_waypoint.get_left_lane()
            if left_waypoint is None or left_waypoint.road_id != waypoint.road_id:
                break
            
            # Check if the lane is of driving type
            if left_waypoint.lane_type != carla.LaneType.Driving:
                break
                
            curr_waypoint = left_waypoint
            iterations += 1
        return curr_waypoint
    except Exception as e:
        print(f"Error in find_leftmost_lane: {e}")
        return waypoint  # Return original waypoint if there's an error


def find_rightmost_lane(waypoint):
    """Find the rightmost lane of the same road that is a driving lane."""
    curr_waypoint = waypoint
    max_iterations = 10  # Safety limit to prevent infinite loops
    iterations = 0
    
    try:
        while iterations < max_iterations:
            right_waypoint = curr_waypoint.get_right_lane()
            if right_waypoint is None or right_waypoint.road_id != waypoint.road_id:
                break
                
            # Check if the lane is of driving type
            if right_waypoint.lane_type != carla.LaneType.Driving:
                break
                
            curr_waypoint = right_waypoint
            iterations += 1
        return curr_waypoint
    except Exception as e:
        print(f"Error in find_rightmost_lane: {e}")
        return waypoint  # Return original waypoint if there's an error


def get_lane_border_points(waypoint):
    """Get the left and right border points of a lane."""
    try:
        # Lane width is measured from the center to one edge, so we multiply by 0.5
        lane_width = waypoint.lane_width * 0.5
        
        # Compute left and right lane borders
        forward_vector = waypoint.transform.get_forward_vector()
        right_vector = carla.Location(x=-forward_vector.y, y=forward_vector.x, z=0)
        
        # Get border points on the waypoint
        left_border = waypoint.transform.location + carla.Location(
            x=right_vector.x * -lane_width, 
            y=right_vector.y * -lane_width
        )
        right_border = waypoint.transform.location + carla.Location(
            x=right_vector.x * lane_width, 
            y=right_vector.y * lane_width
        )
        
        return left_border, right_border
    except Exception as e:
        print(f"Error in get_lane_border_points: {e}")
        return waypoint.transform.location, waypoint.transform.location  # Return original location if error


class SPECDataCollector(AtomicBehavior):
    """
    This class contains a data collector behavior for SPEC scenarios
    Collects data for all vehicles in the scene, including:
    - Coordinates (x, y)
    - Velocity (vx, vy)
    - Lane ID
    - Steering angle
    - Acceleration
    - Ego vehicle marker
    """

    def __init__(self, actor=None, name="SPECDataCollector", lx=10, ly=60, nrad=4, nring=3, visualize_planning_arrow=True):
        """
        Setup for SPEC data collection
        """
        self.task_id = name + "_" + datetime.datetime.now().strftime("%m%d%H%M%S")
        self.finished = False  # Use to mark whether the terminate function has been executed
        
        # SEE parameters
        ## Note: the x and y in lx and ly are different from the x and y in the Carla world
        ## lx and ly are the lengths of the perception area based on the ego vehicle's heading
        ## lx is the length in the direction of the vehicle's heading
        ## ly is the length perpendicular to the vehicle's heading
        # lx set to 10 (not 12) by default because the lanes are less then highway
        self.lx = lx  # Perception area length x
        self.ly = ly  # Perception area length y
        self.nrad = nrad  # Number of radial divisions
        self.nring = nring  # Number of rings
        
        # Visualization flag
        self.visualize_planning_arrow = visualize_planning_arrow
        
        # Initialize visualization figures
        # We don't create the figures here - they'll be created when needed
        
        # Track all vehicles in the scene (mapped by their IDs)
        self._vehicles_data = {}
        
        # Data structure for each vehicle
        self._data_structure = {
            "game_time": [],
            "vehicle_id": [],
            "x": [],
            "y": [],
            "vx": [],
            "vy": [],
            "lane_id": [],
            "steering": [],
            "acceleration": [],
            "is_ego": [],
            "ttr": []
        }
        
        # Add TTR tracking
        self._data_structure["ttr"] = []
        
        # Add road border tracking
        self.road_borders = {
            "left_border": None,
            "right_border": None
        }
        
        # Initialize CommonRoadHandler instance
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the relative path to the scenario file (going up one directory)
        scenario_path = os.path.join(script_dir, '..', '..', '..', '..', '..', 'src', 'configs', 'map_layout', 'town04_route98_interpolated.xml')
        scenario_path = os.path.normpath(scenario_path) # Normalize the path (e.g., remove '..')
        self._commonroad_handler = CommonRoadHandler(scenario_path=scenario_path, debug=False, visualize=True)
        
        # Initialize base class
        super(SPECDataCollector, self).__init__(name, actor)

    def initialise(self):
        """
        Set up initial data structure
        """
        super(SPECDataCollector, self).initialise()

    def update(self):
        """
        Collect data for all vehicles in the scene
        """
        new_status = py_trees.common.Status.RUNNING
        
        # Get current game time
        game_time = GameTime.get_time()
        
        # Identify ego vehicle (the main actor)
        ego_id = self._actor.id if self._actor else None
        
        # Get all vehicles in the scene
        vehicles = CarlaDataProvider.get_all_actors().filter('vehicle.*')
        
        # Get ego vehicle location directly from self._actor
        ego_location = None
        ego_waypoint = None
        if self._actor:
            ego_location = self._actor.get_location()
            # Get ego vehicle's waypoint for road border calculation
            ego_waypoint = CarlaDataProvider.get_map().get_waypoint(ego_location)
            
            # Calculate road borders if we have a valid ego waypoint
            if ego_waypoint:
                try:
                    # Find leftmost and rightmost lanes
                    leftmost_lane = find_leftmost_lane(ego_waypoint)
                    rightmost_lane = find_rightmost_lane(ego_waypoint)
                    
                    # Get lane border points
                    left_border_outer, _ = get_lane_border_points(leftmost_lane)
                    _, right_border_outer = get_lane_border_points(rightmost_lane)
                    
                    # Store road borders for visualization and SEE calculation
                    self.road_borders = {
                        "left_border": left_border_outer,
                        "right_border": right_border_outer
                    }
                    
                    # print(f"Road borders calculated - Left: ({left_border_outer.x:.2f}, {left_border_outer.y:.2f}), "
                    #       f"Right: ({right_border_outer.x:.2f}, {right_border_outer.y:.2f})")
                    
                except Exception as e:
                    print(f"Error calculating road borders: {e}")
        
        for vehicle in vehicles:
            # Skip vehicles that are too far from ego based on distance thresholds
            if ego_location is not None:
                # Calculate relative position to ego vehicle
                vehicle_location = vehicle.get_location()
                rel_x = abs(vehicle_location.x - ego_location.x)
                rel_y = abs(vehicle_location.y - ego_location.y)
                rel_z = abs(vehicle_location.z - ego_location.z)
                
                # Filter out vehicles that are too far away
                if rel_x > self.ly*0.7 or rel_y > self.lx*0.7 or rel_z > 2:
                    continue  # Skip this vehicle
            
            # Get vehicle data
            transform = vehicle.get_transform()
            velocity_vector = vehicle.get_velocity()
            control = vehicle.get_control()
            
            # Get vehicle's waypoint to determine lane ID
            waypoint = CarlaDataProvider.get_map().get_waypoint(transform.location)
            lane_id = waypoint.lane_id if waypoint else -1
            
            # Add data for this vehicle and timestep
            self._data_structure["game_time"].append(game_time)
            self._data_structure["vehicle_id"].append(vehicle.id)
            self._data_structure["x"].append(round(float(transform.location.x), 2))
            self._data_structure["y"].append(round(float(transform.location.y), 2))
            self._data_structure["vx"].append(round(float(velocity_vector.x), 2))
            self._data_structure["vy"].append(round(float(velocity_vector.y), 2))
            self._data_structure["lane_id"].append(lane_id)
            self._data_structure["steering"].append(round(float(control.steer), 2))
            
            # Calculate rough acceleration (we could improve this with CDP data)
            try:
                accel = vehicle.get_acceleration()
                accel_magnitude = math.sqrt(accel.x ** 2 + accel.y ** 2)
            except:
                accel_magnitude = 0.0
                
            self._data_structure["acceleration"].append(round(float(accel_magnitude), 2))
            
            # Mark if this is the ego vehicle
            self._data_structure["is_ego"].append(vehicle.id == ego_id)
        
        # Get planning encoding from CarlaDataProvider
        planning_encoding = CarlaDataProvider.get_planning_encoding()
        
        # Store the planning encoding value for analysis and visualization
        if ego_id is not None:
            if "planning_encoding" not in self._data_structure:
                self._data_structure["planning_encoding"] = []
            
            # Add planning encoding value for this timestep
            # We'll only store it once per timestep, associated with the ego vehicle
            self._data_structure["planning_encoding"].append(planning_encoding)
            
            # Visualize planning encoding in CARLA window if enabled and not moving straight
            if self.visualize_planning_arrow and planning_encoding is not None and self._actor and abs(planning_encoding) >= 0.01:
                try:
                    world = CarlaDataProvider.get_world()
                    ego_transform = self._actor.get_transform()
                    ego_location = ego_transform.location
                    forward_vec = ego_transform.get_forward_vector()
                    right_vec = ego_transform.get_right_vector()

                    arrow_length = 4.0  # Length of the arrow
                    lateral_scale = 1.0  # How much the arrow deviates sideways
                    vertical_offset = 0.2 # Draw arrow slightly above the car

                    start_point = ego_location + carla.Location(z=vertical_offset)

                    # Calculate direction based on planning encoding
                    # Positive encoding -> turn right, Negative encoding -> turn left
                    direction_vec = forward_vec + right_vec * planning_encoding * lateral_scale
                    end_point = start_point + direction_vec.make_unit_vector() * arrow_length

                    # Choose color based on direction
                    color = carla.Color(r=0, g=0, b=255) # Blue for right (positive)
                    if planning_encoding < 0:
                        color = carla.Color(r=255, g=0, b=0) # Red for left (negative)
                    # No need for white color check now as we skip drawing if abs(planning_encoding) < 0.01
                    # elif abs(planning_encoding) < 0.01: # Nearly straight
                    #    color = carla.Color(r=255, g=255, b=255) # White for straight

                    world.debug.draw_arrow(
                        start_point,
                        end_point,
                        arrow_size=0.3,
                        thickness=0.4,
                        life_time=0.1, # Short lifetime to update each frame
                        color=color
                    )
                except Exception as e:
                    print(f"Error drawing planning encoding arrow: {e}")
        
        # Calculate SEE encoding using the latest collected data
        try:
            # Prepare road borders for SEE calculation if available
            road_borders = None
            if self.road_borders["left_border"] and self.road_borders["right_border"]:
                # Convert road borders to the format expected by compute_see_carla
                left_border = (self.road_borders["left_border"].x, self.road_borders["left_border"].y)
                right_border = (self.road_borders["right_border"].x, self.road_borders["right_border"].y)
                
                # Create a simple road boundary with start and end points
                # For simplicity, we'll create points based on ego vehicle's forward vector
                if ego_waypoint:
                    forward_vector = ego_waypoint.transform.get_forward_vector()
                    forward_length = 30.0  # Length to extend road borders forward/backward
                    
                    # Calculate start and end points for left border
                    left_start = (
                        left_border[0] - forward_vector.x * forward_length,
                        left_border[1] - forward_vector.y * forward_length
                    )
                    left_end = (
                        left_border[0] + forward_vector.x * forward_length,
                        left_border[1] + forward_vector.y * forward_length
                    )
                    
                    # Calculate start and end points for right border
                    right_start = (
                        right_border[0] - forward_vector.x * forward_length,
                        right_border[1] - forward_vector.y * forward_length
                    )
                    right_end = (
                        right_border[0] + forward_vector.x * forward_length,
                        right_border[1] + forward_vector.y * forward_length
                    )
                    
                    # Create road borders in the expected format: [[start, end], [start, end]]
                    road_borders = [
                        [left_start, left_end],  # Left border line [start, end]
                        [right_start, right_end]  # Right border line [start, end]
                    ]
            
            # Use compute_see_carla to calculate the SEE matrix with road borders
            see_matrix, points = compute_see_carla(
                lx=self.lx,
                ly=self.ly, 
                nrad=self.nrad,
                nring=self.nring,
                collected_data=self._data_structure,
                ego_id=ego_id,
                road_borders=road_borders  # Add road borders to the function call
            )
            
            # Debug printing
            # print("\n===== SEE Matrix (Timestep: {}) =====".format(game_time))
            # print(see_matrix)
            # print("Number of points considered: {}".format(len(points)))
            
            # Visualize using the same approach as in run_highway_scenario.py
            # self.visualize_results(see_matrix, points, game_time)
            
        except Exception as e:
            print(f"Error calculating SEE encoding: {e}")

        # Calculate DSEE/TTR using compute_dsee_carla
        try:
            # Calculate TTR using the same collected data
            ttr_value = compute_dsee_carla(
                collected_data=self._data_structure,
                ego_id=ego_id,
                commonRoadHandler=self._commonroad_handler
            )
            
            # Store TTR value in data structure
            self._data_structure["ttr"].append(ttr_value)
            
            # Debug print the TTR value
            # if ttr_value < 5: # For debugging, only print TTR value if it's less than 5
            print(f"===== TTR Value at time {game_time:.2f}: {ttr_value:.2f} =====")
            
        except Exception as e:
            print(f"Error calculating TTR/DSEE value: {e}")
            # Add a placeholder value to maintain data structure alignment
            self._data_structure["ttr"].append(None)

        return new_status

    def visualize_results(self, see_matrix, points, game_time):
        """
        Visualize SEE matrix and points using the same approach as in run_highway_scenario.py
        
        Args:
            see_matrix (numpy.ndarray): The SEE matrix to visualize
            points (list): List of (x,y) coordinates
            game_time (float): Current simulation time
        """
        try:
            # Visualize see_matrix using pyplot - similar to run_highway_scenario.py
            plt.figure('SEE Matrix Visualization')
            plt.clf()  # Clear the figure
            
            # Display the SEE matrix as a heatmap
            plt.imshow(see_matrix, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='SEE Value')
            plt.title(f'SEE Matrix - Time: {game_time:.2f}s')
            
            # Add axis labels and grid
            plt.xlabel('Ring Index')
            plt.ylabel('Radial Index')
            plt.grid(False)
            
            # If points are available, visualize them on a separate plot
            if points is not None and len(points) > 0:
                plt.figure('SEE Points')
                plt.clf()
                
                # Convert points to numpy array for easier handling
                points_array = np.array(points)
                
                # Plot points (red dots for vehicles/boundaries)
                plt.scatter(points_array[:, 0], points_array[:, 1], c='red', marker='o', s=30)
                
                # Add ego vehicle marker at the center (green)
                plt.scatter(0, 0, c='green', marker='o', s=50)
                
                # Add perception area boundary
                plt.plot([-self.lx/2, self.lx/2, self.lx/2, -self.lx/2, -self.lx/2],
                         [-self.ly/2, -self.ly/2, self.ly/2, self.ly/2, -self.ly/2],
                         'b--', alpha=0.5)
                
                plt.title(f'SEE Points Distribution - Time: {game_time:.2f}s')
                plt.xlabel('X (relative to ego)')
                plt.ylabel('Y (relative to ego)')
                plt.grid(True)
                
                # Set equal aspect ratio for proper visualization
                # plt.axis('equal')
                
                # Set reasonable limits based on perception area
                limit_x = max(self.lx, 30)  # Use at least 30m for visibility
                limit_y = max(self.ly, 15)
                plt.xlim(-limit_x/2, limit_x/2)
                plt.ylim(-limit_y/2, limit_y/2)

            # Get the planning encoding from CarlaDataProvider instead of calculating it
            planning_encoding = CarlaDataProvider.get_planning_encoding()
            
            # You can add visualization of planning encoding here if needed
            # For example, adding text to a plot or a new plot
            if planning_encoding is not None:
                plt.figure('Planning Encoding')
                plt.clf()
                
                # Create a simple horizontal bar representing the planning encoding value
                plt.barh(['Planning Direction'], [planning_encoding], color='blue' if planning_encoding > 0 else 'red')
                plt.xlim(-1.1, 1.1)  # Set limits to match planning encoding range
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)  # Center line
                
                # Add labels
                plt.title(f'Planning Encoding - Time: {game_time:.2f}s')
                plt.xlabel('Left (-1) <--> Right (1)')
                
                # Add text showing the exact value
                plt.text(planning_encoding, 0, f'{planning_encoding:.2f}', 
                         ha='center', va='center', fontweight='bold')
                
            # Update all visualizations
            plt.pause(0.001)
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            
        return planning_encoding

    def terminate(self, new_status):
        """
        Process collected data when the scenario is over
        """
        if self.finished:
            return

        self.finished = True
        
        # Save data to CSV
        self._save_result_to_csv()
        
        # Close all open figures
        plt.close('all')
        
        super(SPECDataCollector, self).terminate(new_status)

    def _save_result_to_csv(self):
        """
        Save collected data to a CSV file
        """
        csv_filename = f"SPEC_data_{self.task_id}.csv"
        
        with open(csv_filename, "w", newline='') as file:
            header = self._data_structure.keys()
            writer = csv.writer(file)
            writer.writerow(header)
            
            # Get number of data points collected
            num_rows = len(self._data_structure["game_time"])
            
            # Write all rows to CSV
            for i in range(num_rows):
                row = []
                for item in header:
                    try:
                        row.append(self._data_structure[item][i])
                    except:
                        row.append(None)  # Handle missing data
                writer.writerow(row)
                
        print(f"SPEC data collection saved to {csv_filename}")
