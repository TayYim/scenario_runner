#! python3
# -*- encoding: utf-8 -*-

import math
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

    def __init__(self, actor=None, name="SPECDataCollector"):
        """
        Setup for SPEC data collection
        """
        self.task_id = name + "_" + datetime.datetime.now().strftime("%m%d%H%M%S")
        self.finished = False  # Use to mark whether the terminate function has been executed
        
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
            "is_ego": []
        }
        
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
        
        for vehicle in vehicles:
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

        return new_status

    def terminate(self, new_status):
        """
        Process collected data when the scenario is over
        """
        if self.finished:
            return

        self.finished = True
        
        # Save data to CSV
        self._save_result_to_csv()
        
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
