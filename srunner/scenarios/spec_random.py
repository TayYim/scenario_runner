#!/usr/bin/env python

# Copyright (c) 2018-2023 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import py_trees
import random
import numpy as np

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    SetInitSpeed,
    AccelerateToVelocity,
    BasicAgentBehavior,
    ActorDestroy,
    StopVehicle
)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance,
)
from srunner.scenariomanager.scenarioatomics.atomic_osg_behaviors import (
    OASDataCollector,
)


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]["value"])
    else:
        return default


def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]["from"]),
            p_type(config.other_parameters[name]["to"]),
        ]
    else:
        return default


def create_vehicle_and_move_underground(waypoint):
    """
    Creates a vehicle at the given waypoint and moves it underground for preparation
    """
    tf = waypoint.transform
    vehicle = CarlaDataProvider.request_new_actor(
        "vehicle.*",
        tf,
        rolename="scenario",
        attribute_filter={"base_type": "car", "has_lights": True},
    )
    # Move below ground
    vehicle.set_location(tf.location - carla.Location(z=100))
    vehicle.set_simulate_physics(False)

    # This starts the engine, to allow the vehicle to instantly move
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
    return vehicle


def find_non_overlapping_waypoints(reference_waypoint, num_waypoints, min_distance=10.0, max_distance=60.0, max_attempts=100):
    """
    Find waypoints that don't overlap with each other
    
    Args:
        reference_waypoint: The starting reference waypoint
        num_waypoints: Number of waypoints to find
        min_distance: Minimum distance between waypoints
        max_distance: Maximum distance from reference waypoint
        max_attempts: Maximum attempts to find suitable waypoints
    
    Returns:
        List of non-overlapping waypoints
    """
    waypoints = []
    attempts = 0
    
    # Get all available lanes at the reference position
    all_lanes = []
    current_waypoint = reference_waypoint
    
    # Add the reference lane
    all_lanes.append(current_waypoint)
    
    # Add left lanes
    left_waypoint = current_waypoint.get_left_lane()
    while left_waypoint and left_waypoint.lane_type == carla.LaneType.Driving:
        all_lanes.append(left_waypoint)
        left_waypoint = left_waypoint.get_left_lane()
    
    # Add right lanes
    right_waypoint = current_waypoint.get_right_lane()
    while right_waypoint and right_waypoint.lane_type == carla.LaneType.Driving:
        all_lanes.append(right_waypoint)
        right_waypoint = right_waypoint.get_right_lane()
    
    # Not enough lanes, we'll have to reuse lanes with different distances
    while len(waypoints) < num_waypoints and attempts < max_attempts:
        # Select a random lane
        if not all_lanes:
            break
        
        selected_lane = random.choice(all_lanes)
        
        # Select a random distance within range
        distance = random.uniform(min_distance, max_distance)
        
        # Get the waypoint at that distance
        candidate = selected_lane.next(distance)[0]
        
        # Check if it overlaps with existing waypoints on the same lane
        is_overlapping = False
        for existing in waypoints:
            # Only check distance for waypoints on the same lane
            if existing.lane_id == candidate.lane_id and existing.transform.location.distance(candidate.transform.location) < min_distance:
                is_overlapping = True
                break
        
        if not is_overlapping:
            waypoints.append(candidate)
        
        attempts += 1
    
    return waypoints


def find_destination_waypoints(start_waypoints, end_waypoint):
    """
    Find destination waypoints around the scenario's ending location
    
    Args:
        start_waypoints: List of starting waypoints
        end_waypoint: Waypoint representing the scenario's ending location
    
    Returns:
        List of destination waypoints (one for each starting waypoint)
    """
    destinations = []
    
    # Find all available lanes at the ending location
    available_lanes = []
    
    # Add the current lane
    available_lanes.append(end_waypoint)
    
    # Add left lanes
    left_wp = end_waypoint.get_left_lane()
    while left_wp and left_wp.lane_type == carla.LaneType.Driving:
        available_lanes.append(left_wp)
        left_wp = left_wp.get_left_lane()
    
    # Add right lanes
    right_wp = end_waypoint.get_right_lane()
    while right_wp and right_wp.lane_type == carla.LaneType.Driving:
        available_lanes.append(right_wp)
        right_wp = right_wp.get_right_lane()
    
    # If we don't have enough lanes for all vehicles, we'll reuse lanes
    for _ in start_waypoints:
        # Pick a random lane
        if available_lanes:
            destination = random.choice(available_lanes)
            
            # Optionally, apply a small random offset along the lane for more variation
            # This creates a spread of vehicles at the end point rather than all clumped together
            offset = random.uniform(-10.0, 10.0)
            if offset != 0:
                # Get next waypoint with offset (could be forward or backward)
                if offset > 0:
                    dest_waypoints = destination.next(offset)
                else:
                    dest_waypoints = destination.previous(abs(offset))
                
                if dest_waypoints:
                    destination = dest_waypoints[0]
            
            destinations.append(destination)
        else:
            # Fallback - should not happen if map is properly loaded
            destinations.append(end_waypoint)
    
    return destinations


class SPEC_Random(BasicScenario):
    """
    This class implements a scenario that spawns random vehicles, moves them underground,
    and then activates them when the scenario is triggered, making them follow paths
    that will involve lane changes.
    """

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        timeout=180,
    ):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        
        # Get parameters
        self._num_vehicles = get_value_parameter(config, "num_vehicles", int, 10)
        self._min_speed = get_value_parameter(config, "min_speed", float, 5)
        self._max_speed = get_value_parameter(config, "max_speed", float, 15)
        
        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        
        # Get ending point from parameters if available
        if "endding_point" in config.other_parameters:
            end_x = float(config.other_parameters["endding_point"]["x"])
            end_y = float(config.other_parameters["endding_point"]["y"])
            end_z = float(config.other_parameters["endding_point"]["z"])
            end_location = carla.Location(x=end_x, y=end_y, z=end_z)
            self._end_waypoint = self._map.get_waypoint(end_location)
        else:
            # Fallback: use a waypoint far ahead in the same road
            self._end_waypoint = self._reference_waypoint.next(675.0)[0]
        
        # Lists to store vehicles, starting waypoints, and destination waypoints
        self._vehicles = []
        self._start_waypoints = []
        self._destination_waypoints = []

        super().__init__(
            "SPEC_Random",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _initialize_actors(self, config):
        """
        Custom initialization of actors
        """
        # Find non-overlapping starting positions
        self._start_waypoints = find_non_overlapping_waypoints(
            self._reference_waypoint, 
            self._num_vehicles, 
            min_distance=10.0, 
            max_distance=50.0
        )
        
        # If we couldn't find enough waypoints, adjust the number of vehicles
        self._num_vehicles = min(self._num_vehicles, len(self._start_waypoints))
        
        # Find destination waypoints
        self._destination_waypoints = find_destination_waypoints(self._start_waypoints, self._end_waypoint)
        # Print every destination waypoint's location for debugging
        for i, waypoint in enumerate(self._destination_waypoints):
            print(f"Destination waypoint {i}: {waypoint.transform.location}")
        
        # Create vehicles and move them underground
        for i in range(self._num_vehicles):
            vehicle = create_vehicle_and_move_underground(self._start_waypoints[i])
            self._vehicles.append(vehicle)
            self.other_actors.append(vehicle)

    def _create_behavior(self):
        """
        Create behavior tree for the scenario
        """
        # Root sequence
        root = py_trees.composites.Parallel(
            "Main Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        
        behavior = py_trees.composites.Sequence("RandomVehicleBehavior")
        root.add_child(behavior)
        
        # Set vehicles to their starting points
        for i, vehicle in enumerate(self._vehicles):
            behavior.add_child(
                ActorTransformSetter(vehicle, self._start_waypoints[i].transform)
            )
        
        # Randomly assign speeds to each vehicle
        for i, vehicle in enumerate(self._vehicles):
            speed = random.uniform(self._min_speed, self._max_speed)
            behavior.add_child(SetInitSpeed(vehicle, speed))
        
        # Create a parallel behavior for all vehicles to move simultaneously
        npc_behaviors = py_trees.composites.Parallel(
            "NPC Behaviors", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL
        )
        
        # For each vehicle: create a sequence of drive -> hide
        for i, vehicle in enumerate(self._vehicles):
            # Create sequence for this vehicle
            vehicle_seq = py_trees.composites.Sequence(f"Vehicle_{i}_Sequence")
            
            # Add driving behavior
            speed = random.uniform(self._min_speed, self._max_speed)
            vehicle_seq.add_child(
                BasicAgentBehavior(
                    vehicle,
                    target_speed=speed * 3.6,  # Convert to km/h
                    target_location=self._destination_waypoints[i].transform.location,
                )
            )
            
            # After driving, stop the vehicle
            vehicle_seq.add_child(StopVehicle(vehicle, 1.0))
            
            # Move vehicle underground and disable physics
            underground_transform = carla.Transform(
                self._destination_waypoints[i].transform.location - carla.Location(z=100),
                self._destination_waypoints[i].transform.rotation
            )
            vehicle_seq.add_child(ActorTransformSetter(vehicle, underground_transform, physics=False))
            
            # Add this vehicle's sequence to the parallel behavior
            npc_behaviors.add_child(vehicle_seq)
        
        behavior.add_child(npc_behaviors)
        
        # We still keep the final cleanup for safety, though vehicles are already hidden
        for vehicle in self._vehicles:
            behavior.add_child(ActorDestroy(vehicle))
        
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return []
        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
