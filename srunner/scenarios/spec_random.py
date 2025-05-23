#!/usr/bin/env python

# Copyright (c) 2018-2023 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import py_trees
import random
import numpy as np
import datetime
import logging
import os

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
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
from srunner.scenariomanager.scenarioatomics.atomic_spec_behaviors import (
    SPECDataCollector,
)
from srunner.scenariomanager.scenarioatomics.atomic_osg_behaviors import (
    OASDataCollector,
)
from src.utils.db_helper import DBHelper
from src.data_process.hsr_calculation import (
    compute_dsee_carla,
    grid_to_decimal,
    compute_see_carla,
)
from src.utils.common import get_segmented_value
from src.utils.common import generate_random_name_string
from src.configs.environment_configurations import SPECConfig

# Set up logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)

# Add console handler to see logs in terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
_log.addHandler(console_handler)

# Add file handler to write logs to a file
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "spec_random.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
_log.addHandler(file_handler)

# Configuration for HSR calculations
SPEC_CONF = SPECConfig(lx=10)

# Database helper (shared)
try:
    _db_helper = DBHelper()
    _db = _db_helper.get_db()
except Exception as _e:
    print(f"Warning: Failed to connect to MongoDB: {_e}")
    _db = None

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


def create_vehicle_and_move_underground(waypoint, max_retries=3):
    """
    Creates a vehicle at the given waypoint and moves it underground for preparation
    Returns the vehicle if successful, None otherwise
    
    Args:
        waypoint: The waypoint to spawn the vehicle at
        max_retries: Maximum number of spawn attempts with different vehicle types
    """
    tf = waypoint.transform
    vehicle = None
    
    # List of vehicle blueprints to try if the generic one fails
    # Start with the generic filter
    vehicle_filters = [
        "vehicle.*",  # Try generic first
        "vehicle.audi.*",  # Then try specific manufacturers
        "vehicle.tesla.*",
        "vehicle.volkswagen.*",
        "vehicle.ford.*",
        "vehicle.bmw.*",
        "vehicle.mercedes.*",
        "vehicle.toyota.*",
        "vehicle.nissan.*"
    ]
    
    # Try spawning with different filters if needed
    for retry in range(max_retries):
        # Pick a filter based on retry count
        filter_idx = min(retry, len(vehicle_filters) - 1)
        current_filter = vehicle_filters[filter_idx]
        
        # Try to spawn the vehicle
        vehicle = CarlaDataProvider.request_new_actor(
            current_filter,
            tf,
            rolename="scenario",
            attribute_filter={"base_type": "car", "has_lights": True},
        )
        
        # If successful, break out of the loop
        if vehicle is not None:
            print(f"Successfully spawned vehicle using filter {current_filter}")
            break
        else:
            print(f"WARNING: Failed to spawn vehicle with filter {current_filter} at {tf.location}")
    
    # If all retries failed, return None
    if vehicle is None:
        print(f"ERROR: Failed to spawn vehicle after {max_retries} attempts at {tf.location}")
        return None
        
    # Move below ground
    vehicle.set_location(tf.location - carla.Location(z=100))
    vehicle.set_simulate_physics(False)

    # This starts the engine, to allow the vehicle to instantly move
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
    return vehicle


def find_non_overlapping_waypoints(reference_waypoint, num_waypoints, min_distance=-40, max_distance=40, max_attempts=100):
    """
    Find waypoints that don't overlap with each other, both behind and ahead of the reference waypoint
    
    Args:
        reference_waypoint: The starting reference waypoint
        num_waypoints: Number of waypoints to find
        min_distance: Minimum distance from reference waypoint (negative for behind)
        max_distance: Maximum distance from reference waypoint (positive for ahead)
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
        
        # Select a random distance within range (can be negative for behind)
        distance = random.uniform(min_distance, max_distance)
        
        # Get the waypoint at that distance (handle both directions)
        candidate = None
        if distance >= 0:
            next_waypoints = selected_lane.next(distance)
            if next_waypoints:
                candidate = next_waypoints[0]
        else:
            # For negative distance, use previous
            prev_waypoints = selected_lane.previous(abs(distance))
            if prev_waypoints:
                candidate = prev_waypoints[0]
        
        # Skip if couldn't get a valid waypoint
        if candidate is None:
            attempts += 1
            continue
        
        # Check if it overlaps with existing waypoints on the same lane
        is_overlapping = False
        for existing in waypoints:
            # Only check distance for waypoints on the same lane
            if existing.lane_id == candidate.lane_id and existing.transform.location.distance(candidate.transform.location) < 10.0:
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
        
        # Get agent name for collection naming (default to "aba")
        self._agent_name = get_value_parameter(config, "agent_name", str, "aba")
        
        # Set up database collection
        self._collection = None
        if _db is not None:
            collection_name = f"{self._agent_name}_iss"
            self._collection = _db[collection_name]
            print(f"SPEC_Random: Using collection '{collection_name}' for data storage")
        else:
            print("SPEC_Random: No database connection available")
        
        # Get random seed if provided
        self._random_seed = get_value_parameter(config, "random_seed", int, None)
        if self._random_seed is not None:
            random.seed(self._random_seed)
            print(f"SPEC_Random: Using random seed {self._random_seed}")
        else:
            print("SPEC_Random: Using random seed from system time.")
            
        # Generate a random save_name for this scenario run
        self._save_name = generate_random_name_string()
        print(f"SPEC_Random: Using randomly generated save_name: {self._save_name}")

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
        Custom initialization of actors with enhanced error handling
        """
        # Find non-overlapping starting positions randomly
        print("SPEC_Random: Generating random start positions.")
        self._start_waypoints = find_non_overlapping_waypoints(
            self._reference_waypoint,
            self._num_vehicles,
            min_distance=-40,  # 40m behind
            max_distance=40    # 40m ahead
        )

        # Adjust the number of vehicles if fewer waypoints were found
        original_num_vehicles = self._num_vehicles
        self._num_vehicles = min(self._num_vehicles, len(self._start_waypoints))
        if self._num_vehicles < original_num_vehicles:
            print(f"Warning: Could only find {self._num_vehicles} non-overlapping waypoints. Adjusted number of vehicles.")

        # Ensure we have start waypoints before finding destinations
        if not self._start_waypoints:
             print("Error: No start waypoints available. Cannot proceed.")
             return

        # Find destination waypoints based on the final list of start waypoints
        self._destination_waypoints = find_destination_waypoints(self._start_waypoints, self._end_waypoint)
        # Print every destination waypoint's location for debugging
        for i, waypoint in enumerate(self._destination_waypoints):
            print(f"Destination waypoint {i}: {waypoint.transform.location}")
        
        # Create vehicles and move them underground with enhanced error handling
        successful_vehicles = []
        successful_start_waypoints = []
        successful_destination_waypoints = []
        
        for i in range(self._num_vehicles):
            vehicle = create_vehicle_and_move_underground(self._start_waypoints[i])
            if vehicle is not None:
                successful_vehicles.append(vehicle)
                successful_start_waypoints.append(self._start_waypoints[i])
                successful_destination_waypoints.append(self._destination_waypoints[i])
                self.other_actors.append(vehicle)
        
        # Update lists with only successful spawns
        self._vehicles = successful_vehicles
        self._start_waypoints = successful_start_waypoints
        self._destination_waypoints = successful_destination_waypoints
        self._num_vehicles = len(self._vehicles)
        
        print(f"Successfully spawned {self._num_vehicles} vehicles out of {original_num_vehicles} requested")

    def _create_behavior(self):
        """
        Create behavior tree for the scenario
        """
        # Root sequence
        root = py_trees.composites.Parallel(
            "Main Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )
        
        # Add the data collector to the root for database logging
        root.add_child(RuntimeDataCollector(
            "RandomDataCollector", 
            self.ego_vehicles[0],
            collection=self._collection, 
            agent_name=self._agent_name, 
            save_name=self._save_name
        ))
        
        # Check if we have any vehicles to work with
        if not self._vehicles:
            _log.warning("No vehicles were successfully spawned. Creating minimal behavior tree.")
            # Add a dummy sequence that always succeeds
            dummy = py_trees.composites.Sequence("DummyBehavior")
            dummy.add_child(py_trees.behaviours.Success("DummySuccess"))
            root.add_child(dummy)
            return root
        
        behavior = py_trees.composites.Sequence("RandomVehicleBehavior")
        root.add_child(behavior)
        
        # Set vehicles to their starting points
        for i, vehicle in enumerate(self._vehicles):
            # Use the waypoint's transform
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


class RuntimeDataCollector(py_trees.behaviour.Behaviour):
    """
    Behavior that collects data and stores it in MongoDB for random scenarios.
    This is a simplified version without perturbation logic.
    """
    def __init__(self, name="RandomDataCollector", ego_vehicle=None,
                 collection=None, agent_name="aba", save_name="carla_random"):
        super(RuntimeDataCollector, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.spec_data_collector = None
        # DB reference
        self.collection = collection
        self.agent_name = agent_name
        # Keep track of last stored time to avoid duplicate inserts
        self._last_saved_time = -1.0
        # Save name to identify this scenario run in the database
        self.save_name = save_name
        # Track start time for total_wall_time calculation
        self._start_time = datetime.datetime.now()
        # Collision tracking
        self._last_collision_time = -1.0
        self._collision_data = None
        self._collision_sensor = None
        
    def setup(self, timeout=10):
        """
        Setup the behavior
        """
        # Create the actual data collector that will calculate SEE
        if self.ego_vehicle:
            self.spec_data_collector = SPECDataCollector(
                actor=self.ego_vehicle,
                name="SPECCollector",
                visualize_planning_arrow=False,  # Disable visualization to reduce overhead
                output_mode=None
            )
            self.spec_data_collector.setup(timeout)
            
            # Set up collision sensor
            self._setup_collision_sensor()
            
        return True
    
    def _setup_collision_sensor(self):
        """Set up a collision sensor on the ego vehicle to track collisions."""
        try:
            world = self.ego_vehicle.get_world()
            blueprint = world.get_blueprint_library().find('sensor.other.collision')
            transform = carla.Transform(carla.Location(x=0.0, z=0.0))
            self._collision_sensor = world.spawn_actor(blueprint, transform, attach_to=self.ego_vehicle)
            self._collision_sensor.listen(lambda event: self._on_collision(event))
            _log.debug("Collision sensor attached to ego vehicle")
        except Exception as e:
            _log.debug(f"Failed to setup collision sensor: {e}")
            self._collision_sensor = None
    
    def _on_collision(self, event):
        """Callback for collision events."""
        try:
            # Get current time to check if this is a new collision
            current_time = GameTime.get_time()
            
            # Avoid processing the same collision multiple times (within 0.5 second window)
            if abs(current_time - self._last_collision_time) < 0.5:
                return
                
            # Store collision time
            self._last_collision_time = current_time
            
            # Get the other actor involved in the collision
            other_actor = event.other_actor
            
            # Check if this is a collision with a static object
            is_static_object = False
            if other_actor:
                # Get the type ID
                type_id = other_actor.type_id.lower()
                # Check for static objects directly
                if 'static' in type_id:
                    is_static_object = True
                # Check for common static objects that might not have 'static' in their name
                static_keywords = ['guardrail', 'fence', 'barrier', 'wall', 'pole', 
                                  'traffic.stop', 'traffic.light', 'traffic.sign']
                for keyword in static_keywords:
                    if keyword in type_id:
                        is_static_object = True
                        break
            
            if is_static_object:
                _log.warning(f"Collision with static object: {other_actor.type_id}")
                # Don't store collision data for static objects, but still end scenario
                return
            
            # Store velocities for vehicle collisions
            if self.ego_vehicle and other_actor:
                ego_velocity = self.ego_vehicle.get_velocity()
                other_velocity = other_actor.get_velocity()
                
                # Convert to 2D vectors in the plane
                ego_vel_vector = [ego_velocity.x, ego_velocity.y]
                other_vel_vector = [other_velocity.x, other_velocity.y]
                
                # Calculate magnitudes
                ego_mag = np.round(np.linalg.norm(ego_vel_vector), 4)
                other_mag = np.round(np.linalg.norm(other_vel_vector), 4)
                
                # Calculate angle of incidence (in degrees)
                try:
                    from math import atan2, degrees
                    angle_of_incidence = np.round(
                        degrees(atan2(other_velocity.y, other_velocity.x) - 
                                atan2(ego_velocity.y, ego_velocity.x)), 4
                    )
                except Exception as e:
                    _log.debug(f"Failed to calculate angle of incidence: {e}")
                    angle_of_incidence = 0
                
                # Store collision data for later use
                self._collision_data = {
                    "ego_velocity_magnitude": ego_mag,
                    "incident_vehicle_velocity_magnitude": other_mag,
                    "angle_of_incident": angle_of_incidence,
                    "incident_vehicle_type_id": other_actor.type_id,
                    "collided": True
                }
                
                _log.debug(f"Collision detected - Ego velocity: {ego_mag}, Incident velocity: {other_mag}, Angle: {angle_of_incidence}, Type: {other_actor.type_id}")
                
                # Check if the other actor is a vehicle we need to remove
                try:
                    # Make sure it's a vehicle (not a static object, pedestrian, etc.)
                    if 'vehicle' in other_actor.type_id.lower():
                        other_actor_id = other_actor.id
                        _log.debug(f"Collision with vehicle ID {other_actor_id} - removing from scenario")
                        self._remove_npc_vehicle(other_actor)
                except Exception as e:
                    _log.debug(f"Error removing vehicle after collision: {e}")
            
        except Exception as e:
            _log.debug(f"Error processing collision: {e}")
        
    def _remove_npc_vehicle(self, vehicle):
        """Remove an NPC vehicle from the scenario after a collision"""
        try:
            vehicle_id = vehicle.id
            
            # Update the parent scenario class data structures, if we can access them
            # Try to find the parent scenario
            parent_scenario = None
            
            # Method 1: Try to get from CarlaDataProvider if the method exists
            if hasattr(CarlaDataProvider, 'get_running_scenario'):
                parent_scenario = CarlaDataProvider.get_running_scenario()
            
            # Method 2: Use our parent tree to find a SPEC_Random parent
            if parent_scenario is None and hasattr(self, 'parent'):
                node = self.parent
                while node is not None:
                    if isinstance(node, SPEC_Random):
                        parent_scenario = node
                        break
                    if hasattr(node, 'parent'):
                        node = node.parent
                    else:
                        break
            
            if parent_scenario and hasattr(parent_scenario, '_vehicles'):
                try:
                    # Find the index of the vehicle in the vehicles list
                    vehicle_index = -1
                    for i, v in enumerate(parent_scenario._vehicles):
                        if v.id == vehicle_id:
                            vehicle_index = i
                            break
                    
                    # Remove from the scenario's vehicle list if found
                    if vehicle_index >= 0:
                        # Remove the vehicle from the list
                        parent_scenario._vehicles.pop(vehicle_index)
                        _log.debug(f"Removed vehicle {vehicle_id} from scenario's vehicle list at index {vehicle_index}")
                        
                        # Remove corresponding waypoints if they exist and have matching indices
                        if hasattr(parent_scenario, '_start_waypoints') and len(parent_scenario._start_waypoints) > vehicle_index:
                            parent_scenario._start_waypoints.pop(vehicle_index)
                            _log.debug(f"Removed start waypoint for vehicle {vehicle_id}")
                            
                        if hasattr(parent_scenario, '_destination_waypoints') and len(parent_scenario._destination_waypoints) > vehicle_index:
                            parent_scenario._destination_waypoints.pop(vehicle_index)
                            _log.debug(f"Removed destination waypoint for vehicle {vehicle_id}")
                            
                        # Update the scenario's vehicle count
                        if hasattr(parent_scenario, '_num_vehicles'):
                            parent_scenario._num_vehicles -= 1
                            _log.debug(f"Updated scenario vehicle count to {parent_scenario._num_vehicles}")
                except Exception as e:
                    _log.debug(f"Error updating scenario data: {e}")
            
            # Move the vehicle underground to hide it
            try:
                transform = vehicle.get_transform()
                new_transform = carla.Transform(
                    carla.Location(x=transform.location.x, y=transform.location.y, z=-100),
                    transform.rotation
                )
                vehicle.set_transform(new_transform)
                vehicle.set_simulate_physics(False)  # Disable physics
                _log.debug(f"Moved vehicle {vehicle_id} underground and disabled physics")
                
                # Remove from other_actors list in the scenario
                if parent_scenario and hasattr(parent_scenario, 'other_actors'):
                    if vehicle in parent_scenario.other_actors:
                        parent_scenario.other_actors.remove(vehicle)
                        _log.debug(f"Removed vehicle {vehicle_id} from scenario's other_actors list")
                
                # Actually destroy the vehicle
                vehicle.destroy()
                _log.debug(f"Destroyed vehicle {vehicle_id}")
                
            except Exception as e:
                _log.debug(f"Error moving/destroying vehicle: {e}")
                
        except Exception as e:
            _log.debug(f"Error in _remove_npc_vehicle: {e}")

    def initialise(self):
        """
        Initialize the behavior
        """
        if self.spec_data_collector:
            self.spec_data_collector.initialise()
        
    def update(self):
        """
        Update the behavior. This is called at each tick.
        """
        if self.spec_data_collector:
            # Update the underlying data collector
            self.spec_data_collector.update()
            
            # Get the SEE encoding and process storage
            see_encoding = self.spec_data_collector.get_see_encoding()
            if see_encoding is not None:
                # Store data in database
                self._handle_data_storage(see_encoding)

        return py_trees.common.Status.RUNNING
        
    def terminate(self, new_status):
        """
        Terminate the behavior
        """
        if self.spec_data_collector:
            self.spec_data_collector.terminate(new_status)
            
        # Clean up collision sensor
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None

    def _handle_data_storage(self, see_encoding):
        """Store data in MongoDB collection"""
        if self.collection is None:
            # DB not available
            return
        try:
            current_time = GameTime.get_time()
            # Avoid duplicate storage within the same timestep
            if abs(current_time - self._last_saved_time) < 1e-3:
                return
            data_struct = self.spec_data_collector._data_structure
            # Identify indices for current timestep
            latest_indices = [i for i, t in enumerate(data_struct["game_time"]) if abs(t - current_time) < 1e-3]
            if not latest_indices:
                return
            # Determine ego info first
            ego_index = None
            for idx in latest_indices:
                if data_struct["is_ego"][idx]:
                    ego_index = idx
                    break
            if ego_index is None:
                return
            ego_x = data_struct["x"][ego_index]
            ego_y = data_struct["y"][ego_index]
            ego_vx = data_struct["vx"][ego_index]
            ego_vy = data_struct["vy"][ego_index]
            ego_lane = data_struct["lane_id"][ego_index]
            ego_steer = data_struct["steering"][ego_index]
            ego_acc = data_struct["acceleration"][ego_index]
            # Build observation matrix (11 x 9)
            obs_rows = []
            # Ego row
            obs_rows.append([1, 0, 0, ego_vx, ego_vy, 0, ego_lane, ego_steer, ego_acc])
            # NPC rows
            for idx in latest_indices:
                if idx == ego_index:
                    continue
                rel_x = data_struct["x"][idx] - ego_x
                rel_y = data_struct["y"][idx] - ego_y
                obs_rows.append([
                    1,
                    rel_x,
                    rel_y,
                    data_struct["vx"][idx],
                    data_struct["vy"][idx],
                    0,
                    data_struct["lane_id"][idx],
                    data_struct["steering"][idx],
                    data_struct["acceleration"][idx],
                ])
                if len(obs_rows) >= 11:  # Limit to 11 vehicles total (ego + 10 npc)
                    break
            # Pad if fewer than 11 rows
            while len(obs_rows) < 11:
                obs_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
            obs_array = np.array(obs_rows)
            
            # Get the actual planning type directly from CarlaDataProvider
            planning_encoding = CarlaDataProvider.get_planning_encoding()
            
            # Convert to our -1, 0, 1 system:
            # -1: left lane change, 0: straight, 1: right lane change
            planning_type_real = 0  # Default to straight (0)
            if planning_encoding < 0:
                planning_type_real = -1  # Left lane change
            elif planning_encoding > 0:
                planning_type_real = 1   # Right lane change
            
            # Compute DSEE/TTR and segmentation
            try:
                ttr_real = compute_dsee_carla(data_struct)
            except Exception as _e:
                _log.debug(f"compute_dsee_carla failed: {_e}")
                ttr_real = float('inf')
            ttr_seg = get_segmented_value(ttr_real, SPEC_CONF.ttr_segments)
            # Grid decimal
            grid_decimal = grid_to_decimal(see_encoding)
            # Prepare DB document
            doc = {
                "created_at": datetime.datetime.utcnow(),
                "save_name": self.save_name,
                "sim_time": round(float(current_time), 4),
                "external_vehicles": len(obs_rows) - 1,
                "obs": obs_array.tolist(),
                "planning_type": planning_type_real,
                "ttr_real": ttr_real,
                "ttr_seg": ttr_seg,
                "grid_decimal": grid_decimal,
                "ego_position": [round(ego_x, 4), round(ego_y, 4)],
                "ego_velocity": [round(ego_vx, 4), round(ego_vy, 4)],
                "total_wall_time": (datetime.datetime.now() - self._start_time).total_seconds(),
                "total_simulated_time": round(float(current_time), 4),
            }
            
            # Add collision data if a collision has been detected
            if self._collision_data:
                # Add collision data to document
                doc.update(self._collision_data)
                _log.debug(f"Adding collision data to document: {self._collision_data}")
                # Reset collision data after it's been recorded
                self._collision_data = None
                
            self.collection.insert_one(doc)
            # Update last saved time
            self._last_saved_time = current_time
        except Exception as e:
            _log.debug(f"Data storage error: {e}")
