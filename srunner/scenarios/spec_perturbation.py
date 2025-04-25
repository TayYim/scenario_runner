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

# Comments explaining the setup
"""
This scenario implements a traffic scene with SEE-based perturbations of vehicle controls.

Requirements to run this scenario:
1. The Carla simulator must be running
2. The SPECDataCollector class must be properly set up to provide SEE encoding data
3. To use perturbations, the SEEPerturbationManager singleton manages perturbations for all vehicles
4. Each vehicle uses PerturbedAgentBehavior instead of BasicAgentBehavior to allow control perturbation

The workflow is:
1. SEEDataCollector captures the current SEE encoding in each time step
2. SEEPerturbationManager uses this SEE encoding to calculate vehicle-specific perturbations
3. PerturbedAgentBehavior intercepts and modifies vehicle controls based on these perturbations
4. Modified controls are applied to vehicles, affecting their behavior

This implementation allows unique perturbations per vehicle based on the current SEE encoding.
"""

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


class SEEPerturbationManager:
    """
    Singleton class that manages SEE data and provides perturbation values for vehicles.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SEEPerturbationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._see_encoding = None
        self._vehicles = {}  # Dictionary to store vehicle ID -> vehicle object
        self._vehicle_indices = {}  # Dictionary to map vehicle ID -> array index
        print("SEEPerturbationManager initialized")
    
    def register_vehicle(self, vehicle, index):
        """Register a vehicle with the manager"""
        vehicle_id = vehicle.id
        self._vehicles[vehicle_id] = vehicle
        self._vehicle_indices[vehicle_id] = index
        print(f"Registered vehicle {vehicle_id} with index {index}")
    
    def update_see_encoding(self, see_encoding):
        """Update the current SEE encoding"""
        self._see_encoding = see_encoding
        # Only log the first update and use a short representation
        if see_encoding is not None:
            see_shape = see_encoding.shape
            see_preview = str(see_encoding.flatten()[:3]) + "..." if see_encoding.size > 3 else str(see_encoding)
            print(f"SEE encoding updated: shape={see_shape}, preview={see_preview}")
    
    def get_vehicle_perturbation(self, vehicle_id):
        """
        Calculate perturbation for a specific vehicle based on SEE encoding
        Returns a tuple of (throttle_perturbation, steering_perturbation)
        
        This version creates more dramatic perturbations that are clearly visible
        during simulation.
        """
        if self._see_encoding is None:
            # Default perturbations if no SEE data available
            return (0.0, 0.0)
        
        try:
            # Get the vehicle's index and current time
            index = self._vehicle_indices.get(vehicle_id, 0)
            current_time = GameTime.get_time()
            
            # Create a time-based oscillation specific to this vehicle
            # This creates a unique pattern for each vehicle
            time_factor = np.sin(current_time * 0.5 + index * 0.7) * 0.5 + 0.5  # Range [0, 1]
            
            # Use SEE encoding to determine perturbation base values
            # Get row index based on vehicle index
            row_index = index % self._see_encoding.shape[0]
            
            # Extract more information from the SEE matrix
            # Use different aspects of the SEE encoding for different perturbation components
            see_row_sum = np.sum(self._see_encoding[row_index]) / self._see_encoding.shape[1]
            see_max_val = np.max(self._see_encoding[row_index])
            see_mean = np.mean(self._see_encoding)
            
            # Calculate more dramatic steering perturbation
            # Range approximately [-0.4, 0.4] - this is strong enough to be clearly visible
            base_steer = np.clip(see_row_sum * 1.0, -0.5, 0.5)
            
            # Add time-varying oscillation to steering
            # This makes vehicles weave more dramatically
            steer_oscillation = np.sin(current_time * (1.0 + index * 0.1)) * 0.15
            steering_perturbation = base_steer + steer_oscillation
            
            # Also oscillate throttle for speed variations
            # Throttle oscillates between moderate acceleration and slight deceleration
            throttle_base = see_max_val * 0.3
            throttle_oscillation = np.cos(current_time * 0.7 + index) * 0.15
            throttle_perturbation = throttle_base + throttle_oscillation
            
            # Add some SEE-based randomness to make behavior less predictable
            # This creates more chaotic, realistic-looking perturbations
            if see_mean > 0.1:
                noise_factor = 0.1
                steering_perturbation += np.random.normal(0, noise_factor * see_mean)
                throttle_perturbation += np.random.normal(0, noise_factor * see_mean)
            
            # Ensure values stay in valid ranges
            steering_perturbation = np.clip(steering_perturbation, -0.5, 0.5)
            throttle_perturbation = np.clip(throttle_perturbation, -0.3, 0.5)
            
            # Log significant perturbations for debugging
            if abs(steering_perturbation) > 0.2 or abs(throttle_perturbation) > 0.2:
                print(f"Strong perturbation for vehicle {vehicle_id}: steer={steering_perturbation:.2f}, throttle={throttle_perturbation:.2f}")
            
            return (float(throttle_perturbation), float(steering_perturbation))
            
        except Exception as e:
            print(f"Error calculating perturbation for vehicle {vehicle_id}: {e}")
            return (0.0, 0.0)  # Default in case of error


class SEEDataCollector(py_trees.behaviour.Behaviour):
    """
    Behavior that collects SEE data and updates the SEEPerturbationManager
    """
    def __init__(self, name="SEEDataCollector", ego_vehicle=None):
        super(SEEDataCollector, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.manager = SEEPerturbationManager()
        self.spec_data_collector = None
        
    def setup(self, timeout=10):
        """
        Setup the behavior
        """
        # Create the actual data collector that will calculate SEE
        if self.ego_vehicle:
            self.spec_data_collector = SPECDataCollector(
                actor=self.ego_vehicle,
                name="SPECCollector",
                visualize_planning_arrow=False  # Disable visualization to reduce overhead
            )
            self.spec_data_collector.setup(timeout)
        return True
        
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
            
            # Get the SEE encoding from the collector
            see_encoding = self.spec_data_collector.get_see_encoding()
            if see_encoding is not None:
                self.manager.update_see_encoding(see_encoding)
                print(f"Updated SEE encoding, shape: {see_encoding.shape}")
            
        return py_trees.common.Status.RUNNING
        
    def terminate(self, new_status):
        """
        Terminate the behavior
        """
        if self.spec_data_collector:
            self.spec_data_collector.terminate(new_status)


class PerturbedAgentBehavior(BasicAgentBehavior):
    """
    Extension of BasicAgentBehavior that perturbs control values based on SEE encoding
    """
    def __init__(self, vehicle, target_speed, target_location=None, plan=None, vehicle_index=0, name="PerturbedAgentBehavior"):
        """
        Initialize the behavior
        
        Args:
            vehicle: The vehicle to control
            target_speed: Target speed in km/h
            target_location: Target location (only specify this OR plan, not both)
            plan: Plan to follow (only specify this OR target_location, not both)
            vehicle_index: Index of the vehicle in the scenario
        """
        # Call parent constructor with only one of either target_location or plan
        super(PerturbedAgentBehavior, self).__init__(
            vehicle, 
            target_location=target_location, 
            plan=plan, 
            target_speed=target_speed
        )
        
        # Store additional data needed for perturbation
        self.manager = SEEPerturbationManager()
        self.vehicle_id = vehicle.id
        
        # Register the vehicle with the manager
        self.manager.register_vehicle(vehicle, vehicle_index)
        
    def update(self):
        """
        Update the behavior. This is called at each tick.
        
        This method temporarily disables the control application in the parent class,
        then gets the computed control, modifies it, and applies it manually.
        """
        # Save the original _apply_control method
        original_apply_control = self._actor.apply_control
        
        # Temporarily replace apply_control with a dummy method to prevent the parent class from applying control
        control_to_intercept = [None]  # Use a list to hold the reference
        
        def intercept_control(control):
            control_to_intercept[0] = control
        
        # Monkey patch the apply_control method
        self._actor.apply_control = intercept_control
        
        # Call the parent update method, which will call the intercepted apply_control
        status = super(PerturbedAgentBehavior, self).update()
        
        # Restore the original apply_control method
        self._actor.apply_control = original_apply_control
        
        # Get the control that was intercepted
        control = control_to_intercept[0]
        
        if control is not None:
            # Get perturbations for this vehicle based on SEE encoding
            throttle_pert, steering_pert = self.manager.get_vehicle_perturbation(self.vehicle_id)
            
            # Apply perturbations to control
            control.throttle = max(0.0, min(1.0, control.throttle + throttle_pert))
            control.steer = max(-1.0, min(1.0, control.steer + steering_pert))
            
            # Apply the modified control to the vehicle
            self._actor.apply_control(control)
            
            # Print debug information
            if abs(throttle_pert) > 0.01 or abs(steering_pert) > 0.01:
                print(f"Vehicle {self.vehicle_id}: Applied perturbations - throttle: {throttle_pert:.3f}, steering: {steering_pert:.3f}")
                print(f"Vehicle {self.vehicle_id}: Final controls - throttle: {control.throttle:.3f}, steering: {control.steer:.3f}")
        
        return status


class SPEC_Perturbation(BasicScenario):
    """
    This class implements a scenario that spawns random vehicles, moves them underground,
    and then activates them when the scenario is triggered, making them follow paths
    that will involve lane changes.
    
    This version includes SEE-based perturbations to vehicle controls.
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
        
        # Get random seed if provided
        self._random_seed = get_value_parameter(config, "random_seed", int, None)
        if self._random_seed is not None:
            random.seed(self._random_seed)
            print(f"SPEC_Perturbation: Using random seed {self._random_seed}")
        else:
            print("SPEC_Perturbation: Using random seed from system time.")

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
        
        # Initialize the route for the scenario
        self.route = None
        # If config has route information, use it
        if hasattr(config, 'route'):
            print("SPEC_Perturbation: Using route from config")
            self.route = config.route
        
        # Initialize the SEE perturbation manager
        self._perturbation_manager = SEEPerturbationManager()

        super().__init__(
            "SPEC_Perturbation",
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
        # Find non-overlapping starting positions randomly
        print("SPEC_Perturbation: Generating random start positions.")
        self._start_waypoints = find_non_overlapping_waypoints(
            self._reference_waypoint,
            self._num_vehicles,
            min_distance=10.0,
            max_distance=50.0
        )

        # Adjust the number of vehicles if fewer waypoints were found
        original_num_vehicles = self._num_vehicles
        self._num_vehicles = min(self._num_vehicles, len(self._start_waypoints))
        if self._num_vehicles < original_num_vehicles:
            print(f"Warning: Could only find {self._num_vehicles} non-overlapping waypoints. Adjusted number of vehicles.")


        # Ensure we have start waypoints before finding destinations
        if not self._start_waypoints:
             print("Error: No start waypoints available. Cannot proceed.")
             # Handle error appropriately, maybe raise exception or return early
             return # Or raise Exception("Failed to initialize start waypoints")


        # Find destination waypoints based on the final list of start waypoints
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
        
        # Add the SEE data collector to the root
        root.add_child(SEEDataCollector("SEECollector", self.ego_vehicles[0]))
        
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
            
            # Add driving behavior with perturbations
            speed = random.uniform(self._min_speed, self._max_speed) * 3.6  # Convert to km/h
            
            # Create a plan for the agent to follow - this is a better approach than using target_location
            # as it gives the agent more information about the route
            init_waypoint = self._start_waypoints[i]
            end_waypoint = self._destination_waypoints[i]
            
            # Get a clean plan - this avoids the destination/plan conflict
            vehicle_seq.add_child(
                PerturbedAgentBehavior(
                    vehicle,
                    target_speed=speed,
                    target_location=end_waypoint.transform.location,  # Using location, not waypoint
                    vehicle_index=i,
                    name=f"PerturbedBehavior_{i}"
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
