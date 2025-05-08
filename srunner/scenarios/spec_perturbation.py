#!/usr/bin/env python

# Copyright (c) 2018-2023 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import py_trees
import random
import numpy as np
import sys
import os
import datetime
import logging

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
from src.utils.common import calculate_next_status
from src.utils.common import generate_random_name_string
from src.configs.environment_configurations import SPECConfig
from src.configs.environment_configurations import COAXConfig
from src.coax.models.model import process_obs_norm, get_CNN_model
import torch

# Forward declaration to avoid circular imports
SPEC_Perturbation = None

# Planning model global initialization (shared among all scenarios)
SPEC_CONF = SPECConfig()
PLAN_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../src/coax/models/highway_plan_4m.pth",
)
# Resolve absolute path correctly
PLAN_MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                               "../../../../src/coax/models/highway_plan_4m.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    _plan_model = get_CNN_model(3)
    _plan_model.load_state_dict(torch.load(PLAN_MODEL_PATH, map_location=device))
    _plan_model.to(device)
    _plan_model.eval()
except Exception as _e:
    print(f"Warning: Failed to load planning model: {_e}")
    _plan_model = None

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
file_handler = logging.FileHandler(os.path.join(log_dir, "spec_perturbation.log"))
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
_log.addHandler(file_handler)

# Database helper (shared)
try:
    _db_helper = DBHelper()
    _db = _db_helper.get_db()
    _main_collection = _db["aba_coax"]
    _hsr_collection = _db["aba_hsr"]
except Exception as _e:
    print(f"Warning: Failed to connect to MongoDB: {_e}")
    _main_collection = None
    _hsr_collection = None

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


def find_non_overlapping_waypoints(reference_waypoint, num_waypoints, min_distance=-40.0, max_distance=40.0, max_attempts=100):
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


# ---------------------------------------------------------
# Perturbation manager providing coverage-driven action deltas
# ---------------------------------------------------------

class PerturbationManager:
    """
    Singleton class that manages data and provides perturbation values for vehicles.
    Provides coverage-based perturbation sampling looking for uncovered HSR states.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PerturbationManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._initialized = True
        self._vehicles = {}            # vehicle_id -> vehicle object
        self._vehicle_indices = {}     # vehicle_id -> index within obs array (1-based for NPC)
        self._vehicle_deltas = {}      # vehicle_id -> (throttle_delta, steer_delta)
        self._last_recalc_step = -1
        self._last_try_count = 0       # Track how many tries were needed in the last recalculation
        
        # Use COAXConfig for parameters with specified overrides
        coax_config = COAXConfig(
            gap_control=20,
            steering_sample_range=(-np.pi/6, np.pi/6),
        )
        self.gap_control = coax_config.gap_control  
        self.steer_range = coax_config.steering_sample_range  
        self.acc_range = coax_config.acceleration_sample_range  
        self.max_try = coax_config.max_try  
        self.dt_pred = self.gap_control // 2 

    def register_vehicle(self, vehicle, index):
        """Register a vehicle with the manager"""
        vehicle_id = vehicle.id
        self._vehicles[vehicle_id] = vehicle
        self._vehicle_indices[vehicle_id] = index
        print(f"Registered vehicle {vehicle_id} with index {index}")
    
    def remove_vehicle(self, vehicle_id):
        """Remove a vehicle from the manager"""
        if vehicle_id in self._vehicles:
            del self._vehicles[vehicle_id]
            _log.debug(f"Removed vehicle {vehicle_id} from manager vehicles")
        
        if vehicle_id in self._vehicle_indices:
            del self._vehicle_indices[vehicle_id]
            _log.debug(f"Removed vehicle {vehicle_id} from manager indices")
            
        if vehicle_id in self._vehicle_deltas:
            del self._vehicle_deltas[vehicle_id]
            _log.debug(f"Removed vehicle {vehicle_id} from manager deltas")
    
    def get_vehicle_perturbation(self, vehicle_id):
        # Return stored delta if already decided for this iteration
        if vehicle_id in self._vehicle_deltas:
            return self._vehicle_deltas[vehicle_id]

        # Fallback – if no delta computed yet, use zero
        return (0.0, 0.0)

    # -------------------------------------------------
    # Re-compute perturbations every gap_control steps
    # -------------------------------------------------
    def recalculate_perturbations(self, obs_array, spec_conf, hsr_collection, main_collection, plan_model, device, tick_counter):
        """Derive new (steer, throttle) deltas for all NPCs using full HSR coverage check."""
        try:
            # Always record this tick as processed
            self._last_recalc_step = tick_counter
            
            # If not a recalculation tick, don't continue with perturbation logic
            if tick_counter != 2 and tick_counter % self.gap_control != 0:
                return  # only recalc at intervals
                
            # Skip if no NPCs are present in obs_array
            if obs_array[1][0] == 0:
                self._last_try_count = 0  # No tries needed
                print("No NPCs present")
                return  # no NPCs present

            # Extract ego values (row 0)
            ego_row = obs_array[0]
            ego_v = np.linalg.norm(ego_row[3:5])
            ego_steer = ego_row[7]
            ego_acc = ego_row[8]
            
            # Extract ego position (needed for TTR calculation)
            ego_position = None
            if hasattr(self, '_vehicles') and len(self._vehicles) > 0:
                for veh_id, veh in self._vehicles.items():
                    idx = self._vehicle_indices.get(veh_id, -1)
                    if idx == 0:  # Ego vehicle has index 0
                        ego_position = veh.get_location()
                        break

            try_count = 0
            new_deltas = {}

            # Initialize with zero deltas for all vehicles
            for veh_id in self._vehicle_indices.keys():
                new_deltas[veh_id] = (0.0, 0.0)  # (acc_delta, steer_delta)

            while try_count <= self.max_try:
                # Make a copy of the original observation array
                pred_obs = obs_array.copy()
                
                # On first try, use zero perturbations
                # On subsequent tries, generate random perturbations for all NPCs
                if try_count == 0:
                    # First try with zero perturbations - keep pred_obs as is
                    pass
                else:
                    # Generate perturbations for all vehicles and calculate their predicted states
                    for veh_id, idx in list(self._vehicle_indices.items()):  # Use list() to avoid dictionary changed during iteration
                        if idx >= len(obs_array) or idx == 0:  # Skip ego vehicle or invalid indices
                            continue

                        # Skip if the vehicle is not present in the original observation array
                        if obs_array[idx][0] == 0:
                            continue
                        
                        # Skip if vehicle no longer exists (might have been removed after a collision)
                        if veh_id not in self._vehicles:
                            continue
                        
                        # Generate random perturbations
                        steer_delta = float(np.random.uniform(*self.steer_range))
                        acc_delta = float(np.random.uniform(*self.acc_range))
                        
                        # Store these perturbations temporarily
                        new_deltas[veh_id] = (acc_delta, steer_delta)
                        
                        # Calculate predicted state for this vehicle
                        base_row = obs_array[idx].copy()
                        
                        # Get current relative position and velocity
                        x_rel, y_rel = base_row[1], base_row[2]
                        v_mag = np.linalg.norm(base_row[3:5])
                        
                        # Calculate next relative position with perturbations
                        x_next, y_next, vx_next, vy_next = calculate_next_status(
                            x_rel, y_rel, v_mag, base_row[7] + steer_delta, base_row[8] + acc_delta,
                            ego_v, ego_steer, ego_acc, self.dt_pred
                        )
                        
                        # Update the predicted observation for this vehicle
                        pred_row = base_row.copy()
                        # Ensure we're maintaining relative coordinates
                        pred_row[1] = x_next  # Already relative to ego vehicle
                        pred_row[2] = y_next
                        pred_row[3] = vx_next
                        pred_row[4] = vy_next
                        pred_row[7] = base_row[7] + steer_delta
                        pred_row[8] = base_row[8] + acc_delta
                        
                        # Update in the combined prediction
                        pred_obs[idx] = pred_row
                
                # 1. Compute SEE grid_decimal using relative positions of all NPCs
                # Create a temporary data structure for compute_see_carla
                tmp_data_struct = {
                    "game_time": [], "vehicle_id": [], "x": [], "y": [], "vx": [], "vy": [],
                    "lane_id": [], "steering": [], "acceleration": [], "is_ego": []
                }

                current_time = GameTime.get_time()
                # Add ego vehicle data first (position at origin in relative coordinates)
                tmp_data_struct["game_time"].append(current_time)
                tmp_data_struct["vehicle_id"].append(0)  # Use 0 for ego
                tmp_data_struct["x"].append(0)  # Ego is at origin in relative coordinates
                tmp_data_struct["y"].append(0)
                tmp_data_struct["vx"].append(pred_obs[0][3])
                tmp_data_struct["vy"].append(pred_obs[0][4])
                tmp_data_struct["lane_id"].append(pred_obs[0][6])
                tmp_data_struct["steering"].append(pred_obs[0][7])
                tmp_data_struct["acceleration"].append(pred_obs[0][8])
                tmp_data_struct["is_ego"].append(True)

                # Add NPC vehicles
                for idx, row in enumerate(pred_obs[1:], 1):
                    if row[0] == 0:  # Skip inactive vehicles
                        continue
                    
                    # Check if this index corresponds to a vehicle we still have
                    if not any(idx == index for index in self._vehicle_indices.values()):
                        continue  # Skip vehicles that may have been removed
                    
                    tmp_data_struct["game_time"].append(current_time)
                    tmp_data_struct["vehicle_id"].append(idx)
                    tmp_data_struct["x"].append(row[1])  # Already relative to ego
                    tmp_data_struct["y"].append(row[2])
                    tmp_data_struct["vx"].append(row[3])
                    tmp_data_struct["vy"].append(row[4])
                    tmp_data_struct["lane_id"].append(row[6])
                    tmp_data_struct["steering"].append(row[7])
                    tmp_data_struct["acceleration"].append(row[8])
                    tmp_data_struct["is_ego"].append(False)

                # Use compute_see_carla to get the SEE matrix with proper preprocessing
                see_mat, _ = compute_see_carla(
                    spec_conf.lx, 
                    spec_conf.ly, 
                    spec_conf.n_rad, 
                    spec_conf.n_ring, 
                    tmp_data_struct
                )
                grid_decimal = grid_to_decimal(see_mat)
                
                # 2. Compute TTR (DSEE) for the predicted state
                try:
                    ttr_real_pred = compute_dsee_carla(tmp_data_struct)
                    ttr_seg_pred = get_segmented_value(ttr_real_pred, spec_conf.ttr_segments)
                except Exception as e:
                    _log.debug(f"TTR calculation error: {e}")
                    ttr_real_pred = float('inf')
                    ttr_seg_pred = get_segmented_value(ttr_real_pred, spec_conf.ttr_segments)
                
                # 3. Compute planning type using the model
                planning_type_pred = 0  # Default: straight
                if plan_model is not None:
                    try:
                        # Apply proper normalization to observation
                        norm_obs = process_obs_norm(pred_obs)
                        input_tensor = torch.from_numpy(norm_obs).float().unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            output = plan_model(input_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            planning_type_pred = int(torch.argmax(probabilities, dim=1).item() - 1)  # Map to -1,0,1
                    except Exception as e:
                        _log.debug(f"Planning model inference failed: {e}")
                
                # 4. Check if this HSR state exists in the main collection
                check_dict = {
                    "see": grid_decimal, 
                    "dsee": ttr_seg_pred,
                    "pe": planning_type_pred,
                    "covered": False  # Looking for uncovered states
                }
                
                exists = hsr_collection.count_documents(check_dict, limit=1) > 0
                novel_state = exists  # Novel if it exists with covered=False
                
                # If this is a novel HSR state, use these perturbations
                if novel_state:
                    self._vehicle_deltas = new_deltas
                    self._last_try_count = try_count
                    print(f"Found novel HSR state at try: {try_count}")
                    break

                try_count += 1
            
            # Update final try count even if we didn't find a novel state
            self._last_try_count = try_count

        except Exception as e:
            _log.debug(f"Perturbation recalc error: {e}")
            
    def get_last_try_count(self):
        """Return the number of attempts made in the last perturbation recalculation."""
        return self._last_try_count


# ---------------------------------------------------------
# Runtime data collection & perturbation management
# ---------------------------------------------------------

class RuntimeDataCollector(py_trees.behaviour.Behaviour):
    """
    Behavior that collects SEE data and updates the SEEPerturbationManager
    Additionally, it calculates coverage-related metrics and stores the
    information in MongoDB (COAX implementation).
    """
    def __init__(self, name="SEEDataCollector", ego_vehicle=None,
                 main_collection=None, hsr_collection=None,
                 plan_model=None, device=None, save_name="carla_spec"):
        super(RuntimeDataCollector, self).__init__(name)
        self.ego_vehicle = ego_vehicle
        self.manager = PerturbationManager()
        self.spec_data_collector = None
        # DB & model references
        self.main_collection = main_collection
        self.hsr_collection = hsr_collection
        self.plan_model = plan_model
        self.device = device
        # Keep track of last stored time to avoid duplicate inserts
        self._last_saved_time = -1.0
        self._tick_counter = 0
        # same gap control as manager for timing
        self._gap_control = 8
        # Save name to identify this scenario run in the database
        self.save_name = save_name
        # Track start time for total_wall_time calculation
        self._start_time = datetime.datetime.now()
        # Collision tracking
        self._last_collision_time = -1.0
        self._collision_data = None
        self._collision_sensor = None
        self._collision_history = []
        
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
            # Check for both 'static' in type_id and specific static objects
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
                _log.warning(f"Collision with static object: {other_actor.type_id} - Ending scenario")
                
                # End the scenario - Find the root tree and set it to SUCCESS
                self._end_scenario()
                
                # Don't store collision data for static objects
                return
            
            # Store velocities in collision history (for calculating pre-collision velocities)
            # We need two frames to make this calculation, so store current velocities
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
            
            # First, notify the perturbation manager to remove this vehicle
            self.manager.remove_vehicle(vehicle_id)
            
            # Update the parent scenario class data structures, if we can access them
            # This is a bit tricky since we don't have direct access to the SPEC_Perturbation instance
            # Try to find the parent scenario
            parent_scenario = None
            
            # Method 1: Try to get from CarlaDataProvider if the method exists
            if hasattr(CarlaDataProvider, 'get_running_scenario'):
                parent_scenario = CarlaDataProvider.get_running_scenario()
            
            # Method 2: Use our parent tree to find a SPEC_Perturbation parent
            if parent_scenario is None and hasattr(self, 'parent'):
                node = self.parent
                while node is not None:
                    if isinstance(node, SPEC_Perturbation):
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
            
            # Increment tick counter
            self._tick_counter += 1
            
            # First recalculate perturbations for the current tick
            if self.spec_data_collector and hasattr(self.spec_data_collector, '_data_structure'):
                obs_array = np.array(self.spec_data_collector._data_structure['compact_obs']) if 'compact_obs' in self.spec_data_collector._data_structure else None
                # fallback: build from last captured values
                if obs_array is None:
                    try:
                        latest_time = max(self.spec_data_collector._data_structure['game_time'])
                        latest_indices = [i for i, t in enumerate(self.spec_data_collector._data_structure['game_time']) if t == latest_time]
                        
                        # Find ego vehicle index first
                        ego_index = None
                        for idx in latest_indices:
                            if self.spec_data_collector._data_structure['is_ego'][idx]:
                                ego_index = idx
                                break
                                    
                        if ego_index is not None:
                            ego_x = self.spec_data_collector._data_structure['x'][ego_index]
                            ego_y = self.spec_data_collector._data_structure['y'][ego_index]
                            ego_vx = self.spec_data_collector._data_structure['vx'][ego_index]
                            ego_vy = self.spec_data_collector._data_structure['vy'][ego_index]
                            
                            obs_rows = []
                            # Ego row (first row) - position MUST be (0,0)
                            obs_rows.append([
                                1, 
                                0,  # x is always 0 for ego (relative to itself)
                                0,  # y is always 0 for ego (relative to itself)
                                ego_vx,
                                ego_vy,
                                0,
                                self.spec_data_collector._data_structure['lane_id'][ego_index],
                                self.spec_data_collector._data_structure['steering'][ego_index],
                                self.spec_data_collector._data_structure['acceleration'][ego_index]
                            ])
                            
                            # Other vehicles (calculate relative positions)
                            for idx in latest_indices:
                                if idx == ego_index:
                                    continue
                                    
                                # Calculate relative position to ego
                                rel_x = self.spec_data_collector._data_structure['x'][idx] - ego_x
                                rel_y = self.spec_data_collector._data_structure['y'][idx] - ego_y
                                
                                obs_rows.append([
                                    1,
                                    rel_x,  # Relative x position
                                    rel_y,  # Relative y position
                                    self.spec_data_collector._data_structure['vx'][idx],
                                    self.spec_data_collector._data_structure['vy'][idx],
                                    0,
                                    self.spec_data_collector._data_structure['lane_id'][idx],
                                    self.spec_data_collector._data_structure['steering'][idx],
                                    self.spec_data_collector._data_structure['acceleration'][idx]
                                ])
                                
                                if len(obs_rows) >= 11:  # Limit to 11 rows
                                    break
                            
                            # Pad with zeroes to ensure 11 rows
                            while len(obs_rows) < 11:
                                obs_rows.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
                                
                            obs_array = np.array(obs_rows)
                    except Exception as e:
                        _log.debug(f"Failed to create fallback observation: {e}")
                        obs_array = None

                if obs_array is not None:
                    self.manager.recalculate_perturbations(
                        obs_array, 
                        SPEC_CONF, 
                        self.hsr_collection,
                        self.main_collection,
                        self.plan_model,
                        self.device,
                        self._tick_counter
                    )
            
            # Then get the SEE encoding and process storage (which now has access to updated perturbation data)
            see_encoding = self.spec_data_collector.get_see_encoding()
            if see_encoding is not None:
                # Coverage & DB logic
                self._handle_coax_storage(see_encoding)

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

    # New helper method for COAX storage
    def _handle_coax_storage(self, see_encoding):
        if self.main_collection is None or self.hsr_collection is None:
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
            
            # Add try_count only when perturbation happens (on recalculation frames)
            if self._tick_counter % self._gap_control == 0 or self._tick_counter == 2:
                # Since we've moved recalculation before storage, we should always have the count
                doc["try_count"] = self.manager.get_last_try_count()
            
            # Add collision data if a collision has been detected
            if self._collision_data:
                # Add collision data to document
                doc.update(self._collision_data)
                _log.debug(f"Adding collision data to document: {self._collision_data}")
                # Reset collision data after it's been recorded
                self._collision_data = None
                
            self.main_collection.insert_one(doc)
            # Update HSR coverage
            self.hsr_collection.update_one(
                {"see": grid_decimal, "dsee": ttr_seg, "pe": planning_type_real},
                {"$set": {"covered": True}},
                upsert=True,
            )
            # print(f"Actual hsr: see={grid_decimal}, dsee={ttr_seg}, pe={planning_type_real}")
            # Update last saved time
            self._last_saved_time = current_time
        except Exception as e:
            _log.debug(f"COAX storage error: {e}")

    def _end_scenario(self):
        """End the scenario gracefully"""
        try:
            # Find the parent scenario
            parent_scenario = None
            
            # Method 1: Try to get from CarlaDataProvider if the method exists
            if hasattr(CarlaDataProvider, 'get_running_scenario'):
                parent_scenario = CarlaDataProvider.get_running_scenario()
            
            # Method 2: Use our parent tree to find a SPEC_Perturbation parent
            if parent_scenario is None and hasattr(self, 'parent'):
                node = self.parent
                while node is not None:
                    if isinstance(node, SPEC_Perturbation):
                        parent_scenario = node
                        break
                    if hasattr(node, 'parent'):
                        node = node.parent
                    else:
                        break
            
            # Method 3: Find the root node of the behavior tree and tell it to terminate
            root = self
            while hasattr(root, 'parent') and root.parent is not None:
                root = root.parent
            
            # Set this behavior to SUCCESS to end its branch
            self.feedback_message = "Terminated after static collision"
            self.status = py_trees.common.Status.SUCCESS
            
            # Directly notify running scenario if we found it
            if parent_scenario:
                _log.debug("Setting parent scenario to stop")
                # Mark scenario as complete to allow early termination
                if hasattr(parent_scenario, '_scenario_completed'):
                    parent_scenario._scenario_completed = True
                    
            # Also try to notify the scenario manager if we can find it
            try:
                # Import directly, should be already loaded
                import py_trees
                from srunner.scenariomanager.scenario_manager import ScenarioManager
                
                # Try the direct api to get the scenario manager
                scenario_manager = None
                if hasattr(CarlaDataProvider, 'get_scenario_manager'):
                    scenario_manager = CarlaDataProvider.get_scenario_manager()
                
                if scenario_manager and isinstance(scenario_manager, ScenarioManager):
                    _log.debug("Signaling scenario manager to stop scenario")
                    scenario_manager.stop_scenario()
                
                # If we can't get the scenario manager directly, try setting all parent nodes to SUCCESS
                node = self.parent
                while node is not None:
                    if isinstance(node, py_trees.composites.Composite):
                        _log.debug(f"Setting {node.name} to SUCCESS")
                        # Skip internal behaviors
                        if hasattr(node, 'status'):
                            node.status = py_trees.common.Status.SUCCESS
                            # If this is a sequence, also signal its current child that it's done
                            if hasattr(node, 'current_child') and node.current_child:
                                node.current_child.status = py_trees.common.Status.SUCCESS
                                
                    if hasattr(node, 'parent'):
                        node = node.parent
                    else:
                        break
                        
            except Exception as e:
                _log.debug(f"Error signaling scenario manager: {e}")
                
        except Exception as e:
            _log.debug(f"Error ending scenario after static collision: {e}")


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
        self.manager = PerturbationManager()
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
            acc_delta, steering_pert = self.manager.get_vehicle_perturbation(self.vehicle_id)
            
            # Convert acceleration delta to throttle/brake adjustments
            if acc_delta >= 0:
                # Positive acceleration: increase throttle, keep brake at zero
                throttle_adj = min(1.0, acc_delta / 4.0)  # Scale factor of 4.0 m/s² = full throttle
                brake_adj = 0.0
            else:
                # Negative acceleration: zero throttle, apply brake
                throttle_adj = 0.0
                brake_adj = min(1.0, abs(acc_delta) / 10.0)  # Scale factor of 10.0 m/s² = full brake

            # Apply adjustments to control
            if acc_delta >= 0:
                # For positive acceleration, add to existing throttle
                control.throttle = max(0.0, min(1.0, control.throttle + throttle_adj))
                control.brake = 0.0
            else:
                # For negative acceleration, prioritize braking
                control.throttle = 0.0
                control.brake = max(0.0, min(1.0, control.brake + brake_adj))
            
            # Convert steering_pert from radians to control.steer range (-1 to 1)
            # Full π/2 radians (90 degrees) would correspond to full steering lock (±1.0)
            control_steering_pert = steering_pert / (np.pi/2)
            
            # Apply converted steering perturbation
            control.steer = max(-1.0, min(1.0, control.steer + control_steering_pert))
            
            # Apply the modified control to the vehicle
            self._actor.apply_control(control)
            
            # Print debug information
            # if abs(throttle_pert) > 0.01 or abs(steering_pert) > 0.01:
            #     print(f"Vehicle {self.vehicle_id}: Applied perturbations - throttle: {throttle_pert:.3f}, steering: {steering_pert:.3f}")
            #     print(f"Vehicle {self.vehicle_id}: Final controls - throttle: {control.throttle:.3f}, steering: {control.steer:.3f}")
        
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
        
        # Initialize COAXConfig to get npc_num list
        coax_config = COAXConfig()
        
        # Get parameters - use random selection from npc_num for number of vehicles
        self._num_vehicles = int(np.random.choice(coax_config.npc_num))
        self._min_speed = get_value_parameter(config, "min_speed", float, 5)
        self._max_speed = get_value_parameter(config, "max_speed", float, 15)
        
        # Get random seed if provided
        self._random_seed = get_value_parameter(config, "random_seed", int, None)
        if self._random_seed is not None:
            random.seed(self._random_seed)
            print(f"SPEC_Perturbation: Using random seed {self._random_seed}")
        else:
            print("SPEC_Perturbation: Using random seed from system time.")
            
        # Generate a random save_name
        self._save_name = generate_random_name_string()
        print(f"SPEC_Perturbation: Using randomly generated save_name: {self._save_name}")
        print(f"SPEC_Perturbation: Randomly selected {self._num_vehicles} vehicles from npc_num list")

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
        self._perturbation_manager = PerturbationManager()

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
        Custom initialization of actors with coverage-driven positioning
        """
        # Track attempts for logging/debugging
        see_try_count = 0
        inside_try_count = 0
        max_tries = 20  # Maximum number of position configurations to try
        max_inside_tries = 10  # Maximum attempts with the same vehicle count
        found_uncovered = False
        
        # The original number of vehicles requested
        original_num_vehicles = self._num_vehicles
        
        # Get database collection if we want to check for coverage
        hsr_collection = _hsr_collection  # Use the global database reference
        
        # If no database or HSR collection is available, skip coverage-based optimization
        if hsr_collection is None:
            print("No HSR collection available. Using random positions without coverage check.")
            found_uncovered = True
        
        while see_try_count <= max_tries and not found_uncovered:
            # Generate random starting positions like before
            self._start_waypoints = find_non_overlapping_waypoints(
                self._reference_waypoint,
                self._num_vehicles,
                min_distance=-40.0,  # 40m behind
                max_distance=40.0,   # 40m ahead
                max_attempts=100
            )
            
            # Skip if no valid waypoints were found
            if not self._start_waypoints:
                print("Error: No start waypoints available. Cannot proceed.")
                return
            
            # Adjust number of vehicles based on available waypoints
            self._num_vehicles = min(self._num_vehicles, len(self._start_waypoints))
            
            # Get destination waypoints for these starting positions
            self._destination_waypoints = find_destination_waypoints(self._start_waypoints, self._end_waypoint)
            
            # If we don't have a database reference, don't try to optimize
            if hsr_collection is None:
                found_uncovered = True
                break
            
            # Try to check if this configuration creates a new coverage state
            try:
                # Calculate SEE grid using just waypoint positions (no need to create vehicles)
                # Create data structure for SEE calculation
                data_struct = {
                    "game_time": [], "vehicle_id": [], "x": [], "y": [], "vx": [], "vy": [],
                    "lane_id": [], "steering": [], "acceleration": [], "is_ego": []
                }
                
                # Add ego vehicle data (using reference waypoint)
                current_time = GameTime.get_time()
                data_struct["game_time"].append(current_time)
                data_struct["vehicle_id"].append(0)  # Use 0 for ego
                data_struct["x"].append(self._reference_waypoint.transform.location.x)
                data_struct["y"].append(self._reference_waypoint.transform.location.y)
                data_struct["vx"].append(0)  # Stationary for calculation
                data_struct["vy"].append(0)
                data_struct["lane_id"].append(self._reference_waypoint.lane_id)
                data_struct["steering"].append(0)
                data_struct["acceleration"].append(0)
                data_struct["is_ego"].append(True)
                
                # Add NPC vehicle data from waypoints
                for idx, wp in enumerate(self._start_waypoints):
                    data_struct["game_time"].append(current_time)
                    data_struct["vehicle_id"].append(idx + 1)
                    data_struct["x"].append(wp.transform.location.x)
                    data_struct["y"].append(wp.transform.location.y)
                    data_struct["vx"].append(0)  # Stationary for calculation
                    data_struct["vy"].append(0)
                    data_struct["lane_id"].append(wp.lane_id)
                    data_struct["steering"].append(0)
                    data_struct["acceleration"].append(0)
                    data_struct["is_ego"].append(False)
                
                # Calculate SEE grid
                see_mat, _ = compute_see_carla(
                    SPEC_CONF.lx, 
                    SPEC_CONF.ly, 
                    SPEC_CONF.n_rad, 
                    SPEC_CONF.n_ring, 
                    data_struct
                )
                grid_decimal = grid_to_decimal(see_mat)
                
                # Query database to check if this SEE state is already covered
                query = {"see": grid_decimal, "covered": {"$exists": True, "$eq": False}}
                is_uncovered = hsr_collection.count_documents(query, limit=1) > 0
                
                if is_uncovered:
                    print(f"Found uncovered SEE state (grid_decimal: {grid_decimal}) at try {see_try_count}, inner try {inside_try_count}")
                    found_uncovered = True
                else:
                    # This SEE state is already covered, try again
                    inside_try_count += 1
                    
                    # If we've tried too many times with this vehicle count, increment and try again
                    if inside_try_count >= max_inside_tries:
                        inside_try_count = 0
                        see_try_count += 1
                    continue
                
            except Exception as e:
                print(f"Error during SEE calculation: {e}")
                # If calculation fails, just use the current configuration
                found_uncovered = True
            
            # Break if we've tried too many times
            if see_try_count >= max_tries:
                print(f"Reached maximum attempts ({max_tries}). Using current configuration.")
                found_uncovered = True
        
        # Now create the actual vehicles using the final waypoints
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
        
        # Check if we have any vehicles to work with
        if not self._vehicles:
            _log.warning("No vehicles were successfully spawned. Creating minimal behavior tree.")
            # Add just the ego vehicle collector
            root.add_child(RuntimeDataCollector("RuntimeCollector", self.ego_vehicles[0],
                                        main_collection=_main_collection, hsr_collection=_hsr_collection,
                                        plan_model=_plan_model, device=device, save_name=self._save_name))
            # Add a dummy sequence that always succeeds
            dummy = py_trees.composites.Sequence("DummyBehavior")
            dummy.add_child(py_trees.behaviours.Success("DummySuccess"))
            root.add_child(dummy)
            return root
        
        # Add the data collector to the root
        root.add_child(RuntimeDataCollector("RuntimeCollector", self.ego_vehicles[0],
                                        main_collection=_main_collection, hsr_collection=_hsr_collection,
                                        plan_model=_plan_model, device=device, save_name=self._save_name))
        
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

# Update the forward declaration with the actual class
globals()['SPEC_Perturbation'] = SPEC_Perturbation
