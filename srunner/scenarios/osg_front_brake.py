#!/usr/bin/env python

# Copyright (c) 2018-2022 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Hard break scenario:

The scenario spawn a vehicle in front of the ego that drives for a while before
suddenly hard breaking, forcing the ego to avoid the collision
"""

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    Idle,
    ActorTransformSetter,
    ConstantVelocityAgentBehavior,
    SetInitSpeed,
)
from srunner.scenarios.basic_scenario import BasicScenario


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]["value"])
    else:
        return default


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict["x"]), y=float(actor_dict["y"]), z=float(actor_dict["z"])
    )
    return location


class OSG_FrontBrake(BasicScenario):
    """
    This class uses the is the Background Activity at routes to create a hard break scenario.

    This is a single ego vehicle scenario
    """

    timeout = 120  # Timeout of scenario in seconds

    def __init__(
        self,
        world,
        ego_vehicles,
        config,
        randomize=False,
        debug_mode=False,
        criteria_enable=True,
        timeout=60,
    ):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self._relative_p = get_value_parameter(config, "relative_p", float, 10)
        self._relative_v = get_value_parameter(config, "relative_v", float, 5)
        self._absolute_v = get_value_parameter(config, "absolute_v", float, 5)
        self._npc_target_v = max(0.1, self._absolute_v + self._relative_v)

        self.timeout = timeout
        self._before_brake_time = 3  # s
        self._brake_duration = 5  # s
        self.end_distance = 300  # long enough to cover the whole route

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self._npc_target_location = self._reference_waypoint.next(
            self.end_distance + self._relative_p
        )[0].transform.location

        self._target_location_before_brake = self._reference_waypoint.next(
            self._before_brake_time * self._npc_target_v
        )[0].transform.location

        super().__init__(
            "OSG_HardBreak",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        npc_points = self._reference_waypoint.next(self._relative_p)
        if not npc_points:
            raise ValueError("Couldn't find viable position for the emergency vehicle")
        self._npc_start_transform = npc_points[0].transform

        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*",
            self._npc_start_transform,
            attribute_filter={
                "base_type": "car",
                "has_lights": True,
                "special_type": "",
            },
        )
        if actor is None:
            raise Exception("Couldn't spawn the emergency vehicle")

        # Move the actor underground and remove its physics so that it doesn't fall
        actor.set_simulate_physics(False)
        new_location = actor.get_location()
        new_location.z -= 500
        actor.set_location(new_location)

        # This starts the engine, to allow the adversary to instantly move
        actor.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))

        self.other_actors.append(actor)

    def _create_behavior(self):

        npc = self.other_actors[0]

        root = py_trees.composites.Parallel(
            "Main Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        behavior = py_trees.composites.Sequence("Sequence Behavior")

        root.add_child(behavior)

        # Init
        behavior.add_child(ActorTransformSetter(npc, self._npc_start_transform))

        behavior.add_child(SetInitSpeed(self.ego_vehicles[0], self._absolute_v))

        behavior.add_child(SetInitSpeed(npc, self._npc_target_v))

        # Before brake
        before_brake_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        before_brake_behavior.add_child(
            ConstantVelocityAgentBehavior(
                npc, self._npc_target_location, self._npc_target_v
            )
        )

        before_brake_behavior.add_child(Idle(self._before_brake_time))

        behavior.add_child(before_brake_behavior)

        # Braking
        brake_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        brake_behavior.add_child(
            ConstantVelocityAgentBehavior(npc, self._npc_target_location, 0)
        )

        brake_behavior.add_child(Idle(self._brake_duration))

        behavior.add_child(brake_behavior)

        # After brake

        after_brake_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        after_brake_behavior.add_child(
            ConstantVelocityAgentBehavior(
                npc, self._npc_target_location, self._npc_target_v
            )
        )

        behavior.add_child(after_brake_behavior)

        return root

    def _create_test_criteria(self):
        """
        Empty, the route already has a collision criteria
        """
        return []

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
