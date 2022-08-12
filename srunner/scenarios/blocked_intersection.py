#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenario with low visibility, the ego performs a turn only to find out that the end is blocked by another vehicle.
"""

from __future__ import print_function

import carla
import py_trees
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorDestroy, Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import \
    CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import \
    InTriggerDistanceToVehicle
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import HandleJunctionScenario

from srunner.tools.scenario_helper import generate_target_waypoint_in_route


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict['x']),
        y=float(actor_dict['y']),
        z=float(actor_dict['z'])
    )
    return location


class BlockedIntersection(BasicScenario):
    """
    This class holds everything required for a scenario in which,
    the ego performs a turn only to find out that the end is blocked by another vehicle.
    """

    def __init__(self, world, ego_vehicles, config, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)

        self._blocker_location = convert_dict_to_location(
            config.other_parameters['blocker_point'])
        self._blocker_waypoint = self._map.get_waypoint(self._blocker_location)
        self._block_distance = 10  # m. Will stop blocking when ego is within this distance
        self._block_time = 5  # s

        if 'obstacle_model' in config.other_parameters:
            self._obstacle_model = config.other_parameters['obstacle_model']['value']
        else:
            self._obstacle_model = "static.prop.trampoline"

        if 'obstacle_gap' in config.other_parameters:
            self._obstacle_gap = int(
                config.other_parameters['obstacle_gap']['value'])
        else:
            self._obstacle_gap = 2

        # Extra obstacles are not included. One obstacle by default.
        self._obstacle_amount = 1

        # The amount of obstacles that invade the road
        if 'extra_obstacle' in config.other_parameters:
            self._extra_obstacle = int(
                config.other_parameters['extra_obstacle']['value'])
        else:
            self._extra_obstacle = 2

        super(BlockedIntersection, self).__init__("BlockedIntersection",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = generate_target_waypoint_in_route(self._reference_waypoint, config.route)
        waypoint = waypoint.next(self._blocker_distance)[0]

        # Spawn the blocker vehicle
        actor = CarlaDataProvider.request_new_actor(
            "vehicle.*.*", waypoint.transform,
            attribute_filter={'base_type': 'car', 'has_lights': True}
        )
        if actor is None:
            raise Exception("Couldn't spawn the blocker vehicle")
        self.other_actors.append(actor)

    def _create_behavior(self):
        """
        Just wait for a while after the ego closes in on the blocker, then remove it.
        """
        sequence = py_trees.composites.Sequence(name="BlockedIntersection")

        if self.route_mode:
            sequence.add_child(HandleJunctionScenario(
                clear_junction=True,
                clear_ego_entry=False,
                remove_entries=[],
                remove_exits=[],
                stop_entries=True,
                extend_road_exit=0
            ))
        # Ego go behind the blocker
        sequence.add_child(InTriggerDistanceToVehicle(
            self.other_actors[-1], self.ego_vehicles[0],  self._block_distance))
        sequence.add_child(Idle(self._block_time))
        sequence.add_child(ActorDestroy(self.other_actors[-1]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        if self.route_mode:
            return[]

        return [CollisionTest(self.ego_vehicles[0])]

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
