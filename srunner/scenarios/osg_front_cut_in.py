#!/usr/bin/env python

# Copyright (c) 2018-2022 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
    ActorTransformSetter,
    CutIn,
    SetInitSpeed,
    AccelerateToVelocity,
)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest


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


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict["x"]), y=float(actor_dict["y"]), z=float(actor_dict["z"])
    )
    return location


class OSG_CutIn_One(BasicScenario):
    """
    This class holds everything required for a scenario in which another vehicle runs a red light
    in front of the ego, forcing it to react. This vehicles are 'special' ones such as police cars,
    ambulances or firetrucks.
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

        self._same_lane_time = 1
        self._other_lane_time = 50
        self._change_time = 1

        self._relative_p = get_value_parameter(config, "relative_p", float, 10)
        self._relative_v = get_value_parameter(config, "relative_v", float, 5)
        self._absolute_v = get_value_parameter(config, "absolute_v", float, 5)
        self._lc_target_v = max(0.1, self._absolute_v + self._relative_v)

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)

        super().__init__(
            "OSG_CutIn_1",
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
        if self._relative_p > 0:
            self._npc_waypoint = self._reference_waypoint.get_left_lane().next(
                self._relative_p
            )[0]
        elif self._relative_p < 0:
            self._npc_waypoint = self._reference_waypoint.get_left_lane().previous(
                abs(self._relative_p)
            )[0]
        else:
            self._npc_waypoint = self._reference_waypoint.get_left_lane()
        self._npc_transform = self._npc_waypoint.transform

        self._cut_in_vehicle = CarlaDataProvider.request_new_actor(
            "vehicle.*",
            self._npc_transform,
            rolename="scenario",
            attribute_filter={"base_type": "car", "has_lights": True},
        )
        self.other_actors.append(self._cut_in_vehicle)

        # Move below ground
        self._cut_in_vehicle.set_location(
            self._npc_transform.location - carla.Location(z=100)
        )
        self._cut_in_vehicle.set_simulate_physics(False)

        # This starts the engine, to allow the adversary to instantly move
        self._cut_in_vehicle.apply_control(
            carla.VehicleControl(throttle=1.0, brake=0.0)
        )

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """

        # root parallel node
        root = py_trees.composites.Parallel(
            "Main Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        root.add_child(OASDataCollector(self.ego_vehicles[0], name="EGOData"))

        # root.add_child(OASDataCollector(self._cut_in_vehicle, name="NPCData"))

        behavior = py_trees.composites.Sequence("HighwayCutIn")

        root.add_child(behavior)

        behavior.add_child(
            ActorTransformSetter(self._cut_in_vehicle, self._npc_transform)
        )

        behavior.add_child(SetInitSpeed(self.ego_vehicles[0], self._absolute_v))

        # Sync behavior
        behavior.add_child(SetInitSpeed(self._cut_in_vehicle, self._lc_target_v))
        behavior.add_child(
            AccelerateToVelocity(self._cut_in_vehicle, 1, self._lc_target_v)
        )

        # Cut in
        # behavior.add_child(CutIn(
        #     self._cut_in_vehicle, self.ego_vehicles[0], 'right', self._speed_perc,
        #     self._same_lane_time, self._other_lane_time, self._change_time, name="Cut_in")
        # )
        behavior.add_child(
            CutIn(
                self._cut_in_vehicle,
                self._cut_in_vehicle,
                "right",
                100,
                self._same_lane_time,
                self._other_lane_time,
                self._change_time,
                name="Cut_in",
            )
        )

        # after_lance_change = py_trees.composites.Parallel("After LC Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # after_lance_change.add_child(AdaptiveConstantVelocityAgentBehavior(self._cut_in_vehicle, self.ego_vehicles[0], 5))
        # after_lance_change.add_child(Idle(10))
        # behavior.add_child(after_lance_change)
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
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
