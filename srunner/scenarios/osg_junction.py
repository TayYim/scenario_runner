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
    SetInitSpeed,
    AccelerateToVelocity,
    BasicAgentBehavior,
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


def convert_dict_to_location(actor_dict):
    """
    Convert a JSON string to a Carla.Location
    """
    location = carla.Location(
        x=float(actor_dict["x"]), y=float(actor_dict["y"]), z=float(actor_dict["z"])
    )
    return location


def create_vehicle_and_move_underground(waypoint):
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

    # This starts the engine, to allow the adversary to instantly move
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
    return vehicle


class OSG_Junction(BasicScenario):
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

        self._r_ego = get_value_parameter(config, "r_ego", float, 20)
        self._v_ego = get_value_parameter(config, "v_ego", float, 5)
        self._r_1 = get_value_parameter(config, "r_1", float, 10)
        self._v_1 = max(get_value_parameter(config, "v_1", float, 5), 0.1)

        self._center_point = convert_dict_to_location(
            config.other_parameters["center_point"]
        )
        self._npc_stopline_point = convert_dict_to_location(
            config.other_parameters["npc_stopline_point"]
        )
        self._npc_dest = convert_dict_to_location(config.other_parameters["npc_dest"])

        gap_stopline_center = self._npc_stopline_point.distance(self._center_point)
        stopline_waypoint = self._map.get_waypoint(self._npc_stopline_point)
        self._npc_start_point = stopline_waypoint.previous(
            gap_stopline_center + self._r_1
        )[0]

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)

        gap_trigger_center = self._trigger_location.distance(self._center_point)
        self._ego_pre_drive_distance = gap_trigger_center - self._r_ego

        super().__init__(
            "OSG_Junction",
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

        self._npc_vehicle = create_vehicle_and_move_underground(self._npc_start_point)

        self.other_actors.append(self._npc_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is entering a junction in an urban area, at a signalized intersection,
        while another actor runs a red lift, forcing the ego to break.
        """

        behavior = py_trees.composites.Sequence("root_behavior")

        behavior.add_child(
            DriveDistance(self.ego_vehicles[0], self._ego_pre_drive_distance)
        )

        behavior.add_child(
            ActorTransformSetter(self._npc_vehicle, self._npc_start_point.transform)
        )

        main_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Main behavior"
        )

        behavior.add_child(main_behavior)

        main_behavior.add_child(SetInitSpeed(self.ego_vehicles[0], self._v_ego))

        npc_behavior = py_trees.composites.Sequence("NPC behavior")

        main_behavior.add_child(npc_behavior)

        npc_behavior.add_child(SetInitSpeed(self._npc_vehicle, self._v_1))
        npc_behavior.add_child(AccelerateToVelocity(self._npc_vehicle, 1, self._v_1))

        npc_behavior.add_child(
            BasicAgentBehavior(
                self._npc_vehicle,
                target_speed=self._v_1 * 3.6,
                target_location=self._npc_dest,
                opt_dict={"ignore_traffic_lights": True, "ignore_stop_signs": True},
            )
        )

        return behavior

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
