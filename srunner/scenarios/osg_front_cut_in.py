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


def get_relative_left_waypoint(reference_waypoint, relative_p):
    if relative_p > 0:
        target = reference_waypoint.get_left_lane().next(relative_p)[0]
    elif relative_p < 0:
        target = reference_waypoint.get_left_lane().previous(abs(relative_p))[0]
    else:
        target = reference_waypoint.get_left_lane()
    return target


def create_vehicle_left_and_move_underground(waypoint):
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
        self._other_lane_time = 2
        self._change_time = 1

        self._relative_p = get_value_parameter(config, "relative_p", float, 10)
        self._relative_v = get_value_parameter(config, "relative_v", float, 5)
        self._absolute_v = get_value_parameter(config, "absolute_v", float, 5)
        self._lc_target_v = max(0.1, self._absolute_v + self._relative_v)

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self._lc_target_waypoint = self._reference_waypoint.next(300)[0]

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
        self._npc_waypoint = get_relative_left_waypoint(
            self._reference_waypoint, self._relative_p
        )
        self._cut_in_vehicle = create_vehicle_left_and_move_underground(
            self._npc_waypoint
        )

        self.other_actors.append(self._cut_in_vehicle)

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

        root.add_child(OASDataCollector(self._cut_in_vehicle, name="NPCData"))

        behavior = py_trees.composites.Sequence("HighwayCutIn")

        root.add_child(behavior)

        behavior.add_child(
            ActorTransformSetter(self._cut_in_vehicle, self._npc_waypoint.transform)
        )

        behavior.add_child(SetInitSpeed(self.ego_vehicles[0], self._absolute_v))

        # Sync behavior
        behavior.add_child(SetInitSpeed(self._cut_in_vehicle, self._lc_target_v))
        behavior.add_child(
            AccelerateToVelocity(self._cut_in_vehicle, 1, self._lc_target_v)
        )

        # Cut in
        npc_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Cut in behavior"
        )

        cut_in_movement = py_trees.composites.Sequence()
        cut_in_movement.add_child(
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
        cut_in_movement.add_child(
            BasicAgentBehavior(
                self._cut_in_vehicle,
                target_speed=self._lc_target_v * 3.6,
                target_location=self._lc_target_waypoint.transform.location,
            )
        )
        npc_behavior.add_child(cut_in_movement)

        npc_behavior.add_child(
            DriveDistance(self._cut_in_vehicle, 300)
        )  # Long enough to finish the scenario

        behavior.add_child(npc_behavior)

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


class OSG_CutIn_Two(BasicScenario):
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
        self._other_lane_time = 2
        self._change_time = 1

        self._relative_p_1 = get_value_parameter(config, "relative_p_1", float, 10)
        self._relative_p_2 = get_value_parameter(config, "relative_p_2", float, 5)
        self._relative_v_1 = get_value_parameter(config, "relative_v_1", float, 5)
        self._relative_v_2 = get_value_parameter(config, "relative_v_2", float, 5)
        self._absolute_v = get_value_parameter(config, "absolute_v", float, 5)
        self._lc_target_v = max(0.1, self._absolute_v + self._relative_v_1)
        self._follow_target_v = max(0.1, self._absolute_v + self._relative_v_2)

        self._trigger_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._trigger_location)
        self._lc_target_waypoint = self._reference_waypoint.next(300)[0]
        self._follow_target_waypoint = self._lc_target_waypoint.get_left_lane()

        super().__init__(
            "OSG_CutIn_2",
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
        self._lc_waypoint = get_relative_left_waypoint(
            self._reference_waypoint, self._relative_p_1
        )
        self._cut_in_vehicle = create_vehicle_left_and_move_underground(
            self._lc_waypoint
        )

        self.other_actors.append(self._cut_in_vehicle)

        self._follow_waypoint = get_relative_left_waypoint(
            self._reference_waypoint, self._relative_p_2
        )
        self._follow_vehicle = create_vehicle_left_and_move_underground(
            self._follow_waypoint
        )

        self.other_actors.append(self._follow_vehicle)

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
            ActorTransformSetter(self._cut_in_vehicle, self._lc_waypoint.transform)
        )
        behavior.add_child(
            ActorTransformSetter(self._follow_vehicle, self._follow_waypoint.transform)
        )

        behavior.add_child(SetInitSpeed(self.ego_vehicles[0], self._absolute_v))

        # Sync behavior
        behavior.add_child(SetInitSpeed(self._cut_in_vehicle, self._lc_target_v))
        behavior.add_child(SetInitSpeed(self._follow_vehicle, self._follow_target_v))
        behavior.add_child(
            AccelerateToVelocity(self._cut_in_vehicle, 1, self._lc_target_v)
        )

        # Cut in
        npc_behavior = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL, name="Cut in behavior"
        )

        cut_in_movement = py_trees.composites.Sequence()
        cut_in_movement.add_child(
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
        cut_in_movement.add_child(
            BasicAgentBehavior(
                self._cut_in_vehicle,
                target_speed=self._lc_target_v * 3.6,
                target_location=self._lc_target_waypoint.transform.location,
            )
        )
        npc_behavior.add_child(cut_in_movement)

        npc_behavior.add_child(
            BasicAgentBehavior(
                self._follow_vehicle,
                target_speed=self._follow_target_v * 3.6,
                target_location=self._follow_target_waypoint.transform.location,
            )
        )
        npc_behavior.add_child(
            DriveDistance(self._follow_vehicle, 300)
        )  # Long enough to finish the scenario

        behavior.add_child(npc_behavior)
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
