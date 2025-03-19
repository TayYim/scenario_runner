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
    ActorDestroy,
)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (
    DriveDistance,
)

from srunner.scenariomanager.scenarioatomics.atomic_spec_behaviors import (
    SPECDataCollector,
)


class SPEC_Data_Collect(BasicScenario):
    """
    This class implements a SPEC data collection scenario that tracks all vehicles 
    in the scene and records their position, velocity, lane id, steering and acceleration.
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

        super().__init__(
            "SPEC_Data_Collect",
            ego_vehicles,
            config,
            world,
            debug_mode,
            criteria_enable=criteria_enable,
        )

    def _create_behavior(self):
        """
        Create behavior tree for the scenario to collect data for all vehicles
        """
        # root parallel node
        root = py_trees.composites.Parallel(
            "Main Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE
        )

        # Add the SPEC data collector to track all vehicles
        root.add_child(SPECDataCollector(self.ego_vehicles[0], name="SPECDataCollection"))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        return []

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()
