#! python3
# -*- encoding: utf-8 -*-

import math
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import AtomicBehavior
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
import py_trees
from srunner.tools.scenario_helper import (
    transform_world_vector_to_local,
    adaptive_gradient,
)
import os
import csv
import datetime
import json


class DataCollector(AtomicBehavior):
    """
    This class contains a data collector behavior
    """

    def __init__(self, actor=None, name="DataCollector"):
        """
        Setup
        """
        self.task_id = name + "_" + datetime.datetime.now().strftime("%m%d%H%M%S")
        self.finished = False  # Mark whether the terminate() has been executed

        self._data_dict = {
            "game_time": [],
            "velocity_x": [],
            "velocity_y": [],
            "velocity_z": [],
            "velocity": [],
            "acceleration_x": [],
            "acceleration_y": [],
            "acceleration_z": [],
            "acceleration": [],
            "jerk_x": [],
            "jerk_y": [],
            "jerk_z": [],
            "jerk": [],
            "x": [],
            "y": [],
            "z": [],
            "roll": [],
            "pitch": [],
            "yaw": [],
            "ttc": [],  # None if there is no preceding vehicle
            "thw": [],  # None if there is no preceding vehicle
            "relative_distance": [],  # None if there is no preceding vehicle
            "relative_velocity": [],  # None if there is no preceding vehicle
            "front_vehicles_count": [],
            "throttle": [],  # [0,1]
            "brake": [],  # [0,1]
            "steer": [],  # [-1,1]
        }

        # Init global variables
        py_trees.blackboard.Blackboard().set("DC_COLLISION_FLAG", False, True)
        # Colission status: [x, y, v, yaw]
        py_trees.blackboard.Blackboard().set(
            "DC_COLLISION_STATUS", {"EGO": [], "NPC": []}, True
        )

        super(DataCollector, self).__init__(name, actor)

    def initialise(self):
        """
        Set up
        """
        # maybe something to do
        super(DataCollector, self).initialise()

    def update(self):
        """
        Collect data
        """
        new_status = py_trees.common.Status.RUNNING

        if self._actor is None:
            return new_status

        game_time = GameTime.get_time()

        self._data_dict["game_time"].append(game_time)

        # transform
        transform = CarlaDataProvider.get_transform(self._actor)
        self._data_dict["x"].append(round(float(transform.location.x), 2))
        self._data_dict["y"].append(round(float(transform.location.y), 2))
        self._data_dict["z"].append(round(float(transform.location.z), 2))
        self._data_dict["roll"].append(round(float(transform.rotation.roll), 2))
        self._data_dict["pitch"].append(round(float(transform.rotation.pitch), 2))
        self._data_dict["yaw"].append(round(float(transform.rotation.yaw), 2))

        # velocity
        velocity_vector = CarlaDataProvider.get_velocity_vector(self._actor)
        velocity_local = transform_world_vector_to_local(transform, velocity_vector)

        self._data_dict["velocity_x"].append(round(float(velocity_local.x), 2))
        self._data_dict["velocity_y"].append(round(float(velocity_local.y), 2))
        self._data_dict["velocity_z"].append(round(float(velocity_local.z), 2))
        self._data_dict["velocity"].append(round(float(velocity_local.length()), 2))

        # car follow data
        car_follow_data = CarlaDataProvider.get_car_follow_data(self._actor)
        if car_follow_data["ttc"] is None:
            ttc = None
        else:
            ttc = round(float(car_follow_data["ttc"]), 2)
        if car_follow_data["thw"] is None:
            thw = None
        else:
            thw = round(float(car_follow_data["thw"]), 2)
        if car_follow_data["rel_distance"] is None:
            rel_distance = None
        else:
            rel_distance = round(float(car_follow_data["rel_distance"]), 2)
        if car_follow_data["rel_velocity"] is None:
            rel_velocity = None
        else:
            rel_velocity = round(float(car_follow_data["rel_velocity"]), 2)

        self._data_dict["ttc"].append(ttc)
        self._data_dict["thw"].append(thw)
        self._data_dict["relative_distance"].append(rel_distance)
        self._data_dict["relative_velocity"].append(rel_velocity)
        self._data_dict["front_vehicles_count"].append(
            car_follow_data["front_vehicles_count"]
        )

        # car control data
        actor_control = self._actor.get_control()
        self._data_dict["throttle"].append(round(float(actor_control.throttle), 2))
        self._data_dict["brake"].append(round(float(actor_control.brake), 2))
        self._data_dict["steer"].append(round(float(actor_control.steer), 2))

        return new_status

    def get_final_data(self):
        # Process self._data_dict

        # Get acc and jerk
        direction_list = ["", "_x", "_y", "_z"]
        for direction in direction_list:

            vel_data = self._data_dict["velocity" + direction]
            vel_data = [round(float(vel), 2) for vel in vel_data]
            self._data_dict["velocity" + direction] = vel_data

            acc_data = adaptive_gradient(vel_data, self._data_dict["game_time"])
            acc_data = [round(float(acc), 2) for acc in acc_data]
            self._data_dict["acceleration" + direction] = acc_data

            jerk_data = adaptive_gradient(acc_data, self._data_dict["game_time"])
            jerk_data = [round(float(jerk), 2) for jerk in jerk_data]
            self._data_dict["jerk" + direction] = jerk_data

        return self._data_dict

    def terminate(self, new_status):
        """
        process with all data
        """

        if self.finished:
            return

        self.finished = True

        # process final data
        self.get_final_data()

        collision_flag = py_trees.blackboard.Blackboard().get("DC_COLLISION_FLAG")

        if collision_flag:
            min_ttc = 0
        elif len(self._data_dict["ttc"]) == 0:
            min_ttc = 10
        else:
            valid_ttcs = [ttc for ttc in self._data_dict["ttc"] if ttc is not None]
            if len(valid_ttcs) == 0:
                min_ttc = 10
            else:
                min_ttc = min(valid_ttcs)

        collision_status = py_trees.blackboard.Blackboard().get("DC_COLLISION_STATUS")

        # Use self._data_dict["x"] and self._data_dict["y"] to calculate the driven distance
        driven_distance = 0
        for i in range(len(self._data_dict["x"]) - 1):
            dx = self._data_dict["x"][i + 1] - self._data_dict["x"][i]
            dy = self._data_dict["y"][i + 1] - self._data_dict["y"][i]
            driven_distance += math.sqrt(dx**2 + dy**2)
        driven_distance = round(driven_distance, 3)

        # save data
        # TODO CL: add switch to save data
        # self._save_result_to_csv()
        self._save_epoch_result_to_json(
            collision_flag, min_ttc, driven_distance, collision_status
        )
        super(DataCollector, self).terminate(new_status)

    def _save_result_to_csv(self):
        """
        save data to csv
        """
        with open(f"{self.task_id}.csv", "w") as file:
            header = self._data_dict.keys()
            writer = csv.writer(file)
            writer.writerow(header)
            length = len(self._data_dict["velocity"])
            for i in range(length):
                row = []
                for item in header:
                    try:
                        row.append(self._data_dict[item][i])
                    except:
                        pass
                writer.writerow(row)

    def _save_epoch_result_to_json(
        self, collision_flag, min_ttc, driven_distance, collision_status
    ):
        """
        save data to json
        """
        result = {
            "collision_flag": collision_flag,
            "min_ttc": min_ttc,
            "distance": driven_distance,
            "collision_status": collision_status,
        }

        with open(f"epoch_result.json", "w") as file:
            json.dump(result, file)
