import numpy as np
import json
import copy
import torch

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
location_incoming_dict = ["N", "S", "E", "W"]
eight_phase_list = ['ETWT', 'NTST', 'ELWL', 'NLSL', 'WTWL', 'ETEL', 'STSL', 'NTNL']
four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
phase2code = {0: 'ETWT', 1: 'NTST', 2: 'ELWL', 3: 'NLSL'}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {
    "N": "Northern",
    "S": "Southern",
    "E": "Eastern",
    "W": "Western",
}

phase_explanation_dict_detail = {
    "NTST": "- NTST: Northern and southern through lanes.",
    "NLSL": "- NLSL: Northern and southern left-turn lanes.",
    "NTNL": "- NTNL: Northern through and left-turn lanes.",
    "STSL": "- STSL: Southern through and left-turn lanes.",
    "ETWT": "- ETWT: Eastern and western through lanes.",
    "ELWL": "- ELWL: Eastern and western left-turn lanes.",
    "ETEL": "- ETEL: Eastern through and left-turn lanes.",
    "WTWL": "- WTWL: Western through and left-turn lanes.",
}


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data


def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e


def calculate_road_length(road_points):
    length = 0.0
    i = 1
    while i < len(road_points):
        length += np.sqrt(
            (road_points[i]['x'] - road_points[i - 1]['x']) ** 2
            + (road_points[i]['y'] - road_points[i - 1]['y']) ** 2
        )
        i += 1

    return length


def get_state(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {
                    "cells": [0 for _ in range(3)],
                    "queue_len": queue_len,
                }

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {
                    "cells": [0 for _ in range(3)],
                    "queue_len": queue_len,
                }

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][
                            gpt_lane_cell
                        ] += 1

        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {
                "cells": [0 for _ in range(3)],
                "queue_len": queue_len,
            }
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]][
                            "cells"
                        ][gpt_lane_cell] += 1

    return statistic_state, statistic_state_incoming


def get_state_detail(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {
                    "cells": [0 for _ in range(4)],
                    "queue_len": queue_len,
                    "avg_wait_time": 0.0,
                }

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {
                    "cells": [0 for _ in range(4)],
                    "queue_len": queue_len,
                    "avg_wait_time": 0.0,
                }

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][
                            gpt_lane_cell
                        ] += 1
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = (
                            env.waiting_vehicle_list[veh]['time']
                            if veh in env.waiting_vehicle_list
                            else 0.0
                        )
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = (
                    np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                )
                statistic_state[location_direction_dict[lane_group]][
                    "avg_wait_time"
                ] = avg_wait_time

        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {
                "cells": [0 for _ in range(4)],
                "queue_len": queue_len,
            }
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]][
                            "cells"
                        ][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed


def get_state_three_segment(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {
                    "cells": [0 for _ in range(3)],
                    "queue_len": queue_len,
                    "avg_wait_time": 0.0,
                }

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {
                    "cells": [0 for _ in range(3)],
                    "queue_len": queue_len,
                    "avg_wait_time": 0.0,
                }

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][
                            gpt_lane_cell
                        ] += 1
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = (
                            env.waiting_vehicle_list[veh]['time']
                            if veh in env.waiting_vehicle_list
                            else 0.0
                        )
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = (
                    np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                )
                statistic_state[location_direction_dict[lane_group]][
                    "avg_wait_time"
                ] = avg_wait_time

        # incoming lanes
        else:
            queue_len = 0.0
            for lane in range(2):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {
                "cells": [0 for _ in range(3)],
                "queue_len": queue_len,
            }
            incoming_lanes = [f"{r}_{idx}" for idx in range(2)]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]][
                            "cells"
                        ][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed


def trans_prompt_llama(message, chat_history, system_prompt):
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def state2text(state: tuple[int, int, int, bool, bool, float]):
    w1, w2, w3, s2_restricted, s3_restricted, forecast = state
    return f"""当前储水量：
- S1：{w1}万立方米 [可自由取放]
- S2：{w2}万立方米 [{'受咸潮限制' if s2_restricted else '正常'}]
- S3：{w3}万立方米 [{'受咸潮限制' if s3_restricted else '正常'}]

未来7天咸潮发生的概率：{forecast}"""


def get_prompt(state_txt: str, rf_expression: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "您是一个智能水资源管理专家，需要根据实时数据和约束条件优化三个水库的放水策略。",
        },
        {
            "role": "user",
            "content": f"三个水库系统包含S1（内陆）、S2、S3（入海口）。每日决策变量为放水量x1,x2,x3（万立方米）。当前状态：\n"
            f"{state_txt}\n\n"
            "约束条件：\n"
            "1. 水量守恒：x1≤w1, x2≤w2, x3≤w3\n"
            "2. 咸潮约束：当S2/S3受咸潮影响时，x2≥0/x3≥0（仅能放水）\n"
            "3. 最小供水量：x1+x2+x3 ≥ 当日全市需求\n\n"
            f"奖励函数：{rf_expression}\n"
            "示例：当x1=100,x2=0(S2受限),x3=150时，RF=...\n\n"  # TODO
            "请回答：\n"
            "在满足当前约束并考虑未来咸潮风险的情况下，如何设置今日(x1,x2,x3)使年度总收益最大？\n\n"
            "要求：\n"
            "- 分三步分析：当前约束→未来预测→长期收益\n"
            "- 必须用<analysis>标签包裹推理过程\n"
            "- 最终决策用<decision>x1,x2,x3</decision>表示",
        },
    ]


def action2code(action):
    code = four_phase_list[action]

    return code


def code2action(action):
    code = phase2code[action]

    return code


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
