import os
import time
import argparse
from utils.llm_aft_trainer import LLM_Inference, LLM_Inference_VLLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='LLMTLCSRun')
    parser.add_argument("--llm_model", type=str, default="llama_cgpr_13b_jinan_1")
    parser.add_argument(
        "--llm_path", type=str, default="./ft_models/merged/llama_cgpr_13b_jinan_1"
    )
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--new_max_tokens", type=int, default=1024)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="template")
    parser.add_argument("--with_vllm", type=bool, default=False)

    return parser.parse_args()


def main(in_args):
    in_args.model = in_args.memo

    dic_agent_conf_extra = {
        "LLM_PATH": in_args.llm_path,
        "LLM_MODEL": in_args.llm_model,
        "LOG_DIR": f"./{in_args.llm_model}_logs",
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
    }

    dic_traffic_env_conf_extra = {
        "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": in_args.proj_name,
        "NUM_ROUNDS": in_args.num_rounds,
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],
        # 奖励函数
        "DIC_REWARD_INFO": {'queue_length': -0.25},
    }

    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join(
            "model",
            in_args.memo
            + "_"
            + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())),
        ),
        "PATH_TO_WORK_DIRECTORY": os.path.join(
            "records",
            in_args.memo
            + "_"
            + time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())),
        ),
        "PATH_TO_DATA": os.path.join("data"),
    }

    if in_args.with_vllm:
        trainer = LLM_Inference_VLLM(
            dic_agent_conf_extra,
            dic_traffic_env_conf_extra,
            dic_path_extra,
        )
    else:
        trainer = LLM_Inference(
            dic_agent_conf_extra,
            dic_traffic_env_conf_extra,
            dic_path_extra,
        )

    trainer.train_test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
