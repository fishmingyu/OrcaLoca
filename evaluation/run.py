import argparse
import time

import docker

from Orcar import OrcarAgent
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset

default_args_dict = {
    "model": "claude-3-5-sonnet-20241022",
    "image": "hejiaz/swe-agent:latest",
    "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "orcar_swe_bench_run_ctr",
    "split": "test",
    "max_retry": 2,
    "filter_instance": ".*",
    "final_stage": "search",
    "redirect_log": True,
    "cfg_path": "./key.cfg",
}


def parse_inputs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default=default_args_dict["model"],
        help=f"The model to use",
    )
    parser.add_argument(
        "-i",
        "--image",
        default=default_args_dict["image"],
        help=f"The image to use",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=default_args_dict["dataset"],
        help=f"The dataset to use",
    )
    parser.add_argument(
        "-p",
        "--persistent",
        default=default_args_dict["persistent"],
        help=f"The persistent flag",
    )
    parser.add_argument(
        "-c",
        "--container_name",
        default=default_args_dict["container_name"],
        help=f"The container name",
    )
    parser.add_argument(
        "-s",
        "--split",
        default=default_args_dict["split"],
        help=f"The split to use",
    )
    parser.add_argument(
        "-r",
        "--max_retry",
        default=default_args_dict["max_retry"],
        help=f"The max retry",
    )
    parser.add_argument(
        "-ii",
        "--instance_ids",
        nargs="+",
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "-fs",
        "--final_stage",
        default=default_args_dict["final_stage"],
        help=f"The final stage",
    )
    parser.add_argument(
        "-rl",
        "--redirect_log",
        default=default_args_dict["redirect_log"],
        help=f"The redirect log",
    )
    parser.add_argument(
        "-cfg",
        "--cfg_path",
        default=default_args_dict["cfg_path"],
        help=f"The cfg path",
    )
    args = parser.parse_args()
    # Conver args.instance_ids to args.filter_instance
    args.filter_instance = (
        default_args_dict["filter_instance"]
        if not hasattr(args, "instance_ids")
        else "^(" + "|".join(args.instance_ids) + ")$"
    )
    return args


def stop_container_by_name(container_name):
    """
    Stops a Docker container by its name.
    Does nothing if the container does not exist or is already stopped.

    :param container_name: Name of the Docker container.
    """
    client = docker.from_env()
    try:
        # Get the container by name
        container = client.containers.get(container_name)
        if container.status not in ["removing", "exited", "dead"]:
            container.stop()
            # Wait until the container is stopped and status is confirmed
            container.wait(condition="removed")
            print(f"Container '{container_name}' has been stopped.")
        else:
            print(f"Container '{container_name}' is already stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()


def main():
    args = parse_inputs()
    cfg = Config(args.cfg_path)
    llm = get_llm(model=args.model, api_key=cfg["ANTHROPIC_API_KEY"], max_tokens=4096)
    ds = load_filter_hf_dataset(args)

    final_stage = args.final_stage
    redirect_log = args.redirect_log
    agent = OrcarAgent(args=args, llm=llm, final_stage=final_stage)
    agent.set_redirect_log(redirect_log)
    output_insts = agent.output_insts.copy()
    for i, inst in enumerate(ds):
        try:
            agent.env.run_with_handle("ls", err_msg="ls failed")
        except Exception as e:
            print(f"Got env exception for instance {inst['instance_id']}: {e}")
            output_insts.extend(agent.output_insts.copy())
            del agent
            time.sleep(5)
            stop_container_by_name(args.container_name)
            print(f"Restarting Container...")
            agent = OrcarAgent(
                args=args, llm=llm, final_stage=final_stage, current_retry=0
            )
            agent.set_redirect_log(redirect_log)
        print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
        agent.run(dict(inst))
    output_insts.extend(agent.output_insts.copy())

    for current_retry in range(args.max_retry):
        ds = ds.filter(
            input_columns=["instance_id"],
            function=lambda x, output_insts=output_insts: x not in output_insts,
        )
        if len(ds) == 0:
            break
        print("Missing instances:")
        for inst in ds:
            print(inst["instance_id"])
        del agent
        time.sleep(5)
        stop_container_by_name(args.container_name)
        print(f"Restarting Container...")
        agent = OrcarAgent(
            args=args, llm=llm, final_stage=final_stage, current_retry=current_retry + 1
        )
        agent.set_redirect_log(redirect_log)
        output_insts = agent.output_insts.copy()
        for i, inst in enumerate(ds):
            try:
                agent.env.run_with_handle("ls", err_msg="ls failed")
            except Exception as e:
                print(f"Got env exception for instance {inst['instance_id']}: {e}")
                output_insts.extend(agent.output_insts.copy())
                del agent
                time.sleep(5)
                stop_container_by_name(args.container_name)
                print(f"Restarting Container...")
                agent = OrcarAgent(
                    args=args,
                    llm=llm,
                    final_stage=final_stage,
                    current_retry=current_retry + 1,
                )
                agent.set_redirect_log(redirect_log)
            print(f"({i+1:03d}/{len(ds):03d}) Current inst: {inst['instance_id']}")
            agent.run(dict(inst))
        output_insts.extend(agent.output_insts.copy())


if __name__ == "__main__":
    main()
