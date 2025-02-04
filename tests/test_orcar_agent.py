import argparse
import time

import docker

from Orcar import OrcarAgent
from Orcar.gen_config import Config, get_llm
from Orcar.load_cache_dataset import load_filter_hf_dataset

args_dict = {
    # "model": "claude-3-5-sonnet-20241022",
    # "model": "gpt-4o",
    "model": "gemini-2.0-flash-exp",
    "image": "sweagent/swe-agent:latest",
    "dataset": "SWE-bench_common",
    # "dataset": "princeton-nlp/SWE-bench_Lite",
    "persistent": True,
    "container_name": "test",
    "split": "test",
    "max_retry": 2,
    # "idx_list": [0, 3],
    # "idx_range": [0, 1],
    # Short Issue Test
    # "filter_instance": "^(matplotlib__matplotlib-23476)$",
    # "filter_instance": "^(sphinx-doc__sphinx-11445|sphinx-doc__sphinx-8595|sympy__sympy-12419)$",
    # "filter_instance": "^(sympy__sympy-12419)$",
    # "filter_instance": "^(astropy__astropy-14365|django__django-11001|pylint-dev__pylint-7993)$",
    # "filter_instance": "^(django__django-12983|pylint-dev__pylint-7228)$",
    # Long Issue Test
    # "filter_instance": "^(sympy__sympy-23262)$",
    # "filter_instance": "^(pytest-dev__pytest-5692)$",
    "filter_instance": "^(astropy__astropy-12907)$",
    # "filter_instance": (
    #     "^(sympy__sympy-21612|pytest-dev__pytest-7432|matplotlib__matplotlib-24149|"
    #     "sympy__sympy-16792|django__django-11999|matplotlib__matplotlib-25332|"
    #     "scikit-learn__scikit-learn-13496)$"
    # ),
    # whole repo
    # "filter_instance": ".*",
    # internal error
    # "filter_instance": "^(django__django-14580)$",
    # "filter_instance": "^(django__django-15814)$",
    # Multi Issue Test
    # "filter_instance": "^(pylint-dev__pylint-7080|matplotlib__matplotlib-26020|pytest-dev__pytest-7490)$",
    # "filter_instance": ".*",
}


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


def test_agent():
    args = argparse.Namespace(**args_dict)
    cfg = Config("../key.cfg")
    llm = get_llm(model=args.model, max_tokens=4096, orcar_config=cfg)
    ds = load_filter_hf_dataset(args)

    # final_stage = "extract"
    final_stage = "search"
    # final_stage = "edit"
    redirect_log = True
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
    test_agent()
