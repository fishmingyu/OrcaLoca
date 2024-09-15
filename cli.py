import argparse
import json

from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.openai import OpenAI
from termcolor import colored

from Orcar import ExtractAgent, OrcarAgent
from Orcar.environment.benchmark import BenchmarkEnv
from Orcar.environment.utils import (
    ContainerBash,
    generate_container_name,
    get_container,
    get_logger,
    pause_persistent_container,
)
from Orcar.key_config import Config
from Orcar.load_cache_dataset import load_filter_hf_dataset
from Orcar.types import ExtractOutput

logger = get_logger("cli")


def green(text, attrs=None):
    return colored(text, "green", attrs=attrs)


def exit_with_help_message(parser):
    print(green("To execute a prompt with a specified execution type", ["bold"]))
    # retrieve subparsers from parser
    subparsers_actions = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    # there will probably only be one subparser_action,
    # but better save than sorry
    for subparsers_action in subparsers_actions:
        # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print(subparser.format_help())

    print(green("To perform other Orcar operations", ["bold"]))
    parser.print_help()

    parser.exit()


class ArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(ArgumentParser, self).__init__(*args, **kwargs)
        self.error = self.error
        self.exit = self.exit

    def error(self, message):
        exit_with_help_message(self)


def main():
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command", description="valid commands")

    parser_execute = subparsers.add_parser(
        "execute", help="Execute a prompt with a specified execution type"
    )
    default_model = "gpt-4o"
    default_docker_image = "sweagent/swe-agent:latest"
    parser_execute.add_argument(
        "--model",
        default=default_model,
        help=f"The LLM model (only support OpenAI now) (default: {default_model})",
    )
    parser_execute.add_argument(
        "--enable_jit",
        action="store_true",
        help=f"Should JIT be used to parallelly call function tools",
    )
    parser_execute.add_argument(
        "-d",
        "--docker",
        action="store_true",
        help=f"Is the prompt executed in local env or docker",
    )
    parser_execute.add_argument(
        "--image",
        default=default_docker_image,
        help=f"The base docker image (default: {default_docker_image})",
    )
    parser_execute.add_argument(
        "-p",
        "--persistent",
        action="store_true",
        help=f"Is the prompt executed in local env or docker",
    )
    parser_execute.add_argument(
        "-c",
        "--container_name",
        help=f"The name of container, will be generated from image name if not given",
    )
    parser_execute.add_argument("prompt", type=str, help="The prompt to execute")

    parser_execute = subparsers.add_parser(
        "benchmark", help="Run a given huggingface benchmark following swe-bench format"
    )
    default_dataset = "princeton-nlp/SWE-bench_Lite"
    parser_execute.add_argument(
        "--model",
        default=default_model,
        help=f"The LLM model (only support OpenAI now) (default: {default_model})",
    )
    parser_execute.add_argument(
        "--image",
        default=default_docker_image,
        help=f"The base docker image (default: {default_docker_image})",
    )
    parser_execute.add_argument(
        "--dataset",
        default=default_dataset,
        help=f"The target dataset (default: {default_dataset})",
    )
    parser_execute.add_argument(
        "-p",
        "--persistent",
        action="store_true",
        help=f"Is the prompt executed in local env or docker",
    )
    parser_execute.add_argument(
        "-c",
        "--container_name",
        help=f"The name of container, will be generated from image name if not given",
    )
    parser_execute.add_argument(
        "-s",
        "--split",
        default="test",
        help=f"The split you care about, e.g. dev or test",
    )
    parser_execute.add_argument(
        "-f",
        "--filter_instance",
        default=".*",
        help=f"Filter the instances you care about with RegEx",
    )

    args = parser.parse_args()
    cfg = Config("./key.cfg")
    if args.command == "execute":
        if args.docker:
            ctr_name = args.container_name
            if ctr_name is None:
                ctr_name = generate_container_name(args.image)
            docker_ctr_subprocess = get_container(
                ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
            )[0]
            ctr_bash = ContainerBash(
                ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name
            )

            orcar_agent = OrcarAgent(args, cfg, args.enable_jit, ctr_bash)
            response = orcar_agent.chat(args.prompt)
            logger.debug(response)

            ctr_bash.ctr_subprocess.stdin.close()
            if args.persistent:
                pause_persistent_container(ctr_bash)
        else:
            orcar_agent = OrcarAgent(args, cfg, args.enable_jit)
            response = orcar_agent.chat(args.prompt)
            logger.debug(response)
    elif args.command == "benchmark":
        ctr_name = args.container_name
        if ctr_name is None:
            ctr_name = generate_container_name(args.image)
        docker_ctr_subprocess = get_container(
            ctr_name=ctr_name, image_name=args.image, persistent=args.persistent
        )[0]
        ctr_bash = ContainerBash(
            ctr_subprocess=docker_ctr_subprocess, ctr_name=ctr_name
        )

        ds = load_filter_hf_dataset(args)
        llm = OpenAI(
            model=args.model,
            api_key=cfg["OPENAI_API_KEY"],
            api_base=cfg["OPENAI_API_BASE_URL"],
        )
        benchmark_env = BenchmarkEnv(args, ctr_bash)
        extractor = ExtractAgent(llm=llm, env=benchmark_env)

        for inst in ds:
            benchmark_env.setup(inst)
            agent_chat_response: AgentChatResponse = extractor.chat(
                json.dumps(dict(inst))
            )
            extract_output = ExtractOutput.parse_raw(agent_chat_response.response)
            logger.debug(extract_output)

        # Run Test on Benchmark
        # TBD

        ctr_bash.ctr_subprocess.stdin.close()
        if args.persistent:
            pause_persistent_container(ctr_bash)
    else:
        exit_with_help_message(parser)
