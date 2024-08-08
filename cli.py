from Orcar import OrcarAgent
import subprocess
import argparse

from termcolor import colored
from Orcar.key_config import Config
from Orcar.environment.utils import get_container, generate_container_name


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
        dest="model",
        default=default_model,
        type=str,
        help=f"The LLM model (only support OpenAI now) (default: {default_model})",
    )
    parser_execute.add_argument("prompt", type=str, help="The prompt to execute")

    parser_docker = subparsers.add_parser(
        "docker", help="Execute a prompt with a specified execution type in a docker image"
    )
    parser_docker.add_argument(
        "--model",
        dest="model",
        default=default_model,
        type=str,
        help=f"The LLM model (only support OpenAI now) (default: {default_model})",
    )
    parser_docker.add_argument(
        "--image",
        dest="image",
        default=default_docker_image,
        type=str,
        help=f"The base docker image (default: {default_docker_image})",
    )
    parser_docker.add_argument("prompt", type=str, help="The prompt to execute")

    
    args = parser.parse_args()
    if args.command == "execute":
        orcar_agent = OrcarAgent(args, Config('./key.cfg'))
        response = orcar_agent.chat(args.prompt)
        print(response)
    elif args.command == "docker":
        ctr_name = generate_container_name(args.image)
        docker_ctr_subprocess = get_container(ctr_name=ctr_name, image_name=args.image)[0]
        orcar_agent = OrcarAgent(args, Config('./key.cfg'), ctr_name)
        response = orcar_agent.chat(args.prompt)
        print(response)
        docker_ctr_subprocess.stdin.close()
    else:
        exit_with_help_message(parser)
