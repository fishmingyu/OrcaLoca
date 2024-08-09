from Orcar import OrcarAgent
import argparse

from termcolor import colored
from Orcar.key_config import Config
from Orcar.environment.utils import get_container, generate_container_name, pause_persistent_container


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
        action='store_true',
        help=f"Should JIT be used to parallelly call function tools",
    )
    parser_execute.add_argument(
        "-d", "--docker",
        action='store_true',
        help=f"Is the prompt executed in local env or docker",
    )
    parser_execute.add_argument(
        "--image",
        default=default_docker_image,
        help=f"The base docker image (default: {default_docker_image})",
    )
    parser_execute.add_argument(
        "-p", "--persistent",
        action='store_true',
        help=f"Is the prompt executed in local env or docker",
    )
    parser_execute.add_argument(
        "--container_name",
        help=f"The name of container, will be generated from image name if not given",
    )
    parser_execute.add_argument("prompt", type=str, help="The prompt to execute")

    
    args = parser.parse_args()
    if args.command == "execute":
        if (args.docker):
            ctr_name = args.container_name
            if ctr_name is None:
                ctr_name = generate_container_name(args.image)
            docker_ctr_subprocess = get_container(ctr_name=ctr_name, image_name=args.image, persistent=args.persistent)[0]
            orcar_agent = OrcarAgent(args, Config('./key.cfg'), args.enable_jit, ctr_name)
            response = orcar_agent.chat(args.prompt)
            print(response)
            docker_ctr_subprocess.stdin.close()
            if (args.persistent):
                pause_persistent_container(ctr_name)
        else:
            orcar_agent = OrcarAgent(args, Config('./key.cfg'), args.enable_jit)
            response = orcar_agent.chat(args.prompt)
            print(response)
    else:
        exit_with_help_message(parser)
