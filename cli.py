from Orcar import OrcarAgent
import subprocess
import argparse

from termcolor import colored

def green(text, attrs=None):
    return colored(text, 'green', attrs=attrs)

def exit_with_help_message(parser):
    print(green("To execute a prompt with a specified execution type", ['bold']))
    # retrieve subparsers from parser
    subparsers_actions = [
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)]
    # there will probably only be one subparser_action,
    # but better save than sorry
    for subparsers_action in subparsers_actions:
        # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print(subparser.format_help())

    print(green("To perform other Orcar operations", ['bold']))
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
    subparsers = parser.add_subparsers(dest='subparser_name')
    parser_execute = subparsers.add_parser('execute', help='Execute a prompt with a specified execution type')
    parser_execute.add_argument('prompt', type=str, help='The prompt to execute')

    orcar_agent = OrcarAgent()
    args = parser.parse_args()
    if args.subparser_name == 'execute':
        response = orcar_agent.chat(args.prompt)
        print(response)
    else:
        exit_with_help_message(parser)
