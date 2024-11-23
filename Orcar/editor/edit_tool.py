import os
import subprocess
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import get_args

from ..search.build_graph import LocInfo
from ..search.search_tool import SearchManager
from .edit_utils import (
    SNIPPET_LINES,
    CLIResult,
    Command,
    ToolError,
    maybe_truncate,
    run_cmd,
)

ReviseInfo = namedtuple("ReviseInfo", ["content", "file", "start_line", "end_line"])


# Revised from anthropic
# https://github.com/anthropics/anthropic-quickstarts/blob/main/computer-use-demo/computer_use_demo/tools/edit.py
class Editor:
    """
    An filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    _file_history: dict[Path, list[str]]
    name = "StringReplaceEditor"

    def __init__(self, repo_path: str) -> None:
        self._file_history = defaultdict(list)
        self.repo_path = repo_path  # Path to the repository
        self.search_manager = SearchManager(repo_path)
        super().__init__()

    def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ):
        _path = Path(path)
        self.validate_path(command, _path)
        if command == "view":
            return self.view(_path, view_range)
        elif command == "create":
            if not file_text:
                raise ToolError("Parameter `file_text` is required for command: create")
            self.write_file(_path, file_text)
            self._file_history[_path].append(file_text)
            return CLIResult(output=f"File created successfully at: {_path}")
        elif command == "str_replace":
            if not old_str:
                raise ToolError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            return self.str_replace(_path, old_str, new_str)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if not new_str:
                raise ToolError("Parameter `new_str` is required for command: insert")
            return self.insert(_path, insert_line, new_str)
        elif command == "undo_edit":
            return self.undo_edit(_path)
        raise ToolError(
            f'Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command))}'
        )

    def editor_execute(
        self,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ):
        """
        Execute the editor tool with the given parameters.
        Usage:
        1. editor_execute(command="view", path="/path/to/file", view_range=[1, 10])
            View the content of a file.
            Arg view_range is optional and should be a list of two integers. Leave it as None to view the entire file.
        2. editor_execute(command="create", path="/path/to/file", file_text="file content")
            Create a new file with the given content.
        3. editor_execute(command="str_replace", path="/path/to/file", old_str="old string", new_str="new string")
            Replace the old string with the new string in the file content.
        4. editor_execute(command="insert", path="/path/to/file", insert_line=5, new_str="new line")
            Insert a new line at the specified line in the file content.
        5. editor_execute(command="undo_edit", path="/path/to/file")
            Undo the last edit to the file content.

        """
        return self.__call__(
            command=command,
            path=path,
            file_text=file_text,
            view_range=view_range,
            old_str=old_str,
            new_str=new_str,
            insert_line=insert_line,
            **kwargs,
        )

    def validate_path(self, command: str, path: Path):
        """
        Check that the path/command combination is valid.
        """
        # Check if its an absolute path
        if not path.is_absolute():
            suggested_path = Path("") / path
            raise ToolError(
                f"The path {path} is not an absolute path, it should start with `/`. Maybe you meant {suggested_path}?"
            )
        # Check if path exists
        if not path.exists() and command != "create":
            raise ToolError(
                f"The path {path} does not exist. Please provide a valid path."
            )
        if path.exists() and command == "create":
            raise ToolError(
                f"File already exists at: {path}. Cannot overwrite files using command `create`."
            )
        # Check if the path points to a directory
        if path.is_dir():
            if command != "view":
                raise ToolError(
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

    def view(self, path: Path, view_range: list[int] | None = None):
        """Implement the view command"""
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            _, stdout, stderr = run_cmd(rf"find {path} -maxdepth 2 -not -path '*/\.*'")
            if not stderr:
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            return CLIResult(output=stdout, error=stderr)

        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError(
                    "Invalid `view_range`. It should be a list of two integers."
                )
            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. It's second element `{final_line}` should be larger or equal than its first `{init_line}`"
                )

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    def str_replace(self, path: Path, old_str: str, new_str: str | None):
        """Implement the str_replace command, which replaces old_str with new_str in the file content"""
        # Read the file content
        file_content = self.read_file(path).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # Check if old_str is unique in the file
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
            )

        # Replace old_str with new_str
        new_file_content = file_content.replace(old_str, new_str)

        # Write the new content to the file
        self.write_file(path, new_file_content)

        # Save the content to history
        self._file_history[path].append(file_content)

        # Create a snippet of the edited section
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # Prepare the success message
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return CLIResult(output=success_msg)

    def insert(self, path: Path, insert_line: int, new_str: str):
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return CLIResult(output=success_msg)

    def undo_edit(self, path: Path):
        """Implement the undo_edit command."""
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return CLIResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path):
        """Read the content of a file from a given path; raise a ToolError if an error occurs."""
        try:
            return path.read_text()
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write the content of a file to a given path; raise a ToolError if an error occurs."""
        try:
            path.write_text(file)
        except Exception as e:
            raise ToolError(f"Ran into {e} while trying to write to {path}") from None

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ):
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )

    def _get_bug_code(self, bug_query: str, file_path: str) -> ReviseInfo | None:
        """Get the code snippet in the file."""
        # Get the code snippet in the file
        # print(f"Searcing for bug query: {bug_query}, in file: {file_path}")
        locinfo: LocInfo = self.search_manager._get_query_in_file(file_path, bug_query)
        if locinfo is None:
            return None
        loc = locinfo.loc
        joined_path = os.path.join(self.repo_path, loc.file_name)
        content = self.search_manager._get_code_snippet(
            joined_path, loc.start_line, loc.end_line
        )

        return ReviseInfo(content, joined_path, loc.start_line, loc.end_line)

    def _edit_with_new_code(self, revise_info: ReviseInfo, new_code: str) -> str:
        """Edit the code snippet with new code chunk."""
        # Edit the code snippet with new code
        file = revise_info.file
        start_line = revise_info.start_line
        end_line = revise_info.end_line

        try:
            # Read the file
            with open(file, "r") as f:
                lines = f.readlines()

            # Split new_code into lines
            new_lines = new_code.splitlines(keepends=True)

            # Replace the old lines with new ones
            lines[start_line - 1 : end_line] = new_lines
            updated_content = "".join(lines)

            # Write back to file
            with open(file, "w") as f:
                f.write(updated_content)
        except Exception as e:
            raise Exception(f"Error: Unable to get access to file {file}. {e}")

        return updated_content

    def revise_bug(self, bug_query: str, file_path: str, new_code: str) -> None:
        """
        Chunk Edit:
        Revise the bug with new code.
        """
        # Get the code snippet in the file
        revise_info = self._get_bug_code(bug_query, file_path)

        # Edit the code snippet with new code
        self._edit_with_new_code(revise_info, new_code)

        # Get relative path from repo root
        file_name = os.path.relpath(revise_info.file, self.repo_path)

        # Use git commands with explicit working directory
        subprocess.run(["git", "add", file_name], cwd=self.repo_path, check=True)

    # Create a git diff patch
    def create_patch(self) -> str:
        """
        Create a git diff patch.
        To tract all added and deleted files, use `git add --all` before creating the patch.
        """
        subprocess.run(
            ["git", "add", "--all"],
            cwd=self.repo_path,
            check=True,
        )
        result = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
