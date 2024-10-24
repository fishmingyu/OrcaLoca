import os
import subprocess
from collections import namedtuple

from ..search.build_graph import LocInfo
from ..search.search_tool import SearchManager

ReviseInfo = namedtuple("ReviseInfo", ["content", "file", "start_line", "end_line"])


class Editor:
    def __init__(self, repo_path: str) -> None:
        self.repo_path = repo_path  # Path to the repository
        self.search_manager = SearchManager(repo_path)

    def _get_bug_code(self, bug_query: str, file_path: str) -> str:
        """Get the code snippet in the file."""
        # Get the code snippet in the file
        locinfo: LocInfo = self.search_manager._get_query_in_file(file_path, bug_query)
        loc = locinfo.loc

        joined_path = os.path.join(self.repo_path, loc.file_name)
        content = self.search_manager._get_code_snippet(
            joined_path, loc.start_line, loc.end_line
        )

        return ReviseInfo(content, joined_path, loc.start_line, loc.end_line)

    def _edit_with_new_code(self, revise_info: ReviseInfo, new_code: str) -> str:
        """Edit the code snippet with new code."""
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
        """Revise the bug with new code."""
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
        """Create a git diff patch."""
        result = subprocess.run(
            ["git", "diff", "--cached", "--no-index"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
