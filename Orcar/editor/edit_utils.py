import subprocess
from typing import Literal, Optional, Tuple

from llama_index.core.bridge.pydantic import BaseModel

# Original code
# https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo/computer_use_demo/tools

Command = Literal[
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4
TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)
MAX_RESPONSE_LEN: int = 16000


class CLIResult(BaseModel):
    """A ToolResult that can be rendered as a CLI output."""

    output: str | None = None
    error: str | None = None


class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


def maybe_truncate(content: str, truncate_after: int | None = MAX_RESPONSE_LEN):
    """Truncate content and append a notice if content exceeds the specified length."""
    return (
        content
        if not truncate_after or len(content) <= truncate_after
        else content[:truncate_after] + TRUNCATED_MESSAGE
    )


def run_cmd(
    cmd: str,
    timeout: Optional[float] = 120.0,  # seconds
    truncate_after: Optional[int] = MAX_RESPONSE_LEN,
) -> Tuple[int, str, str]:
    """
    Run a shell command synchronously with a timeout.

    Args:
        cmd: The command to run
        timeout: Maximum time to wait for command completion in seconds
        truncate_after: Maximum length for stdout/stderr before truncation

    Returns:
        Tuple of (return_code, stdout, stderr)

    Raises:
        TimeoutError: If command execution exceeds timeout
        subprocess.SubprocessError: For other subprocess-related errors
    """
    try:
        # Run the command with timeout
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,  # Automatically decode output to strings
        )

        # Wait for command completion with timeout
        stdout, stderr = process.communicate(timeout=timeout)

        # Truncate outputs if needed
        if truncate_after:
            stdout = maybe_truncate(stdout, truncate_after=truncate_after)
            stderr = maybe_truncate(stderr, truncate_after=truncate_after)

        return (process.returncode or 0, stdout, stderr)

    except subprocess.TimeoutExpired as exc:
        # Kill the process if it times out
        try:
            process.kill()
            process.wait()  # Clean up the process
        except (ProcessLookupError, AttributeError):
            pass  # Process already terminated

        raise TimeoutError(
            f"Command '{cmd}' timed out after {timeout} seconds"
        ) from exc
