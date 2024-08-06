import subprocess


def shell(shell_command: str) -> str:
    """
    Executes a shell command and returns the output (result).
    """
    process = subprocess.Popen(
        shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, _ = process.communicate()
    exit_code = process.returncode
    return f"Exit code: {exit_code}, Output:\n{output.decode()}"