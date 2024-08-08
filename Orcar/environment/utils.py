import docker
from docker.models.containers import Container
import subprocess
import time
import shlex
from subprocess import PIPE, STDOUT
import logging
import os
import platform
import datetime
import hashlib
from rich.logging import RichHandler
from typing import Any, Callable

_SET_UP_LOGGERS = set()
_ADDITIONAL_HANDLERS = []

logging.TRACE = 5  # type: ignore
logging.addLevelName(logging.TRACE, "TRACE")  # type: ignore
DOCKER_START_UP_DELAY = 1

_STREAM_LEVEL = logging.DEBUG
_FILE_LEVEL = logging.TRACE


def get_logger(name: str) -> logging.Logger:
    """Get logger. Use this instead of `logging.getLogger` to ensure
    that the logger is set up with the correct handlers.
    """
    logger = logging.getLogger(name)
    if name in _SET_UP_LOGGERS:
        # Already set up
        return logger
    handler = RichHandler(
        show_time=bool(os.environ.get("SWE_AGENT_LOG_TIME", False)),
        show_path=False,
    )
    handler.setLevel(_STREAM_LEVEL)
    logger.setLevel(min(_STREAM_LEVEL, _FILE_LEVEL))
    logger.addHandler(handler)
    logger.propagate = False
    _SET_UP_LOGGERS.add(name)
    for handler in _ADDITIONAL_HANDLERS:
        logger.addHandler(handler)
    return logger

logger = get_logger("env_utils")

def get_container(ctr_name: str, image_name: str, persistent: bool = False) -> tuple[subprocess.Popen, set]:
    """
    Get a container object for a given container name and image name

    Arguments:
        ctr_name (str): Name of container
        image_name (str): Name of image
        persistent (bool): Whether to use a persistent container or not
    Returns:
        Container object
    """
    if not image_exists(image_name):
        msg = (
            f"Image {image_name} not found. Please ensure it is built and available. "
            "Please double-check that you followed all installation/setup instructions from the "
            "readme."
        )
        raise RuntimeError(msg)

    if persistent:
        return _get_persistent_container(ctr_name, image_name)
    else:
        return _get_non_persistent_container(ctr_name, image_name)
    

def image_exists(image_name: str) -> bool:
    """
    Check that the image exists and give some better error messages.

    Arguments:
        image_name: Name of image
    Returns:
        bool: True if image exists
    """
    try:
        client = docker.from_env()
    except docker.errors.DockerException as e:
        docker_not_running = any(
            (
                "connection aborted" in str(e).lower(),
                "connection refused" in str(e).lower(),
                "error while fetching server api version" in str(e).lower(),
            ),
        )
        if docker_not_running:
            msg = (
                "Probably the Docker daemon is not running. Please start the Docker daemon and try again. "
                "You might need to allow the use of the docker socket "
                "(https://github.com/princeton-nlp/SWE-agent/issues/159) or symlink the socket "
                "if it's at a non-standard location "
                "(https://github.com/princeton-nlp/SWE-agent/issues/20#issuecomment-2047506005)."
            )
            raise RuntimeError(msg) from e
        raise
    filterred_images = client.images.list(filters={"reference": image_name})
    if len(filterred_images) == 0:
        return False
    elif len(filterred_images) > 1:
        RuntimeError(f"Multiple images found for {image_name}, that's weird.")
    attrs = filterred_images[0].attrs
    if attrs is not None:
        logger.info(
            f"Found image {image_name} with tags: {attrs['RepoTags']}, created: {attrs['Created']} "
            f"for {attrs['Os']} {attrs['Architecture']}.",
        )
    return True

def _get_non_persistent_container(ctr_name: str, image_name: str) -> tuple[subprocess.Popen, set[str]]:
    startup_cmd = [
        "docker",
        "run",
        "-i",
        "--rm",
        "--name",
        ctr_name,
        image_name,
        "/bin/bash",
        "-l",
    ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout(container, lambda: list(), timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # bash PID is always 1 for non-persistent containers
    return container, {
        "1",
    }


def _get_persistent_container(
    ctr_name: str, image_name: str, persistent: bool = False
) -> tuple[subprocess.Popen, set[str]]:
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"name": ctr_name})
    if ctr_name in [c.name for c in containers]:
        container_obj = client.containers.get(ctr_name)
        if container_obj.status in {"created"}:
            container_obj.start()
        elif container_obj.status in {"running"}:
            pass
        elif container_obj.status in {"exited"}:
            container_obj.restart()
        elif container_obj.status in {"paused"}:
            container_obj.unpause()
        else:
            msg = f"Unexpected container status: {container_obj.status}"
            raise RuntimeError(msg)
    else:
        container_obj = client.containers.run(
            image_name,
            command="/bin/bash -l -m",
            name=ctr_name,
            stdin_open=True,
            tty=True,
            detach=True,
            auto_remove=not persistent,
        )
        container_obj.start()
    startup_cmd = [
        "docker",
        "exec",
        "-i",
        ctr_name,
        "/bin/bash",
        "-l",
    ]
    logger.debug("Starting container with command: %s", shlex.join(startup_cmd))
    container = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    time.sleep(DOCKER_START_UP_DELAY)
    # try to read output from container setup (usually an error), timeout if no output
    output = read_with_timeout(container, lambda: list(), timeout_duration=2)
    if output:
        logger.error(f"Unexpected container setup output: {output}")
    # Get the process IDs of the container
    # There should be at least a head process and possibly one child bash process
    bash_pids, other_pids = get_background_pids(container_obj)
    total_time_slept = DOCKER_START_UP_DELAY
    # Let's wait for a maximum of 5 x DOCKER_START_UP_DELAY seconds
    # and then check again.
    while len(bash_pids) > 1 or len(other_pids) > 0:
        time.sleep(1)
        total_time_slept += 1
        bash_pids, other_pids = get_background_pids(container_obj)
        if total_time_slept > 5 * DOCKER_START_UP_DELAY:
            break
    bash_pid = 1
    if len(bash_pids) == 1:
        bash_pid = bash_pids[0][0]
    elif len(bash_pids) > 1 or len(other_pids) > 0:
        msg = (
            "Detected alien processes attached or running. Please ensure that no other agents "
            f"are running on this container. PIDs: {bash_pids}, {other_pids}"
        )
        raise RuntimeError(msg)
    return container, {str(bash_pid), "1"}

def read_with_timeout(container: subprocess.Popen, pid_func: Callable, timeout_duration: int | float) -> str:
    """
    Read data from a subprocess with a timeout.
    This function uses a file descriptor to read data from the subprocess in a non-blocking way.

    Args:
        container: The subprocess container.
        pid_func: A function that returns a list of process IDs (except the PID of the main process).
        timeout_duration: The timeout duration in seconds.

    Returns:
        output: The data read from the subprocess, stripped of trailing newline characters.

    Raises:
        TimeoutError: If the timeout duration is reached while reading from the subprocess.
    """
    buffer = b""
    fd = container.stdout.fileno()
    end_time = time.time() + timeout_duration

    import select

    def ready_to_read(fd) -> bool:
        return bool(select.select([fd], [], [], 0.01)[0])

    while time.time() < end_time:
        pids = pid_func()
        if len(pids) > 0:
            # There are still PIDs running
            time.sleep(0.05)
            continue
        if ready_to_read(fd):
            data = os.read(fd, 4096)
            if data:
                buffer += data
        else:
            # No more data to read
            break
        time.sleep(0.05)  # Prevents CPU hogging

    if container.poll() is not None:
        msg = f"Subprocess exited unexpectedly.\nCurrent buffer: {buffer.decode()}"
        raise RuntimeError(msg)
    if time.time() >= end_time:
        msg = f"Timeout reached while reading from subprocess.\nCurrent buffer: {buffer.decode()}\nRunning PIDs: {pids}"
        raise TimeoutError(msg)
    return buffer.decode()

def get_background_pids(container_obj: Container):
    pids = container_obj.exec_run("ps -eo pid,comm --no-headers").output.decode().split("\n")
    pids = [x.split() for x in pids if x]
    pids = [x for x in pids if x[1] not in {"ps"} and x[0] != "1"]
    bash_pids = [x for x in pids if x[1] == "bash"]
    other_pids = [x for x in pids if x[1] not in {"bash"}]
    return bash_pids, other_pids

def get_children_pids(container_obj: Container, parent_pid: int):
    pids = container_obj.exec_run(f"ps -o pid --no-headers --ppid {parent_pid}").output.decode().split("\n")
    pids = [int(x) for x in pids if x]
    return pids


def run_command_in_container(container: subprocess.Popen, command: str, timeout: int = 5) -> str:
    """
    Run a command in a container and return the output.

    Args:
        container: The container subprocess.
        command: The command to run.
        timeout: The timeout in seconds.

    Returns:
        output: The output of the command.

    Raises:
        TimeoutError: If the command times out.
    """

    container.stdin.write(f"{command}\n")
    container.stdin.flush()
    output = read_with_timeout(container, lambda: list(), timeout)
    logger.debug(f"Run command in container: {command}")
    if output:
        logger.info(f"Command output: {output}")
    return output

def get_ctr_from_name(ctr_name: str) -> Container:
    client = docker.from_env()
    containers = client.containers.list(all=True, filters={"name": ctr_name})
    if ctr_name in [c.name for c in containers]:
        return client.containers.get(ctr_name)
    else:
        raise ValueError(f"get_ctr_from_name(): Cannot find container {ctr_name}")
    

def run_bash_in_ctr(ctr: Container, command: str) -> str:
    """
    Run a command with a new process in started container and return the output.

    Args:
        ctr: The container object.
        command: The command to run.

    Returns:
        output: The output of the command.
    """
    ctr_name: str = ctr.name
    startup_cmd = [
        "docker",
        "exec",
        "-i",
        ctr_name,
        "/bin/bash",
        "-l"
    ]
    #logger.debug(f"Starting process in container {ctr_name}")
    ctr_subprocess = subprocess.Popen(
        startup_cmd,
        stdin=PIPE,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
        bufsize=1,  # line buffered
    )
    ctr_subprocess.stdin.write(f"echo $$\n")
    ctr_subprocess.stdin.flush()
    output = ""
    while (output == ""):
        output = read_with_timeout(ctr_subprocess, lambda: list(), 5)
        time.sleep(0.05)
    ctr_bash_pid = output.split('\n')[0]
    logger.debug(f"Started bash process {ctr_bash_pid} in container {ctr_name}")

    ctr_subprocess.stdin.write(f"{command}\n")
    ctr_subprocess.stdin.flush()
    output = read_with_timeout(ctr_subprocess, lambda: get_children_pids(ctr, ctr_bash_pid), 1)
    #if output:
    #    logger.info(f"Command output: {output}")
    exit_code = ctr_subprocess.returncode
    ctr_subprocess.stdin.close()
    return f"Exit code: {exit_code}, Output:\n{output}"

def generate_container_name(image_name: str) -> str:
    """Return name of container"""
    process_id = str(os.getpid())
    current_time = str(datetime.datetime.now())
    unique_string = current_time + process_id
    hash_object = hashlib.sha256(unique_string.encode())
    image_name_sanitized = image_name.replace("/", "-")
    image_name_sanitized = image_name_sanitized.replace(":", "-")
    return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"