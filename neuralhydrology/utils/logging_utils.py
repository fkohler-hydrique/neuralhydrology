import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def setup_logging(log_file: str) -> None:
    """Initialize logging to `log_file` and stdout.

    This:
    - Logs INFO and above to both the given file and stdout.
    - Installs a `sys.excepthook` that logs uncaught exceptions.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    """
    file_handler = logging.FileHandler(filename=log_file)
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        handlers=[file_handler, stdout_handler],
        level=logging.INFO,
        format="%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Make sure we log uncaught exceptions
    def exception_logging(exc_type, value, tb):
        LOGGER.exception("Uncaught exception", exc_info=(exc_type, value, tb))

    sys.excepthook = exception_logging

    LOGGER.info("Logging to %s initialized.", log_file)


def get_git_hash() -> Optional[str]:
    """Get git commit hash of the project if it is a git repository.

    Returns
    -------
    Optional[str]
        Git commit hash if project is a git repository, else None.
    """
    current_dir = str(Path(__file__).absolute().parent)
    try:
        # Check if we are inside a git repository
        if subprocess.call(
            ["git", "-C", current_dir, "branch"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        ) == 0:
            return (
                subprocess.check_output(
                    ["git", "-C", current_dir, "describe", "--always"],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode("ascii")
            )
    except OSError:
        # Likely: git not installed or inaccessible
        return None

    return None


def save_git_diff(run_dir: Path) -> None:
    """Try to store the git diff to a file in the run directory.

    The diff includes staged and unstaged changes (`git diff HEAD`).
    If the diff did not change since the last saved one, it is not written again.

    Parameters
    ----------
    run_dir : Path
        Directory of the current run.
    """
    base_dir = str(Path(__file__).absolute().parent)
    try:
        out = subprocess.check_output(
            ["git", "-C", base_dir, "diff", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        LOGGER.warning(
            "Could not store git diff, likely because git is not installed "
            "or because your version of git is too old (< 1.8.5)."
        )
        return

    new_diff = out.strip().decode("utf-8")
    if not new_diff:
        return

    existing_diffs = list(run_dir.glob("neuralhydrology*.diff"))
    if existing_diffs:
        last_diff_path = run_dir / f"neuralhydrology-{len(existing_diffs) - 1}.diff"
        try:
            with last_diff_path.open("r", encoding="utf-8") as last_diff_file:
                last_diff = last_diff_file.read()
        except OSError:
            last_diff = ""

        if last_diff == new_diff:
            LOGGER.info(
                "Git repository contains uncommitted changes that are already stored in %s.",
                last_diff_path,
            )
            return

    file_path = run_dir / f"neuralhydrology-{len(existing_diffs)}.diff"
    LOGGER.warning(
        "Git repository contains uncommitted changes. Writing diff to %s.",
        file_path,
    )
    with file_path.open("w", encoding="utf-8") as diff_file:
        diff_file.write(new_diff)
