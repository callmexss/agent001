import logging
import re
from pathlib import Path, PureWindowsPath


def generate_safe_filename(filename: str):
    # Replace spaces and - with underscores
    filename = filename.replace(" ", "_")
    filename = filename.replace("-", "_")

    # Remove invalid characters
    filename = re.sub(r"[^0-9a-zA-Z_-]", "", filename)

    # Remove consecutive periods
    filename = re.sub("\.\.+", ".", filename)

    # Truncate to 63 characters
    filename = filename[:63]

    # Remove leading and trailing non-alphanumeric characters
    filename = re.sub("^[^0-9a-zA-Z]+|[^0-9a-zA-Z]+$", "", filename)

    # Ensure the filename is not an IPv4 address
    if re.match("^([0-9]{1,3}\.){3}[0-9]{1,3}$", filename):
        raise ValueError("Filename cannot be a valid IPv4 address")

    return filename


def windows_path_to_wsl(windows_path: str) -> str:
    path = windows_path.strip('"').split(":")[-1].replace("\\", "/")
    logging.info(path)
    wsl_path = f"/mnt/c{path}"
    logging.info(f"Converting {windows_path} to {wsl_path}")
    if Path(wsl_path).exists():
        return wsl_path
