import logging
from pathlib import Path


def get_logger(
    name: str = "a3tgcn",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create and configure a project logger.

    Parameters
    ----------
    name : str
        Logger name.

    log_file : str or None
        Optional file path for saving logs.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger