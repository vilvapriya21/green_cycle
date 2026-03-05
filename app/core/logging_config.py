import logging


def setup_logging() -> None:
    """
    Configure application-wide logging.

    Ensures consistent formatting across all modules.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
