"""Misc helpers."""

import logging
from termcolor import colored


def color(string: str, color_name: str = 'yellow') -> str:
    """Returns colored string for output to terminal"""
    return colored(string, color_name)


def print_update(message: str, width: int = 140, fillchar: str = ":") -> str:
    """Prints an update message
    Args:
        message (str): message
        width (int): width of new update message
        fillchar (str): character to be filled to L and R of message
    Returns:
        str: print-ready update message
    """
    message = message.center(len(message) + 2, " ")
    print(color(message.center(width, fillchar)))
