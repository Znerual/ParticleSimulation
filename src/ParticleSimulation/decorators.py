import typing

import numpy as np


def for_all_objects(func):
    """Decorator Applying a function to all objects, other arguments are passed directly"""

    def wrapper(caller, objectIDs, *args, **kwargs):
        for objID in objectIDs:
            func(caller, objID, *args, **kwargs)

    return wrapper


def for_all_positional(func):
    """Decorator Applying a function to all objects, keyword arguments are passed directly,
    positional arguments are reduced for each object as well"""

    def wrapper(caller, objectIDs, *args, **kwargs):
        if len(args) == 0:
            return for_all_objects(func)
        elif len(args) == 1:
            for objID, parameter in zip(objectIDs, *args):
                func(caller, objID, parameter, **kwargs)
        else:
            for objID, *parameter in zip(objectIDs, *args):
                func(caller, objID, *parameter, **kwargs)

    return wrapper


def for_all(func):
    """Decorator Applying a function to all objects, keyword arguments and
    positional arguments are reduced for each object as well"""

    def wrapper(caller, objectIDs: np.ndarray[typing.Tuple[int], np.dtype[np.uint16]], *args, **kwargs):
        keywords = kwargs.keys()
        amount_keywords = len(keywords)
        if amount_keywords == 0:
            return for_all_positional(func)

        else:
            for objID, parameter in zip(objectIDs, *args, **kwargs):  # type: ignore
                func(caller, objID, *parameter[:-amount_keywords], **dict(zip(keywords, parameter[:amount_keywords])))
    return wrapper