from typing import Type

from . import constants as c
from .model_utils import ModelUtils

def is_transform_type(module_type: Type):
    is_transform = False
    for transform_type in c.BASE_TRANSFORM_TYPES:
        if issubclass(module_type, transform_type):
            is_transform = True
            break
    return is_transform
