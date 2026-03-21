import numpy as np


def ensure_numpy_legacy_aliases() -> None:
    legacy_aliases = {
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
    }
    for alias_name, alias_value in legacy_aliases.items():
        if not hasattr(np, alias_name):
            setattr(np, alias_name, alias_value)
