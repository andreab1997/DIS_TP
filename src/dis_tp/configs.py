"""Tools related to the configuration file handling."""
import copy
import pathlib

import tomli

name = "dis_tp.toml"
"Name of the config file (wherever it is placed)"

# better to declare immediately the correct type
configs = {}
"Holds loaded configurations"


def defaults(base_configs):
    """Provide additional defaults.

    Parameters
    ----------
    base_config : dict
        user provided configuration

    Returns
    -------
    configs_ : dict
        enhanced configuration

    Note
    ----
    The general rule is to never replace user provided input.
    """
    configs_ = copy.deepcopy(base_configs)
    enhance_paths(configs_)

    return configs_


def enhance_paths(configs_):
    """Check required path and enhance them with root path.

    The changes are done inplace.

    Parameters
    ----------
    configs_ : dict
        configuration
    """
    # required keys without default
    for key in ["theory_cards", "operator_cards", "results"]:
        if key not in configs_["paths"]:
            raise ValueError(f"Configuration is missing a 'paths.{key}' key")
        if pathlib.Path(configs_["paths"][key]).anchor == "":
            configs_["paths"][key] = configs_["paths"]["root"] / configs_["paths"][key]
        else:
            configs_["paths"][key] = pathlib.Path(configs_["paths"][key])


def detect(path=None):
    """Autodetect configuration file path.

    Parameters
    ----------
    path : str or os.PathLike
        user provided guess

    Returns
    -------
    pathlib.Path :
        file path
    """
    # If a path is provided then we only look for the dis_tp.toml file there.
    if path is not None:
        path = pathlib.Path(path)
        configs_file = path / name if path.is_dir() else path
        if configs_file.is_file():
            return configs_file
        else:
            raise ValueError(
                "Provided path is not pointing to (or does not contain) the dis_tp.toml file"
            )

    # If no path is provided we need to look after the file.
    # We want to check cwd and all its parent folders (without their subfolders
    # of course) up to root.
    cwd_path = pathlib.Path.cwd().absolute()
    paths = [cwd_path] + list(cwd_path.parents)

    for p in paths:
        configs_file = p / name

        if configs_file.is_file():
            return configs_file

    raise FileNotFoundError("No configurations file detected.")


def load(path=None):
    """Load config file.

    Parameters
    ----------
    path : str or os.PathLike
        file path

    Returns
    -------
    loaded : dict
        configuration dictionary
    """
    path = detect(path)

    with open(path, "rb") as fd:
        loaded = tomli.load(fd)

    if "paths" not in loaded:
        loaded["paths"] = {}
    if "root" not in loaded["paths"]:
        loaded["paths"]["root"] = pathlib.Path(path).parent

    return loaded
