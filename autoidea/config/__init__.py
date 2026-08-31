"""Configuration package for AutoIdea.

Re-exports all public symbols from settings so that existing
``from autoidea.config import X`` imports continue to work.
"""

from .settings import (
    get_config_dir,
    get_config_path,
    get_state_dir,
    AutoIdeaConfig,
    load_config,
    save_config,
    reset_config,
    get_config_value,
    set_config_value,
    list_config,
    get_active_env_override,
    get_effective_config,
    apply_config_to_env,
    validate_runtime_config,
)

__all__ = [
    "get_config_dir",
    "get_config_path",
    "get_state_dir",
    "AutoIdeaConfig",
    "load_config",
    "save_config",
    "reset_config",
    "get_config_value",
    "set_config_value",
    "list_config",
    "get_active_env_override",
    "get_effective_config",
    "apply_config_to_env",
    "validate_runtime_config",
]
