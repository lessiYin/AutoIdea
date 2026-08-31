from __future__ import annotations

import stat
from pathlib import Path

from autoidea.config import AutoIdeaConfig
from autoidea.config import settings


def test_checkpoints_are_fully_automatic_by_default() -> None:
    assert AutoIdeaConfig().auto_approve is True


def test_config_and_state_paths_live_outside_package(monkeypatch, tmp_path: Path) -> None:
    config_home = tmp_path / "config-home"
    state_home = tmp_path / "state-home"
    monkeypatch.delenv("AUTOIDEA_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AUTOIDEA_CONFIG_FILE", raising=False)
    monkeypatch.delenv("AUTOIDEA_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    assert settings.get_config_path() == config_home / "autoidea" / "config.yaml"
    assert settings.get_state_dir() == state_home / "autoidea"


def test_saved_config_uses_private_permissions(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "config.yaml"
    monkeypatch.setenv("AUTOIDEA_CONFIG_FILE", str(config_path))

    settings.save_config(AutoIdeaConfig(provider="openai", openai_api_key="secret"))

    assert settings.load_config().provider == "openai"
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600


def test_runtime_validation_requires_selected_provider_credentials() -> None:
    errors = settings.validate_runtime_config(
        AutoIdeaConfig(provider="custom-openai", model="test", custom_openai_api_key="x")
    )

    assert "Missing CUSTOM_OPENAI_BASE_URL" in errors[0]
