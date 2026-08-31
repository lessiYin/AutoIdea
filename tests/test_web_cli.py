from __future__ import annotations

import os
import yaml
from typer.testing import CliRunner

from autoidea.config import AutoIdeaConfig
from autoidea.cli._app import app
from autoidea.cli import commands  # noqa: F401


def test_web_command_is_registered_in_help() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "web" in result.output


def test_web_command_creates_missing_workspace(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "new-workspace"
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)

    result = CliRunner().invoke(
        app,
        ["web", "--workspace", str(workspace), "--no-open"],
    )

    assert result.exit_code == 0
    assert workspace.is_dir()
    assert "AutoIdea Research Console" in result.output


def test_web_command_does_not_promote_saved_defaults_to_environment(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    workspace = tmp_path / "workspace"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOIDEA_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AUTOIDEA_PROVIDER", raising=False)
    monkeypatch.delenv("AUTOIDEA_MODEL", raising=False)
    monkeypatch.setattr("uvicorn.run", lambda *_args, **_kwargs: None)
    from autoidea.config import set_config_value

    set_config_value("provider", "custom-openai")
    set_config_value("model", "saved-model")

    result = CliRunner().invoke(
        app,
        ["web", "--workspace", str(workspace), "--no-open"],
    )

    assert result.exit_code == 0
    assert "AUTOIDEA_PROVIDER" not in os.environ
    assert "AUTOIDEA_MODEL" not in os.environ


def test_cli_defaults_to_automatic_and_accepts_manual_override(
    tmp_path, monkeypatch
) -> None:
    import autoidea.config as config_module

    calls: list[bool] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda: AutoIdeaConfig(anthropic_api_key="test-key"),
    )
    monkeypatch.setattr(config_module, "apply_config_to_env", lambda _config: None)
    monkeypatch.setattr(config_module, "validate_runtime_config", lambda _config: [])
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr(commands, "set_workspace_root", lambda _path: None)
    monkeypatch.setattr(
        commands,
        "cmd_interactive",
        lambda **kwargs: calls.append(kwargs["auto_approve"]),
    )

    automatic = CliRunner().invoke(app, [])
    manual = CliRunner().invoke(app, ["--manual-checkpoints"])
    conflict = CliRunner().invoke(
        app,
        ["--auto-approve", "--manual-checkpoints"],
        terminal_width=200,
    )

    assert automatic.exit_code == 0
    assert manual.exit_code == 0
    assert calls == [True, False]
    assert conflict.exit_code == 2
    normalized_error = " ".join(conflict.output.replace("│", " ").split())
    assert "cannot be used together" in normalized_error


def test_cli_explicit_provider_and_model_override_runtime_defaults(
    tmp_path, monkeypatch
) -> None:
    import autoidea.config as config_module

    captured: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config_module,
        "get_effective_config",
        lambda: AutoIdeaConfig(
            provider="anthropic",
            model="environment-model",
            anthropic_api_key="test-key",
            custom_openai_api_key="test-custom-key",
            custom_openai_base_url="https://models.example/v1",
        ),
    )
    monkeypatch.setattr(config_module, "apply_config_to_env", lambda _config: None)
    monkeypatch.setattr(config_module, "validate_runtime_config", lambda _config: [])
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr(commands, "set_workspace_root", lambda _path: None)
    monkeypatch.setattr(
        commands,
        "cmd_interactive",
        lambda **kwargs: captured.update(
            provider=kwargs["config"].provider,
            model=kwargs["config"].model,
        ),
    )

    result = CliRunner().invoke(
        app,
        ["--provider", "custom-openai", "--model", "run-specific-model"],
    )

    assert result.exit_code == 0
    assert captured == {
        "provider": "custom-openai",
        "model": "run-specific-model",
    }


def test_cli_config_set_warns_when_environment_still_overrides_value(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    monkeypatch.setenv("AUTOIDEA_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AUTOIDEA_MODEL", "environment-model")

    result = CliRunner().invoke(
        app,
        ["config", "set", "model", "saved-default-model"],
    )

    assert result.exit_code == 0
    saved = yaml.safe_load((config_dir / "config.yaml").read_text(encoding="utf-8"))
    assert saved["model"] == "saved-default-model"
    assert "AUTOIDEA_MODEL" in result.output
    assert "overrides" in result.output


def test_doctor_reports_missing_provider_key(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOIDEA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("AUTOIDEA_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Missing ANTHROPIC_API_KEY" in result.output


def test_doctor_accepts_valid_provider_configuration(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOIDEA_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("AUTOIDEA_PROVIDER", "anthropic")
    monkeypatch.setenv("AUTOIDEA_MODEL", "test-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "ready to start CLI and Web runs" in result.output
