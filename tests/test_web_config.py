from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from autoidea.config import settings
from autoidea.web.configuration import _display_config_path
from autoidea.web.server import create_app


FIXTURE = Path(__file__).parent / "fixtures" / "sample_workspace"


def test_web_config_endpoint_exposes_grouped_masked_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "max_tokens": 32000,
                "openai_api_key": "test-openai-secret-value",
                "show_thinking": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)

    client = TestClient(create_app(FIXTURE))
    response = client.get("/api/config")

    assert response.status_code == 200
    data = response.json()
    assert data["path"] == str(config_path)
    assert data["groups"][0]["id"] == "quick"
    assert "provider" in data["groups"][0]["fields"]

    provider = data["fields"]["provider"]
    assert provider["type"] == "select"
    assert provider["value"] == "openai"
    assert "custom-openai" in provider["options"]
    assert provider["default"] == "openai"

    max_tokens = data["fields"]["max_tokens"]
    assert max_tokens["type"] == "int"
    assert max_tokens["value"] == 32000

    show_thinking = data["fields"]["show_thinking"]
    assert show_thinking["type"] == "bool"
    assert show_thinking["value"] is False

    secret = data["fields"]["openai_api_key"]
    assert secret["secret"] is True
    assert secret["value"] == ""
    assert secret["is_set"] is True
    assert secret["masked_value"].startswith("test")
    assert "secret-value" not in response.text


def test_web_config_endpoint_marks_environment_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: file-model\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)
    monkeypatch.setenv("AUTOIDEA_MODEL", "env-model")

    client = TestClient(create_app(FIXTURE))
    response = client.get("/api/config")

    assert response.status_code == 200
    model = response.json()["fields"]["model"]
    assert model["value"] == "file-model"
    assert model["effective_value"] == "env-model"
    assert model["env_var"] == "AUTOIDEA_MODEL"
    assert model["env_overridden"] is True


def test_web_config_endpoint_treats_environment_secret_as_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: custom-openai\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)
    monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "environment-secret-value")

    client = TestClient(create_app(FIXTURE))
    response = client.get("/api/config")

    assert response.status_code == 200
    secret = response.json()["fields"]["custom_openai_api_key"]
    assert secret["is_set"] is True
    assert secret["stored_is_set"] is False
    assert secret["masked_value"] == ""
    assert secret["env_overridden"] is True
    assert "environment-secret-value" not in response.text


def test_web_config_path_abbreviates_the_current_home_directory() -> None:
    home = Path("/Users/example")

    assert _display_config_path(
        home / ".config" / "autoidea" / "config.yaml",
        home=home,
    ) == "~/.config/autoidea/config.yaml"
    assert _display_config_path(
        Path("/opt/autoidea/config.yaml"),
        home=home,
    ) == "/opt/autoidea/config.yaml"


def test_web_config_patch_persists_typed_values_and_preserves_omitted_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "provider: anthropic\nopenai_api_key: test-existing-secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)

    client = TestClient(create_app(FIXTURE))
    response = client.patch(
        "/api/config",
        json={
            "values": {
                "provider": "custom-openai",
                "max_tokens": "24576",
                "show_thinking": False,
                "enable_web_search": "false",
                "custom_openai_base_url": "https://models.example/v1",
                "openai_api_key": "",
            }
        },
    )

    assert response.status_code == 200
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["provider"] == "custom-openai"
    assert saved["max_tokens"] == 24576
    assert saved["show_thinking"] is False
    assert saved["enable_web_search"] is False
    assert saved["custom_openai_base_url"] == "https://models.example/v1"
    assert saved["openai_api_key"] == "test-existing-secret"
    assert response.json()["fields"]["provider"]["value"] == "custom-openai"


def test_web_config_patch_rejects_unknown_key_and_invalid_int(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt-4o\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)

    client = TestClient(create_app(FIXTURE))

    unknown = client.patch("/api/config", json={"values": {"not_a_setting": "x"}})
    assert unknown.status_code == 400
    assert "Unknown config key" in unknown.json()["detail"]

    invalid = client.patch("/api/config", json={"values": {"max_tokens": "many"}})
    assert invalid.status_code == 400
    assert "max_tokens" in invalid.json()["detail"]


def test_web_config_reset_restores_defaults(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: openai\nmodel: gpt-4o\n", encoding="utf-8")
    monkeypatch.setattr(settings, "get_config_path", lambda: config_path)

    client = TestClient(create_app(FIXTURE))
    response = client.post("/api/config/reset")

    assert response.status_code == 200
    assert not config_path.exists()
    data = response.json()
    assert data["fields"]["provider"]["value"] == "openai"
    assert data["fields"]["model"]["value"] == "gpt-5.6-sol"
