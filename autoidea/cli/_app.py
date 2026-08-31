"""Typer application objects -- deliberately free of intra-package imports.

Keeping this module import-free (aside from ``typer`` itself) prevents
circular-dependency issues that arise when commands.py or interactive.py
try to import the ``app`` or ``config_app`` objects during registration.
"""

import typer  # type: ignore[import-untyped]

# Root CLI application
app = typer.Typer(
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Config subcommand group
config_app = typer.Typer(
    help="Configuration management commands",
    invoke_without_command=True,
)
app.add_typer(config_app, name="config")
