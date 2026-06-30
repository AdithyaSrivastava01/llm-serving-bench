import pytest
from typer.testing import CliRunner
from llm_bench.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "analyze" in result.stdout
    assert "report" in result.stdout


def test_cli_run_help() -> None:
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.stdout


def test_cli_analyze_help() -> None:
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--results-dir" in result.stdout


def test_cli_report_help() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
