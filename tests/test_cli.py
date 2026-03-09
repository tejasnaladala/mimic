# tests/test_cli.py


def test_version_import():
    from mimic import __version__

    assert __version__ == "0.1.0"


def test_cli_app_exists():
    from mimic.cli.app import app

    assert app is not None
