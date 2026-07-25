"""Shared pytest configuration.

A plain ``pytest`` run must pass offline, on a laptop, with no credentials and no large
downloads. Two markers are therefore opt-in rather than opt-out:

- ``network`` needs internet access
- ``slow`` downloads model weights or datasets measured in gigabytes

Both are skipped unless explicitly requested. ``--strict-markers`` in ``pyproject.toml``
means a typo in a marker name is an error rather than a silently-never-run test.

    pytest                       # offline, fast, the default
    pytest -m network            # only the network tests
    pytest --run-network         # everything offline plus the network tests
    pytest -m slow --run-slow    # the large downloads, on purpose
"""

from __future__ import annotations

import pytest

OPT_IN_MARKERS = {
    "network": ("--run-network", "needs network access"),
    "slow": ("--run-slow", "downloads large files or model weights"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    for marker, (flag, reason) in OPT_IN_MARKERS.items():
        parser.addoption(
            flag,
            action="store_true",
            default=False,
            help=f"run tests marked {marker!r} ({reason})",
        )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip opt-in tests unless the user asked for them by flag or by ``-m``."""
    selected = config.getoption("-m", default="") or ""
    for marker, (flag, reason) in OPT_IN_MARKERS.items():
        # An explicit "-m network" is a request to run them, so do not second-guess it.
        if config.getoption(flag) or marker in selected:
            continue
        skip = pytest.mark.skip(reason=f"{reason}; run with {flag} or -m {marker}")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
