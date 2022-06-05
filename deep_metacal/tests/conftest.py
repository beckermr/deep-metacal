import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slowdeepmdet",
        action="store_true",
        default=False,
        help="run slow tests for deep-field mdet",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slowdeepmdet: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slowdeepmdet"):
        # --runs-low given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --run-slowdeepmdet option to run")
    for item in items:
        if "slowdeepmdet" in item.keywords:
            item.add_marker(skip_slow)
