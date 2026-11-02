def pytest_configure(config):
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
