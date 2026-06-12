import pytest

@pytest.mark.integration
def test_dummy_integration():
    """Dummy integration test to prevent pytest from exiting with an error code when no integration tests are found."""
    assert True
