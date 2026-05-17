"""AMATIS Test Suite.

Test organization:
    - unit/: Component tests (fast, isolated)
    - integration/: Multi-component tests (slower)
    - fixtures/: Test data and helpers
    - conftest.py: Shared fixtures and configuration

Run tests:
    pytest tests/                    # All tests
    pytest tests/unit/               # Unit tests only
    pytest -m "not slow"             # Skip slow tests
    pytest --cov=amatix --cov-report=html  # With coverage
"""
