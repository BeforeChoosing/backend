from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import audit, auth, career, profile, trial


@pytest.fixture
def authenticated_client(tmp_path, monkeypatch):
    """Business API tests use real authentication, never bypass middleware.

    Separate security tests deliberately construct anonymous clients instead.
    All auth/audit writes stay inside the test's temporary directory.
    """
    settings = replace(main.settings, profile_db_path=str(tmp_path / 'accounts.db'))
    monkeypatch.setattr(main, 'settings', settings)
    for module in (audit, auth, career, profile, trial):
        monkeypatch.setattr(module, 'get_settings', lambda: settings)
    with TestClient(main.app) as client:
        response = client.post('/api/v1/auth/register', json={
            'email': 'business-test@example.com', 'password': 'test-password-123',
        })
        assert response.status_code == 201
        client.headers['Authorization'] = 'Bearer ' + response.json()['access_token']
        yield client
