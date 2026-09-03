from dataclasses import replace
import json

from fastapi import APIRouter
from fastapi.testclient import TestClient

from app import main
from app.api import auth
from app.services.runtime_log import log_event


def test_server_owns_request_id_and_returns_stable_auth_error(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'settings', replace(main.settings, profile_db_path=str(tmp_path / 'db.sqlite3')))
    with TestClient(main.app) as client:
        response = client.get('/api/v1/profile/cards', headers={'X-Client-Request-Id': 'client-42'})
    payload = response.json()
    assert response.status_code == 401
    assert payload['error'] == {
        'code': 'AUTH_REQUIRED', 'message': '正式模式需要先登录。',
        'request_id': response.headers['X-Request-Id'], 'retryable': False,
    }
    assert payload['request_id'] != 'client-42'


def test_validation_error_does_not_echo_input(tmp_path, monkeypatch):
    settings = replace(main.settings, profile_db_path=str(tmp_path / 'db.sqlite3'))
    monkeypatch.setattr(main, 'settings', settings)
    monkeypatch.setattr(auth, 'get_settings', lambda: settings)
    secret = 'password-that-must-not-be-echoed-' * 8
    with TestClient(main.app) as client:
        response = client.post('/api/v1/auth/register', json={'email': 'bad', 'password': secret})
    assert response.status_code == 422
    assert secret not in response.text
    assert response.json()['error']['code'] == 'VALIDATION_ERROR'


def test_unhandled_error_is_masked_and_structured(tmp_path, monkeypatch):
    settings = replace(main.settings, profile_db_path=str(tmp_path / 'db.sqlite3'))
    monkeypatch.setattr(main, 'settings', settings)
    monkeypatch.setattr(auth, 'get_settings', lambda: settings)
    router = APIRouter()
    marker = 'private-database-detail'

    @router.get('/_test/runtime-error')
    def runtime_error():
        raise RuntimeError(marker)

    main.app.include_router(router, prefix='/api/v1')
    with TestClient(main.app, raise_server_exceptions=False) as client:
        registration = client.post('/api/v1/auth/register', json={
            'email': 'error-test@example.com', 'password': 'test-password-123',
        })
        response = client.get('/api/v1/_test/runtime-error', headers={
            'Authorization': 'Bearer ' + registration.json()['access_token'],
        })
    assert response.status_code == 500
    assert response.json()['error']['code'] == 'INTERNAL_ERROR'
    assert marker not in response.text


def test_runtime_log_levels_are_explicit_and_bounded():
    for level in ('debug', 'info', 'warn', 'error'):
        log_event('level_test', level=level)
    try:
        log_event('invalid_level', level='warning')
    except ValueError as exc:
        assert 'unsupported log level' in str(exc)
    else:
        raise AssertionError('unknown log levels must be rejected')
