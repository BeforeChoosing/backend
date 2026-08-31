"""Run a real local HTTP isolation smoke test without touching user data or Qwen.

Usage: python scripts/check_user_isolation.py
Requires the backend test dependencies (httpx). Starts its own single-worker
Uvicorn on a free loopback port and removes only its temporary test database.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx


def check() -> dict:
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='before-choosing-isolation-') as directory:
        env = {**os.environ, 'PROFILE_DB_PATH': str(Path(directory) / 'profile.db'),
               'DASHSCOPE_API_KEY': ''}
        with open(Path(directory) / 'server.log', 'w+') as log:
            process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1',
                '--port', str(port), '--workers', '1', '--no-access-log',
            ], cwd=root, env=env, stdout=log, stderr=log)
            started = time.perf_counter()
            try:
                with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=10) as client:
                    for _ in range(100):
                        try:
                            if client.get('/api/v1/health').status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if process.poll() is not None:
                            raise RuntimeError('测试服务启动失败')
                        time.sleep(.05)
                    else:
                        raise RuntimeError('等待测试服务超时')

                    for headers in [{}, {'X-App-Mode': 'demo'}, {'X-App-Mode': 'use'},
                                    {'X-App-Mode': 'unknown'}, {'Authorization': 'Bearer invalid'}]:
                        assert client.get('/api/v1/profile/cards', headers=headers).status_code == 401

                    def user(i):
                        response = client.post('/api/v1/auth/register', json={
                            'email': f'user-{i}@example.com', 'password': 'isolation-test-123',
                        })
                        assert response.status_code == 201
                        headers = {'Authorization': 'Bearer ' + response.json()['access_token']}
                        payload = {'id': 'same-id', 'title': f'用户{i}的卡', 'category': '洞察分析',
                                   'description': '测试描述', 'detail': '测试详情', 'evidence_quote': '测试证据',
                                   'next_verification': '测试验证', 'match_reason': '测试依据', 'workplace_application': '测试应用'}
                        for _ in range(5):
                            assert client.post('/api/v1/profile/cards/confirm', headers=headers,
                                               json={'cards': [payload]}).status_code == 200
                            own = client.get('/api/v1/profile/cards', headers=headers)
                            assert own.json()['cards'][0]['title'] == f'用户{i}的卡'
                            assert len(own.json()['cards']) == 1
                        response = client.post('/api/v1/trial/workbench/sessions', headers=headers,
                                               json={'task_id': 'A-02'})
                        assert response.status_code == 200
                        return headers, response.json()['id']

                    with ThreadPoolExecutor(max_workers=5) as pool:
                        users = list(pool.map(user, range(5)))
                    denied = 0
                    for i, (headers, own_id) in enumerate(users):
                        for j, (_, other_id) in enumerate(users):
                            if i == j:
                                continue
                            path = '/api/v1/trial/workbench/sessions/' + other_id
                            assert client.get(path, headers=headers).status_code == 404
                            assert client.post(path + '/event', headers=headers).status_code == 404
                            denied += 2
                        # Forged client mode does not disable audit for real operations.
                        assert client.post('/api/v1/audit/events', headers={**headers, 'X-App-Mode': 'demo'},
                                           json={'action': f'owner-{i}'}).json()['recorded']
                        events = client.get('/api/v1/audit/events', headers=headers).json()
                        assert any(e['action'] == f'owner-{i}' for e in events)
                        assert not any(e['action'] in {f'owner-{j}' for j in range(5) if j != i} for e in events)
                        assert client.post('/api/v1/auth/logout', headers=headers).status_code == 200
                        assert client.get('/api/v1/profile/cards', headers=headers).status_code == 401
                    assert client.get('/api/v1/trial/catalog').status_code == 200
                    return {'passed': True, 'concurrent_users': 5, 'same_id_card_roundtrips': 25,
                            'cross_user_operations_denied': denied, 'anonymous_checks': 5,
                            'logout_checks': 5, 'model_calls': 0,
                            'elapsed_seconds': round(time.perf_counter() - started, 3)}
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    print(json.dumps(check(), ensure_ascii=False, indent=2))
