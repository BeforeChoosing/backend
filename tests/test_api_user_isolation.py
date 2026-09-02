from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import audit, auth, career, profile, trial
from app.schemas.profile import CardProposal, ProfileExplorationResponse
from app.services.profile_store import ProfileStore
from app.services.trial_store import TrialStore
from app.services.dynamic_trial_store import DynamicTrialStore


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = replace(main.settings, profile_db_path=str(tmp_path / 'profile.db'))
    monkeypatch.setattr(main, 'settings', settings)
    for module in (audit, auth, career, profile, trial):
        monkeypatch.setattr(module, 'get_settings', lambda: settings)
    with TestClient(main.app) as client:
        yield client, settings


def account(client, name):
    response = client.post('/api/v1/auth/register', json={
        'email': f'{name}@example.com', 'password': 'isolation-test-123',
    })
    assert response.status_code == 201
    return {'Authorization': 'Bearer ' + response.json()['access_token']}


def card(card_id='same-id', title='用户研究'):
    return {'id': card_id, 'title': title, 'category': '洞察分析',
            'description': '整理用户访谈', 'detail': '根据访谈调整设计',
            'evidence_quote': '我访谈了六位同学', 'next_verification': '验证改进',
            'match_reason': '具有访谈证据', 'workplace_application': '需求分析'}


@pytest.mark.parametrize('mode', [None, 'demo', 'unknown', '', 'use', 'USE', 'garbage'])
@pytest.mark.parametrize('method,path,body', [
    ('GET', '/profile/cards', None), ('GET', '/profile/overview', None),
    ('GET', '/profile/conversations', None), ('GET', '/profile/conversation-snapshots', None),
    ('GET', '/audit/events', None),
    ('GET', '/audit/usage', None), ('POST', '/profile/cards/confirm', {'cards': [card()]}),
    ('POST', '/profile/exploration/messages', {'experience_text': '我负责校园项目访谈，整理六位同学的反馈并据此改进方案。'}),
    ('POST', '/trial/sessions', {'task_id': 'A-02'}),
    ('POST', '/trial/workbench/sessions', {'task_id': 'A-02'}),
])
def test_private_api_requires_auth_regardless_of_mode(api, mode, method, path, body):
    client, _ = api
    headers = {} if mode is None else {'X-App-Mode': mode}
    response = client.request(method, '/api/v1' + path, headers=headers,
                              **({'json': body} if body else {}))
    assert response.status_code == 401


def test_cards_versions_and_same_ids_are_account_scoped(api):
    c, _ = api
    a, b = account(c, 'alice'), account(c, 'bob')
    assert c.post('/api/v1/profile/cards/confirm', headers=a, json={'cards': [card()]}).status_code == 200
    assert c.get('/api/v1/profile/overview', headers=b).json()['cards'] == []
    assert c.get('/api/v1/profile/overview', headers=b).json()['version'] == 0
    for method, body in [('PATCH', {'title': '恶意覆盖'}), ('DELETE', None)]:
        assert c.request(method, '/api/v1/profile/cards/same-id', headers=b,
                         **({'json': body} if body else {})).status_code == 404
    assert c.post('/api/v1/profile/cards/confirm', headers=b,
                  json={'cards': [card(title='Bob 的独立卡')]}).status_code == 200
    assert c.get('/api/v1/profile/cards', headers=a).json()['cards'][0]['title'] == '用户研究'
    assert c.delete('/api/v1/profile/cards/same-id', headers=b).status_code == 200
    assert len(c.get('/api/v1/profile/cards', headers=a).json()['cards']) == 1
    assert c.get('/api/v1/profile/cards', headers=a).json()['version'] == 1


@pytest.mark.parametrize('prefix', ['/trial/sessions', '/trial/workbench/sessions'])
def test_all_session_operations_require_ownership(api, prefix):
    c, _ = api
    a, b = account(c, 'alice'), account(c, 'bob')
    created = c.post('/api/v1' + prefix, headers=a, json={'task_id': 'A-02'})
    assert created.status_code == 200
    path = '/api/v1' + prefix + '/' + created.json()['id']
    operations = [('GET', '', None), ('PUT', '/answer', {'answer': {}}),
                  ('POST', '/event', None), ('POST', '/submit', None)]
    if 'workbench' in prefix:
        operations.append(('POST', '/coach', {'level': 1}))
    for method, suffix, body in operations:
        response = c.request(method, path + suffix, headers=b, **({'json': body} if body else {}))
        assert response.status_code == 404, (method, suffix, response.text)
    own = c.get(path, headers=a)
    assert own.status_code == 200 and not own.json()['event_revealed']
    assert c.post(path + '/event', headers=a).status_code == 200


def test_old_unowned_data_is_preserved_but_not_assigned_to_new_accounts(api):
    c, settings = api
    legacy = ProfileStore(settings.profile_db_path)
    legacy.confirm_cards([CardProposal.model_validate(card('legacy'))])
    old_sessions = [('/trial/sessions/', TrialStore(settings.profile_db_path).create_session('A-02').id),
                    ('/trial/workbench/sessions/', DynamicTrialStore(settings.profile_db_path).create_session('A-02').id)]
    a = account(c, 'alice')
    assert c.get('/api/v1/profile/cards', headers=a).json()['cards'] == []
    for prefix, sid in old_sessions:
        assert c.get('/api/v1' + prefix + sid, headers=a).status_code == 404
    assert legacy.get_profile().cards[0].id == 'legacy'


def test_chat_cache_and_logs_are_account_scoped_even_with_demo_header(api, monkeypatch):
    c, _ = api
    a, b = account(c, 'alice'), account(c, 'bob')
    calls = []
    class Agent:
        async def explore(self, request, trace_id):
            calls.append(trace_id)
            return ProfileExplorationResponse(trace_id=trace_id, reply='请补充你的决策依据。',
                focus_dimension='decision', evidence_found=['完成访谈'], evidence_gap='缺少取舍依据',
                potential_hypotheses=[])
    monkeypatch.setattr(profile, '_profile_agent', Agent)
    body = {'experience_text': '我负责校园项目访谈，整理六位同学的反馈并据此改进方案。'}
    a = {**a, 'X-App-Mode': 'demo'}
    first = c.post('/api/v1/profile/exploration/messages', headers=a, json=body)
    again = c.post('/api/v1/profile/exploration/messages', headers=a, json=body)
    second = c.post('/api/v1/profile/exploration/messages', headers=b, json=body)
    assert first.status_code == again.status_code == second.status_code == 200
    assert len(calls) == 2
    assert first.json()['trace_id'] == again.json()['trace_id'] != second.json()['trace_id']
    history = c.get('/api/v1/profile/conversations', headers=b).json()
    assert len(history) == 1 and history[0]['trace_id'] == second.json()['trace_id']
    assert c.post('/api/v1/audit/events', headers=a, json={'action': 'alice-only'}).json()['recorded']
    own_events = c.get('/api/v1/audit/events', headers=a).json()
    other_events = c.get('/api/v1/audit/events', headers=b).json()
    assert any(e['action'] == 'alice-only' for e in own_events)
    assert not any(e['action'] == 'alice-only' for e in other_events)


def test_conversation_snapshots_sync_and_remain_account_scoped(api):
    c, _ = api
    alice, bob = account(c, 'snapshot-alice'), account(c, 'snapshot-bob')
    payload = {
        'title': '校园项目访谈',
        'messages': [
            {'id': 'message-1', 'role': 'user', 'content': '我访谈了六位同学。'},
            {'id': 'message-2', 'role': 'ai', 'content': '你如何选择访谈对象？'},
        ],
        'evidence': '我访谈了六位同学。',
        'materials': [],
        'target_career_state': 'has_target',
        'target_role': 'AI 产品经理',
        'model_tier': 'balanced',
    }
    path = '/api/v1/profile/conversation-snapshots/conversation-1'
    created = c.put(path, headers=alice, json=payload)
    assert created.status_code == 200
    assert created.json()['id'] == 'conversation-1'
    assert c.get('/api/v1/profile/conversation-snapshots', headers=bob).json() == []

    payload['title'] = '更新后的标题'
    updated = c.put(path, headers=alice, json=payload)
    assert updated.status_code == 200
    history = c.get('/api/v1/profile/conversation-snapshots', headers=alice).json()
    assert len(history) == 1
    assert history[0]['title'] == '更新后的标题'

    assert c.delete(path, headers=bob).status_code == 404
    assert c.delete(path, headers=alice).json() == {'deleted': True}
    assert c.get('/api/v1/profile/conversation-snapshots', headers=alice).json() == []


def test_public_catalog_preflight_and_revoked_token(api):
    c, _ = api
    assert c.get('/api/v1/health').status_code == 200
    assert c.get('/api/v1/trial/catalog', headers={'X-App-Mode': 'demo'}).status_code == 200
    assert c.options('/api/v1/profile/cards', headers={
        'Origin': 'http://localhost:3000', 'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'authorization,x-app-mode'}).status_code == 200
    a = account(c, 'alice')
    lower = {'Authorization': a['Authorization'].replace('Bearer ', 'bearer ')}
    assert c.post('/api/v1/auth/logout', headers=lower).status_code == 200
    assert c.get('/api/v1/profile/cards', headers=a).status_code == 401
    assert c.get('/api/v1/profile/cards/', headers={'Authorization': 'Bearer invalid'}).status_code == 401
    denied = c.get('/api/v1/profile/cards', headers={'Origin': 'http://localhost:3000'})
    assert denied.status_code == 401
    assert denied.headers['access-control-allow-origin'] == 'http://localhost:3000'
    assert denied.headers['cache-control'] == 'no-store'


def test_five_users_concurrent_same_card_ids(api):
    c, _ = api
    headers = [account(c, f'user{i}') for i in range(5)]
    def operate(i):
        h = headers[i]
        for _ in range(3):
            assert c.post('/api/v1/profile/cards/confirm', headers=h,
                          json={'cards': [card(title=f'用户{i}的卡')]}).status_code == 200
            own = c.get('/api/v1/profile/cards', headers=h).json()['cards']
            assert [x['title'] for x in own] == [f'用户{i}的卡']
        return True
    with ThreadPoolExecutor(max_workers=5) as pool:
        assert all(pool.map(operate, range(5)))


def test_selected_cards_cannot_reference_another_account(api):
    c, _ = api
    a, b = account(c, 'alice'), account(c, 'bob')
    c.post('/api/v1/profile/cards/confirm', headers=a, json={'cards': [card('alice-only')]})
    for path in ['/career/recommendations', '/trial/recommendations']:
        response = c.post('/api/v1' + path, headers=b,
                          json={'selected_card_ids': ['alice-only'], 'target_role': 'AI 产品经理'})
        assert response.status_code == 422


def test_submitted_evidence_only_updates_its_owner(api, monkeypatch):
    # Reuse existing model fixtures, but NOT their single-store monkeypatches:
    # this covers production storage routing through the full submit flow.
    from test_trial_api import FakeTrialAgent, FakeReflectionAgent, _complete_dynamic_answer
    c, _ = api
    a, b = account(c, 'alice'), account(c, 'bob')
    monkeypatch.setattr(trial, '_trial_agent', FakeTrialAgent)
    monkeypatch.setattr(trial, '_reflection_agent', FakeReflectionAgent)
    c.post('/api/v1/profile/cards/confirm', headers=a,
           json={'cards': [card('trial-card-1', '拆解并验证产品方案')]})
    created = c.post('/api/v1/trial/workbench/sessions', headers=a, json={'task_id': 'F-01'})
    path = '/api/v1/trial/workbench/sessions/' + created.json()['id']
    assert c.put(path + '/answer', headers=a, json={'answer': _complete_dynamic_answer()}).status_code == 200
    assert c.post(path + '/event', headers=a).status_code == 200
    submitted = c.post(path + '/submit', headers=a)
    assert submitted.status_code == 200, submitted.text
    own = c.get('/api/v1/profile/overview', headers=a).json()
    other = c.get('/api/v1/profile/overview', headers=b).json()
    assert own['completed_task_ids'] == ['F-01'] and len(own['evidence']) == 1
    assert other['completed_task_ids'] == [] and other['evidence'] == [] and other['version'] == 0
    assert c.post(path + '/submit', headers=b).status_code == 404


def test_private_storage_and_audit_fail_closed_without_context(tmp_path):
    from fastapi import HTTPException
    from app.services.user_data import user_data_path
    for call in [lambda: user_data_path(tmp_path / 'profile.db'), audit.get_audit_usage,
                 lambda: audit.list_audit_events(limit=10)]:
        with pytest.raises(HTTPException) as error:
            call()
        assert error.value.status_code == 401
