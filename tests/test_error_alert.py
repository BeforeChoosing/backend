from app.services import error_alert


class ImmediateExecutor:
    def submit(self, function, *args):
        function(*args)


def test_alert_is_redacted_and_rate_limited(monkeypatch):
    sent = []
    monkeypatch.setenv('ERROR_ALERT_ENABLED', 'true')
    monkeypatch.setenv('RESEND_API_KEY', 'test-key')
    monkeypatch.setenv('ERROR_ALERT_TO', 'owner@example.com')
    monkeypatch.setenv('ERROR_ALERT_FROM', 'alerts@example.com')
    monkeypatch.setenv('ERROR_ALERT_COOLDOWN_SECONDS', '60')
    monkeypatch.setattr(error_alert, '_executor', ImmediateExecutor())
    monkeypatch.setattr(error_alert, '_send', lambda *args: sent.append(args))
    error_alert._reset_for_tests()
    fields = {'request_id': 'req-1', 'route': '/profile/cards', 'error_code': 'INTERNAL_ERROR',
              'exception_type': 'RuntimeError', 'password': 'secret', 'answer': 'private answer'}
    error_alert.queue_error_alert('request_failed', fields)
    error_alert.queue_error_alert('request_failed', fields)
    assert len(sent) == 1
    safe = sent[0][-1]
    assert safe['request_id'] == 'req-1'
    assert 'password' not in safe and 'answer' not in safe


def test_alert_requires_complete_configuration(monkeypatch):
    monkeypatch.delenv('ERROR_ALERT_ENABLED', raising=False)
    monkeypatch.setattr(error_alert, '_executor', object())
    error_alert.queue_error_alert('request_failed', {'request_id': 'req'})
