from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_local_override_and_env_helper_exist():
    compose = (ROOT / 'docker-compose.local.yml').read_text(encoding='utf-8')
    helper = (ROOT / 'scripts/make_local_env.py').read_text(encoding='utf-8')
    assert '.env.local' in compose
    assert 'SESSION_COOKIE_SECURE' in helper
    assert 'APP_ENV' in helper
