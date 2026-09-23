from pathlib import Path

from scripts.upgrade_env_050 import NEVER_TOUCH, upgrade


def test_env_upgrade_preserves_secrets_and_adds_release_settings(tmp_path: Path):
    env = tmp_path / '.env'
    env.write_text(
        '\n'.join(
            [
                'APP_ENV=development',
                'APP_SECRET_KEY=secret-app-value-that-must-stay',
                'POSTGRES_PASSWORD=db-secret',
                'DATABASE_URL=postgresql+psycopg://u:db-secret@postgres:5432/d',
                'INTERNAL_SERVICE_TOKEN=internal-secret',
                'OPENAI_API_KEY=openai-secret',
                'BOOTSTRAP_ADMIN_PASSWORD=admin-secret',
                'CUSTOM_KEEP_ME=yes',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    backup, _changed, _added = upgrade(env)
    assert backup is not None and backup.is_file()
    current = env.read_text(encoding='utf-8')
    before = backup.read_text(encoding='utf-8')
    for key in NEVER_TOUCH:
        original_line = next(line for line in before.splitlines() if line.startswith(f'{key}='))
        assert original_line in current
    assert 'APP_ENV=production' in current
    assert 'QUERY_INFERENCE_URL=http://inference-query:8090' in current
    assert 'INGESTION_INFERENCE_URL=http://inference-ingest:8090' in current
    assert 'INGEST_WORKER_CONCURRENCY=1' in current
    assert 'HELPFUL_CONTEXT_MODE=rich' in current
    assert 'HELPFUL_CONTEXT_MAX_SECTIONS=6' in current
    assert 'CUSTOM_KEEP_ME=yes' in current
    assert (env.stat().st_mode & 0o777) == 0o600
