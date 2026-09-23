from __future__ import annotations

from pathlib import Path

SOURCE = Path('.env')
TARGET = Path('.env.local')
OVERRIDES = {
    'APP_ENV': 'development',
    'SESSION_COOKIE_SECURE': 'false',
    'ONLINE_CPUSET': '0-2',
    'BACKGROUND_CPUSET': '3',
    'QUERY_ML_NUM_THREADS': '3',
    'INGEST_ML_NUM_THREADS': '1',
    'QUERY_MAX_ACTIVE_PER_API': '2',
    'QUERY_MAX_WAITING_PER_API': '16',
}


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit('Run from the project root after creating .env')
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    seen = set()
    output = []
    for line in lines:
        if '=' in line and not line.lstrip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in OVERRIDES:
                output.append(f'{key}={OVERRIDES[key]}')
                seen.add(key)
                continue
        output.append(line)
    for key, value in OVERRIDES.items():
        if key not in seen:
            output.append(f'{key}={value}')
    TARGET.write_text('\n'.join(output) + '\n', encoding='utf-8')
    TARGET.chmod(0o600)
    print('Created .env.local. Secrets were copied from .env; do not commit it.')
    print('Default local CPU layout assumes at least 4 logical CPUs (0-3).')


if __name__ == '__main__':
    main()
