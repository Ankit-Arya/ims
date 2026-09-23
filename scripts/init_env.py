#!/usr/bin/env python3
import argparse
import secrets
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a secure local .env from .env.example")
    parser.add_argument("--force", action="store_true", help="overwrite an existing .env")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    src = root / ".env.example"
    dst = root / ".env"
    if dst.exists() and not args.force:
        raise SystemExit(".env already exists; use --force only if replacement is intentional")
    admin_password = secrets.token_urlsafe(18)
    database_password = secrets.token_urlsafe(24)
    text = src.read_text(encoding="utf-8")
    text = text.replace("replace-with-at-least-32-random-characters", secrets.token_urlsafe(48))
    text = text.replace("replace-with-another-long-random-value", secrets.token_urlsafe(48))
    text = text.replace("change-this-before-running", admin_password)
    # Support both the current production-safe template placeholder and the legacy local
    # placeholder so a fresh source archive always produces one internally consistent DB
    # password/URL pair.
    for placeholder in ("replace-with-strong-database-password", "knowledge-local-only"):
        text = text.replace(f"POSTGRES_PASSWORD={placeholder}", f"POSTGRES_PASSWORD={database_password}")
        text = text.replace(
            f"DATABASE_URL=postgresql+psycopg://knowledge:{placeholder}@postgres:5432/knowledge",
            f"DATABASE_URL=postgresql+psycopg://knowledge:{database_password}@postgres:5432/knowledge",
        )
    dst.write_text(text, encoding="utf-8")
    print(f"Created {dst}")
    print(f"Bootstrap admin username: admin")
    print(f"Bootstrap admin password: {admin_password}")
    print("Set OPENAI_API_KEY in .env before starting the application.")


if __name__ == "__main__":
    main()
