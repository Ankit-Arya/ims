.PHONY: up down logs ps test lint migrate shell reset

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f --tail=200 api worker inference valkey

ps:
	docker compose ps

migrate:
	docker compose run --rm migrate

test:
	python -m pytest

lint:
	ruff check src inference_service tests scripts

shell:
	docker compose exec api /bin/sh

reset:
	docker compose down -v
