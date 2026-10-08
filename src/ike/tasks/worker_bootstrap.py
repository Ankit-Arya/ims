"""Start the ingestion worker with CPU- and memory-aware prefork concurrency."""

import logging
import os

from ike.core.config import get_settings

logger = logging.getLogger(__name__)


def _memory_gb() -> float:
    # Respect a Docker/cgroup memory ceiling when one is configured.
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw and raw != "max":
            return int(raw) / 1024 / 1024 / 1024
    except (OSError, ValueError):
        pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return kib / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 4.0


def calculate_concurrency() -> tuple[int, dict[str, float | int]]:
    settings = get_settings()
    override = settings.ingest_worker_concurrency.strip().lower()
    cpus = max(1, os.cpu_count() or 1)
    memory_gb = max(1.0, _memory_gb())
    target_percent = max(10, min(95, settings.ingest_cpu_target_percent))
    target_threads = max(1, int(cpus * target_percent / 100))
    per_document_threads = max(1, settings.docling_num_threads_per_document)
    cpu_limit = max(1, target_threads // per_document_threads)
    usable_memory_gb = max(1.0, memory_gb - max(0.0, settings.ingest_memory_reserve_gb))
    memory_limit = max(1, int(usable_memory_gb // max(0.5, settings.ingest_memory_gb_per_process)))
    automatic = max(1, min(settings.ingest_max_concurrency, cpu_limit, memory_limit))

    if override != "auto":
        try:
            chosen = max(1, int(override))
        except ValueError as exc:
            raise SystemExit(
                "INGEST_WORKER_CONCURRENCY must be 'auto' or a positive integer"
            ) from exc
    else:
        chosen = automatic

    return chosen, {
        "logical_cpus": cpus,
        "memory_gb": round(memory_gb, 1),
        "usable_memory_gb": round(usable_memory_gb, 1),
        "target_percent": target_percent,
        "target_threads": target_threads,
        "threads_per_document": per_document_threads,
        "cpu_limit": cpu_limit,
        "memory_limit": memory_limit,
        "max_concurrency": settings.ingest_max_concurrency,
    }


def main() -> None:
    concurrency, details = calculate_concurrency()
    print(
        "IKE ingestion worker: "
        f"concurrency={concurrency}, CPUs={details['logical_cpus']}, "
        f"target={details['target_percent']}%, memory={details['memory_gb']}GiB, "
        f"Docling threads/document={details['threads_per_document']}"
    )
    command = [
        "celery",
        "-A",
        "ike.tasks.celery_app:celery_app",
        "worker",
        "-I",
        "ike.tasks.ingestion",
        "--loglevel=INFO",
        "-Q",
        "ingestion",
        "--pool=prefork",
        f"--concurrency={concurrency}",
        "--prefetch-multiplier=1",
        f"--max-tasks-per-child={max(1, int(os.environ.get('INGEST_MAX_TASKS_PER_CHILD', '20')))}",
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
