import os

import pytest

os.environ.setdefault("CHEMRAG_LLM_PROVIDER", "none")
os.environ.setdefault("CHEMRAG_RUNS_DIR", "runs/test")

from chemrag.settings import get_settings  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    s = get_settings()
    if not s.db_path.exists():
        from chemrag.etl.build_db import build_database

        build_database(s.csv_path, s.db_path, s.groups_path, log=lambda *a: None)
    return s


@pytest.fixture(scope="session")
def engine(settings):
    from chemrag.query.tools import QueryEngine

    return QueryEngine(settings.db_path)


@pytest.fixture(scope="session")
def resolver(engine, settings):
    from chemrag.retrieval.resolver import EntityResolver

    return EntityResolver(engine, settings, use_vectors=False)


@pytest.fixture(scope="session")
def orch(settings):
    from chemrag.orchestrator import Orchestrator

    return Orchestrator(no_llm=True, use_vectors=False)
