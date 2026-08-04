from sqlalchemy import create_engine, text
from app.db.introspect import introspect_schema

SQLGLOT_DIALECT_MAP = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "mssql": "mssql",
    "sqlite": "sqlite",
}


class DBState:
    engine = None
    schema_text: str | None = None
    sqlglot_dialect: str = "postgres"


DRIVER_MAP = {
    "postgresql": "postgresql",
    "mysql": "mysql+pymysql",
    "mssql": "mssql",
    "sqlite": "sqlite",
}


def connect(host: str, port: int, user: str, password: str, dbname: str, dialect: str = "postgresql"):
    driver = DRIVER_MAP.get(dialect, dialect)
    url = f"{driver}://{user}:{password}@{host}:{port}/{dbname}"
    engine = create_engine(url, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    DBState.engine = engine
    DBState.schema_text = introspect_schema(engine)
    DBState.sqlglot_dialect = SQLGLOT_DIALECT_MAP.get(dialect, "postgres")
    return engine
