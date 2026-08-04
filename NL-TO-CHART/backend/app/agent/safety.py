import sqlglot
from sqlglot import exp, parse
from app.db.connection import DBState


def is_safe_query(sql: str) -> tuple[bool, str]:
    parsed = parse(sql, error_level=exp.ErrorLevel.RAISE)
    if not parsed or len(parsed) != 1:
        return False, "Query must be exactly one statement"

    statement = parsed[0]
    if not isinstance(statement, exp.Select):
        return False, "Only SELECT statements are allowed"

    has_limit = any(isinstance(node, exp.Limit) for node in statement.walk())
    if not has_limit:
        statement = statement.limit(1000)

    dialect = DBState.sqlglot_dialect
    rewritten = statement.sql(dialect=dialect)
    return True, rewritten
