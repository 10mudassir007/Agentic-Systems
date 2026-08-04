from sqlalchemy import inspect as sa_inspect


def introspect_schema(engine) -> str:
    inspector = sa_inspect(engine)
    lines = []
    for table_name in inspector.get_table_names():
        lines.append(f"Table: {table_name}")
        columns = inspector.get_columns(table_name)
        for col in columns:
            col_type = str(col["type"])
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            lines.append(f"  - {col['name']} ({col_type}, {nullable})")
        lines.append("")
    return "\n".join(lines)
