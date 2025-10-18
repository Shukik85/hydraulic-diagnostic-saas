# Generated migration for adding concurrent indexes to optimize queries
from django.db import migrations

SQL_STATEMENTS = [
    # Documents
    (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_language_format "
        "ON rag_assistant_document(language, format)"
    ),
    (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_created_at "
        "ON rag_assistant_document(created_at)"
    ),
    # Query logs
    (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ragquerylog_timestamp "
        "ON rag_assistant_ragquerylog(timestamp)"
    ),
    (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ragquerylog_system_ts "
        "ON rag_assistant_ragquerylog(system_id, timestamp)"
    ),
]

REVERSE_SQL = [
    "DROP INDEX IF EXISTS idx_document_language_format",
    "DROP INDEX IF EXISTS idx_document_created_at",
    "DROP INDEX IF EXISTS idx_ragquerylog_timestamp",
    "DROP INDEX IF EXISTS idx_ragquerylog_system_ts",
]


def create_indexes(apps, schema_editor):
    # Run only on PostgreSQL
    if schema_editor.connection.vendor != "postgresql":
        return
    for stmt in SQL_STATEMENTS:
        schema_editor.execute(stmt)


def drop_indexes(apps, schema_editor):
    # Run only on PostgreSQL
    if schema_editor.connection.vendor != "postgresql":
        return
    for stmt in REVERSE_SQL:
        schema_editor.execute(stmt)


class Migration(migrations.Migration):
    dependencies = [
        ("rag_assistant", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(code=create_indexes, reverse_code=drop_indexes, elidable=True),
    ]
