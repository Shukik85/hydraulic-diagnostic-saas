# Generated migration for missing fields

from django.db import migrations, models
from django.db.models.functions import TruncDay


class Migration(migrations.Migration):
    dependencies = [
        ('diagnostics', '0001_initial'),
    ]

    operations = [
        # Добавляем поле created_by если его еще нет
        migrations.RunSQL(
            "ALTER TABLE diagnostics_diagnosticreport ADD COLUMN IF NOT EXISTS created_by_id UUID;",
            reverse_sql="ALTER TABLE diagnostics_diagnosticreport DROP COLUMN IF EXISTS created_by_id;"
        ),
        
        # Добавляем индекс для created_by если его еще нет
        migrations.RunSQL(
            "CREATE INDEX IF NOT EXISTS diagnostics_diagnosticreport_created_by_id ON diagnostics_diagnosticreport(created_by_id);",
            reverse_sql="DROP INDEX IF EXISTS diagnostics_diagnosticreport_created_by_id;"
        ),
        
        # Добавляем внешний ключ constraint для created_by
        migrations.RunSQL(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.table_constraints 
                    WHERE constraint_name = 'diagnostics_diagnost_created_by_id_fk'
                    AND table_name = 'diagnostics_diagnosticreport'
                ) THEN
                    ALTER TABLE diagnostics_diagnosticreport 
                    ADD CONSTRAINT diagnostics_diagnost_created_by_id_fk 
                    FOREIGN KEY (created_by_id) 
                    REFERENCES users_user(id) 
                    ON DELETE SET NULL;
                END IF;
            END
            $$;
            """,
            reverse_sql="ALTER TABLE diagnostics_diagnosticreport DROP CONSTRAINT IF EXISTS diagnostics_diagnost_created_by_id_fk;"
        ),
    ]