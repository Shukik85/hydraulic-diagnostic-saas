# Generated migration for missing fields
from django.db import migrations, models

class Migration(migrations.Migration):
    dependencies = [
        # Исправлено: users миграция идёт первой, diagnostics — после
        ('users', '0001_initial'),
        ('diagnostics', '0001_initial'),
    ]
    operations = [
        # Стандартно добавляем ForeignKey, убран raw SQL
        migrations.AddField(
            model_name='diagnosticreport',
            name='created_by',
            field=models.ForeignKey(null=True, on_delete=models.SET_NULL, to='users.user', related_name='diagnostics_reports'),
        ),
    ]
