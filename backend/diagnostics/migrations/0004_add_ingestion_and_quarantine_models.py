# Generated manually for IngestionJob and QuarantinedReading models
# Date: 2025-11-07 23:45 MSK

import django.contrib.postgres.indexes
import uuid
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):
    """
    Add IngestionJob and QuarantinedReading models for data quality management.
    
    Features:
    - Job status tracking for observability
    - Quarantine workflow for invalid data
    - Full audit trail
    - TimescaleDB optimized indexes
    """

    dependencies = [
        ('diagnostics', '0003_convert_sensordata_to_hypertable'),
        ('users', '0001_initial'),  # Adjust based on your users app migration
    ]

    operations = [
        # Create IngestionJob model
        migrations.CreateModel(
            name='IngestionJob',
            fields=[
                ('id', models.UUIDField(
                    default=uuid.uuid4,
                    editable=False,
                    help_text='Unique job identifier',
                    primary_key=True,
                    serialize=False
                )),
                ('status', models.CharField(
                    choices=[
                        ('queued', 'Queued'),
                        ('processing', 'Processing'),
                        ('completed', 'Completed'),
                        ('failed', 'Failed')
                    ],
                    db_index=True,
                    default='queued',
                    help_text='Current job status',
                    max_length=20
                )),
                ('created_at', models.DateTimeField(
                    db_index=True,
                    default=django.utils.timezone.now,
                    help_text='When job was created'
                )),
                ('started_at', models.DateTimeField(
                    blank=True,
                    help_text='When processing started',
                    null=True
                )),
                ('completed_at', models.DateTimeField(
                    blank=True,
                    help_text='When job finished (success or failure)',
                    null=True
                )),
                ('total_readings', models.PositiveIntegerField(
                    default=0,
                    help_text='Total number of readings in batch'
                )),
                ('inserted_readings', models.PositiveIntegerField(
                    default=0,
                    help_text='Successfully inserted readings'
                )),
                ('quarantined_readings', models.PositiveIntegerField(
                    default=0,
                    help_text='Readings quarantined due to validation errors'
                )),
                ('system_id', models.UUIDField(
                    db_index=True,
                    help_text='Target hydraulic system UUID'
                )),
                ('error_message', models.TextField(
                    blank=True,
                    default='',
                    help_text='Error message if job failed'
                )),
                ('processing_time_ms', models.PositiveIntegerField(
                    blank=True,
                    help_text='Total processing time in milliseconds',
                    null=True
                )),
                ('celery_task_id', models.CharField(
                    blank=True,
                    db_index=True,
                    default='',
                    help_text='Celery task ID for tracking',
                    max_length=255
                )),
                ('created_by', models.ForeignKey(
                    blank=True,
                    help_text='User who initiated ingestion',
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='ingestion_jobs',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'db_table': 'diagnostics_ingestion_job',
                'ordering': ['-created_at'],
                'verbose_name': 'Ingestion Job',
                'verbose_name_plural': 'Ingestion Jobs',
            },
        ),
        
        # Create QuarantinedReading model
        migrations.CreateModel(
            name='QuarantinedReading',
            fields=[
                ('id', models.UUIDField(
                    default=uuid.uuid4,
                    editable=False,
                    primary_key=True,
                    serialize=False
                )),
                ('job_id', models.UUIDField(
                    db_index=True,
                    help_text='UUID of the ingestion job that quarantined this reading'
                )),
                ('sensor_id', models.UUIDField(
                    help_text='Original sensor ID from reading'
                )),
                ('timestamp', models.DateTimeField(
                    db_index=True,
                    help_text='Original timestamp from reading'
                )),
                ('value', models.FloatField(
                    help_text='Original sensor value'
                )),
                ('unit', models.CharField(
                    help_text='Original unit of measurement',
                    max_length=32
                )),
                ('quality', models.IntegerField(
                    default=0,
                    help_text='Original quality score (0-100)'
                )),
                ('system_id', models.UUIDField(
                    blank=True,
                    db_index=True,
                    help_text='System ID if available',
                    null=True
                )),
                ('reason', models.CharField(
                    choices=[
                        ('out_of_range', 'Value out of valid range'),
                        ('invalid_timestamp', 'Invalid or future timestamp'),
                        ('duplicate', 'Duplicate reading detected'),
                        ('parse_error', 'Failed to parse reading'),
                        ('system_not_found', 'System ID not found'),
                        ('invalid_unit', 'Invalid measurement unit'),
                        ('other', 'Other validation error')
                    ],
                    db_index=True,
                    help_text='Reason for quarantine',
                    max_length=32
                )),
                ('reason_details', models.TextField(
                    blank=True,
                    default='',
                    help_text='Detailed explanation of validation failure'
                )),
                ('review_status', models.CharField(
                    choices=[
                        ('pending', 'Pending Review'),
                        ('approved', 'Approved - Will Retry'),
                        ('rejected', 'Rejected - Discard'),
                        ('fixed', 'Fixed and Reprocessed')
                    ],
                    db_index=True,
                    default='pending',
                    max_length=20
                )),
                ('reviewed_at', models.DateTimeField(
                    blank=True,
                    help_text='When review was completed',
                    null=True
                )),
                ('review_notes', models.TextField(
                    blank=True,
                    default='',
                    help_text='Notes from manual review'
                )),
                ('quarantined_at', models.DateTimeField(
                    db_index=True,
                    default=django.utils.timezone.now
                )),
                ('reviewed_by', models.ForeignKey(
                    blank=True,
                    help_text='User who reviewed this reading',
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='reviewed_quarantined_readings',
                    to=settings.AUTH_USER_MODEL
                )),
            ],
            options={
                'db_table': 'diagnostics_quarantined_reading',
                'ordering': ['-quarantined_at'],
                'verbose_name': 'Quarantined Reading',
                'verbose_name_plural': 'Quarantined Readings',
            },
        ),
        
        # Add indexes for IngestionJob
        migrations.AddIndex(
            model_name='ingestionjob',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['status', 'created_at'],
                name='idx_ij_status_created'
            ),
        ),
        migrations.AddIndex(
            model_name='ingestionjob',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['system_id', 'created_at'],
                name='idx_ij_system_created'
            ),
        ),
        migrations.AddIndex(
            model_name='ingestionjob',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['celery_task_id'],
                name='idx_ij_celery_task'
            ),
        ),
        
        # Add indexes for QuarantinedReading
        migrations.AddIndex(
            model_name='quarantinedreading',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['job_id', 'quarantined_at'],
                name='idx_qr_job_time'
            ),
        ),
        migrations.AddIndex(
            model_name='quarantinedreading',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['reason', 'quarantined_at'],
                name='idx_qr_reason_time'
            ),
        ),
        migrations.AddIndex(
            model_name='quarantinedreading',
            index=django.contrib.postgres.indexes.BTreeIndex(
                fields=['review_status', 'quarantined_at'],
                name='idx_qr_status_time'
            ),
        ),
        migrations.AddIndex(
            model_name='quarantinedreading',
            index=django.contrib.postgres.indexes.BrinIndex(
                autosummarize=True,
                fields=['quarantined_at'],
                name='brin_qr_time'
            ),
        ),
    ]
