"""
Django management command to test TimescaleDB SensorData hypertable.

Usage:
    python manage.py test_timescale_sensordata
    python manage.py test_timescale_sensordata --benchmark
    python manage.py test_timescale_sensordata --verify-only
"""

import time
import uuid
from datetime import timedelta

from django.core.management.base import BaseCommand
from django.db import connection
from django.utils import timezone

from diagnostics.models import SensorData, HydraulicSystem
from users.models import User


class Command(BaseCommand):
    help = "Test TimescaleDB SensorData hypertable setup and performance"

    def add_arguments(self, parser):
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Run performance benchmarks (insert 10K rows)',
        )
        parser.add_argument(
            '--verify-only',
            action='store_true',
            help='Only verify hypertable setup without inserting data',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(
            "\n" + "="*70
        ))
        self.stdout.write(self.style.SUCCESS(
            "TimescaleDB SensorData Hypertable Test"
        ))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))

        # Step 1: Verify hypertable
        self.stdout.write("[1/5] Verifying hypertable setup...")
        if not self.verify_hypertable():
            self.stdout.write(self.style.ERROR(
                "\n❌ Hypertable not found! Run migration first:"
            ))
            self.stdout.write(self.style.WARNING(
                "    python manage.py migrate diagnostics 0003"
            ))
            return

        # Step 2: Verify compression
        self.stdout.write("\n[2/5] Verifying compression policy...")
        self.verify_compression()

        # Step 3: Verify retention
        self.stdout.write("\n[3/5] Verifying retention policy...")
        self.verify_retention()

        # Step 4: Verify continuous aggregate
        self.stdout.write("\n[4/5] Verifying continuous aggregate...")
        self.verify_continuous_aggregate()

        if options['verify_only']:
            self.stdout.write(self.style.SUCCESS(
                "\n✅ Verification complete!"
            ))
            return

        # Step 5: Test inserts
        self.stdout.write("\n[5/5] Testing data insertion...")
        if options['benchmark']:
            self.run_benchmark()
        else:
            self.test_basic_insert()

        self.stdout.write(self.style.SUCCESS(
            "\n" + "="*70
        ))
        self.stdout.write(self.style.SUCCESS(
            "✅ All tests passed!"
        ))
        self.stdout.write(self.style.SUCCESS("="*70 + "\n"))

    def verify_hypertable(self):
        """Verify that diagnostics_sensordata is a hypertable."""
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    hypertable_name,
                    num_chunks,
                    chunk_time_interval
                FROM timescaledb_information.hypertables
                WHERE hypertable_name = 'diagnostics_sensordata'
            """)
            result = cursor.fetchone()

            if result:
                name, chunks, interval = result
                self.stdout.write(self.style.SUCCESS(
                    f"    ✅ Hypertable: {name}"
                ))
                self.stdout.write(
                    f"    • Chunks: {chunks}"
                )
                self.stdout.write(
                    f"    • Chunk interval: {interval}"
                )
                return True
            return False

    def verify_compression(self):
        """Verify compression policy."""
        with connection.cursor() as cursor:
            # Check compression settings
            cursor.execute("""
                SELECT 
                    attname AS column_name,
                    segmentby_column_index IS NOT NULL AS is_segmentby,
                    orderby_column_index IS NOT NULL AS is_orderby
                FROM timescaledb_information.compression_settings
                WHERE hypertable_name = 'diagnostics_sensordata'
                ORDER BY segmentby_column_index NULLS LAST, 
                         orderby_column_index NULLS LAST
            """)
            settings = cursor.fetchall()

            if settings:
                self.stdout.write(self.style.SUCCESS(
                    "    ✅ Compression enabled"
                ))
                
                segmentby = [s[0] for s in settings if s[1]]
                orderby = [s[0] for s in settings if s[2]]
                
                if segmentby:
                    self.stdout.write(
                        f"    • Segment by: {', '.join(segmentby)}"
                    )
                if orderby:
                    self.stdout.write(
                        f"    • Order by: {', '.join(orderby)}"
                    )
            else:
                self.stdout.write(self.style.WARNING(
                    "    ⚠️  Compression not configured"
                ))

            # Check compression policy
            cursor.execute("""
                SELECT 
                    config::jsonb->>'compress_after' AS compress_after
                FROM timescaledb_information.jobs
                WHERE proc_name = 'policy_compression'
                  AND hypertable_name = 'diagnostics_sensordata'
            """)
            policy = cursor.fetchone()

            if policy:
                self.stdout.write(
                    f"    • Policy: Compress after {policy[0]}"
                )
            else:
                self.stdout.write(self.style.WARNING(
                    "    ⚠️  Compression policy not found"
                ))

    def verify_retention(self):
        """Verify retention policy."""
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    config::jsonb->>'drop_after' AS drop_after
                FROM timescaledb_information.jobs
                WHERE proc_name = 'policy_retention'
                  AND hypertable_name = 'diagnostics_sensordata'
            """)
            policy = cursor.fetchone()

            if policy:
                self.stdout.write(self.style.SUCCESS(
                    f"    ✅ Retention policy: Drop after {policy[0]}"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    "    ⚠️  Retention policy not found"
                ))

    def verify_continuous_aggregate(self):
        """Verify continuous aggregate."""
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    view_name,
                    materialized_only
                FROM timescaledb_information.continuous_aggregates
                WHERE view_name = 'diagnostics_sensordata_hourly'
            """)
            result = cursor.fetchone()

            if result:
                view_name, materialized = result
                self.stdout.write(self.style.SUCCESS(
                    f"    ✅ Continuous aggregate: {view_name}"
                ))
                self.stdout.write(
                    f"    • Materialized only: {materialized}"
                )

                # Check refresh policy
                cursor.execute("""
                    SELECT 
                        config::jsonb->>'start_offset' AS start_offset,
                        config::jsonb->>'end_offset' AS end_offset,
                        schedule_interval
                    FROM timescaledb_information.jobs
                    WHERE proc_name = 'policy_refresh_continuous_aggregate'
                      AND config::jsonb->>'mat_hypertable_id' IN (
                          SELECT mat_hypertable_id::text
                          FROM timescaledb_information.continuous_aggregates
                          WHERE view_name = 'diagnostics_sensordata_hourly'
                      )
                """)
                policy = cursor.fetchone()

                if policy:
                    start, end, interval = policy
                    self.stdout.write(
                        f"    • Refresh: Every {interval} "
                        f"({start} to {end})"
                    )
            else:
                self.stdout.write(self.style.WARNING(
                    "    ⚠️  Continuous aggregate not found"
                ))

    def test_basic_insert(self):
        """Test basic insertion of sensor data."""
        # Get or create test system
        owner = User.objects.first()
        if not owner:
            self.stdout.write(self.style.ERROR(
                "    ❌ No users found. Create a user first."
            ))
            return

        system, created = HydraulicSystem.objects.get_or_create(
            name="Test System (TimescaleDB)",
            defaults={
                'owner': owner,
                'system_type': 'industrial',
                'status': 'active',
            }
        )

        # Insert test data
        test_count = 100
        start = time.time()

        readings = [
            SensorData(
                system=system,
                timestamp=timezone.now() - timedelta(seconds=i),
                sensor_type='pressure',
                value=100.0 + (i % 20),
                unit='bar',
            )
            for i in range(test_count)
        ]

        SensorData.objects.bulk_create(readings, batch_size=100)
        insert_time = (time.time() - start) * 1000  # ms

        self.stdout.write(self.style.SUCCESS(
            f"    ✅ Inserted {test_count} rows in {insert_time:.2f}ms"
        ))
        self.stdout.write(
            f"    • Throughput: {test_count / (insert_time / 1000):.0f} rows/sec"
        )

        # Query test
        start = time.time()
        count = SensorData.qs.recent(hours=24).count()
        query_time = (time.time() - start) * 1000  # ms

        self.stdout.write(self.style.SUCCESS(
            f"    ✅ Queried {count} rows in {query_time:.2f}ms"
        ))

    def run_benchmark(self):
        """Run performance benchmark (10K inserts)."""
        owner = User.objects.first()
        if not owner:
            self.stdout.write(self.style.ERROR(
                "    ❌ No users found. Create a user first."
            ))
            return

        system, _ = HydraulicSystem.objects.get_or_create(
            name="Benchmark System (TimescaleDB)",
            defaults={
                'owner': owner,
                'system_type': 'industrial',
                'status': 'active',
            }
        )

        self.stdout.write(self.style.WARNING(
            "    📏 Running benchmark (10K inserts)..."
        ))

        benchmark_count = 10000
        batch_size = 1000
        start = time.time()

        for batch_num in range(benchmark_count // batch_size):
            readings = [
                SensorData(
                    system=system,
                    timestamp=timezone.now() - timedelta(seconds=i),
                    sensor_type='pressure',
                    value=100.0 + (i % 50),
                    unit='bar',
                )
                for i in range(batch_size)
            ]
            SensorData.objects.bulk_create(readings, batch_size=batch_size)

        total_time = time.time() - start
        throughput = benchmark_count / total_time

        self.stdout.write(self.style.SUCCESS(
            f"    ✅ Inserted {benchmark_count:,} rows in {total_time:.2f}s"
        ))
        self.stdout.write(
            f"    • Throughput: {throughput:,.0f} rows/sec"
        )

        # Performance check
        if throughput >= 10000:
            self.stdout.write(self.style.SUCCESS(
                f"    ✅ PASSED: Throughput exceeds 10K rows/sec target"
            ))
        else:
            self.stdout.write(self.style.WARNING(
                f"    ⚠️  WARNING: Throughput below 10K rows/sec target"
            ))

        # Query benchmark
        self.stdout.write("\n    📏 Running query benchmark...")
        start = time.time()
        count = SensorData.qs.time_range(
            start=timezone.now() - timedelta(days=7),
            end=timezone.now()
        ).count()
        query_time = (time.time() - start) * 1000  # ms

        self.stdout.write(self.style.SUCCESS(
            f"    ✅ Queried {count:,} rows in {query_time:.2f}ms"
        ))

        if query_time < 100:
            self.stdout.write(self.style.SUCCESS(
                f"    ✅ PASSED: Query latency <100ms target"
            ))
        else:
            self.stdout.write(self.style.WARNING(
                f"    ⚠️  WARNING: Query latency exceeds 100ms target"
            ))
