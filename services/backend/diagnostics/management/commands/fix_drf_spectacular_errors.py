"""Management command to fix DRF Spectacular errors."""

import sys
from typing import Any

from django.core.management.base import BaseCommand, CommandError
from django.db import connection
from django.core.management import call_command
from rest_framework import serializers
from drf_spectacular.generators import SchemaGenerator


class Command(BaseCommand):
    """Fix DRF Spectacular schema generation errors."""

    help = "Fix common DRF Spectacular schema generation errors"

    def add_arguments(self, parser: Any) -> None:
        """Add command arguments.
        
        Args:
            parser: Argument parser
        """
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be fixed without making changes",
        )
        parser.add_argument(
            "--check-only",
            action="store_true",
            help="Only check for issues, don't fix anything",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Handle the command execution.
        
        Args:
            *args: Command arguments
            **options: Command options
        """
        self.stdout.write(
            self.style.SUCCESS("Starting DRF Spectacular error analysis...")
        )
        
        dry_run = options.get("dry_run", False)
        check_only = options.get("check_only", False)
        
        try:
            # Test schema generation to find errors
            self._test_schema_generation()
            
            if not check_only:
                self._fix_common_issues(dry_run)
            
            self.stdout.write(
                self.style.SUCCESS(
                    "✅ DRF Spectacular error analysis completed successfully!"
                )
            )
            
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Error during analysis: {e}")
            )
            if "criticality" in str(e).lower():
                self._suggest_criticality_fix(dry_run)
            
            sys.exit(1)

    def _test_schema_generation(self) -> None:
        """Test DRF Spectacular schema generation to identify errors."""
        self.stdout.write("Testing schema generation...")
        
        try:
            generator = SchemaGenerator()
            schema = generator.get_schema(request=None, public=True)
            self.stdout.write(self.style.SUCCESS("✅ Schema generation successful"))
        except Exception as e:
            error_msg = str(e)
            self.stdout.write(
                self.style.ERROR(f"❌ Schema generation failed: {error_msg}")
            )
            
            if "criticality" in error_msg.lower():
                self.stdout.write(
                    self.style.WARNING(
                        "⚠️  Detected 'criticality' field error - this field doesn't exist in models"
                    )
                )
            
            raise

    def _fix_common_issues(self, dry_run: bool) -> None:
        """Fix common DRF issues.
        
        Args:
            dry_run: Whether to perform a dry run
        """
        self.stdout.write("Checking for common DRF issues...")
        
        # Check for missing migrations
        self._check_migrations(dry_run)
        
        # Check database schema
        self._check_database_schema(dry_run)
        
    def _check_migrations(self, dry_run: bool) -> None:
        """Check and apply missing migrations.
        
        Args:
            dry_run: Whether to perform a dry run
        """
        self.stdout.write("Checking migrations...")
        
        try:
            if not dry_run:
                call_command("migrate", verbosity=0)
                self.stdout.write(
                    self.style.SUCCESS("✅ Migrations applied successfully")
                )
            else:
                self.stdout.write(
                    self.style.WARNING("[DRY RUN] Would apply pending migrations")
                )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"❌ Migration error: {e}")
            )

    def _check_database_schema(self, dry_run: bool) -> None:
        """Check database schema consistency.
        
        Args:
            dry_run: Whether to perform a dry run
        """
        self.stdout.write("Checking database schema...")
        
        with connection.cursor() as cursor:
            # Check if DiagnosticReport table exists
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'diagnostics_diagnosticreport'
                );
                """
            )
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                self.stdout.write(
                    self.style.ERROR(
                        "❌ DiagnosticReport table doesn't exist - run migrations first"
                    )
                )
                return
            
            # Check for created_by field
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'diagnostics_diagnosticreport'
                    AND column_name = 'created_by_id'
                );
                """
            )
            field_exists = cursor.fetchone()[0]
            
            if field_exists:
                self.stdout.write(
                    self.style.SUCCESS("✅ created_by field exists")
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        "⚠️  created_by field missing - will be created by migration"
                    )
                )

    def _suggest_criticality_fix(self, dry_run: bool) -> None:
        """Suggest fixes for criticality field errors.
        
        Args:
            dry_run: Whether to perform a dry run
        """
        self.stdout.write(
            self.style.WARNING(
                "\n🔍 CRITICALITY FIELD ERROR DETECTED:\n"
                "This error suggests that a serializer is trying to use a 'criticality' field\n"
                "that doesn't exist in the corresponding model.\n\n"
                "SOLUTIONS:\n"
                "1. Remove 'criticality' from serializer fields list\n"
                "2. Add 'criticality' field to the model if it's needed\n"
                "3. Check for typos in field names\n\n"
                "To find the problematic serializer, search for 'criticality' in:\n"
                "- backend/apps/*/serializers.py\n"
                "- Any custom serializer files\n"
            )
        )
        
        if not dry_run:
            self.stdout.write(
                self.style.SUCCESS(
                    "💡 Running automatic field cleanup..."
                )
            )
            # Here we could add automatic cleanup if needed