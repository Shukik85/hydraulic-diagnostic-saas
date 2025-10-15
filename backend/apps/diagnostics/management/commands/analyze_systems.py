from django.core.management.base import BaseCommand
from django.utils import timezone
from apps.equipment.models import HydraulicSystem
from apps.diagnostics.services import DiagnosticEngine
import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Analyze all active hydraulic systems and create diagnostics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--system-id',
            type=int,
            help='Analyze specific system by ID',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force analysis even for systems with recent diagnostics',
        )

    def handle(self, *args, **options):
        system_id = options.get('system_id')
        force = options.get('force', False)

        # Get systems to analyze
        if system_id:
            systems = HydraulicSystem.objects.filter(id=system_id, is_active=True)
            if not systems.exists():
                self.stdout.write(
                    self.style.ERROR(f'System with ID {system_id} not found or inactive')
                )
                return
        else:
            systems = HydraulicSystem.objects.filter(is_active=True)

        total_systems = systems.count()
        self.stdout.write(
            self.style.SUCCESS(f'Found {total_systems} active system(s) to analyze')
        )

        success_count = 0
        error_count = 0

        for system in systems:
            try:
                self.stdout.write(f'\nAnalyzing system: {system.name} (ID: {system.id})')
                
                # Check if system has recent diagnostics
                if not force:
                    recent_diagnostic = system.diagnostics.filter(
                        created_at__gte=timezone.now() - timezone.timedelta(hours=1)
                    ).first()
                    
                    if recent_diagnostic:
                        self.stdout.write(
                            self.style.WARNING(
                                f'  Skipping - recent diagnostic found (created {recent_diagnostic.created_at})'
                            )
                        )
                        continue

                # Get latest sensor data
                latest_data = system.sensor_data.order_by('-timestamp').first()
                
                if not latest_data:
                    self.stdout.write(
                        self.style.WARNING('  No sensor data available for this system')
                    )
                    continue

                # Create diagnostic engine and perform analysis
                engine = DiagnosticEngine(system)
                diagnostic = engine.analyze()

                if diagnostic:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'  ✓ Diagnostic created: {diagnostic.diagnosis_type} - '
                            f'Severity: {diagnostic.severity}, Confidence: {diagnostic.confidence_score}%'
                        )
                    )
                    success_count += 1
                else:
                    self.stdout.write(
                        self.style.WARNING('  No diagnostic was created')
                    )
                    error_count += 1

            except Exception as e:
                logger.error(f'Error analyzing system {system.id}: {str(e)}', exc_info=True)
                self.stdout.write(
                    self.style.ERROR(f'  ✗ Error: {str(e)}')
                )
                error_count += 1

        # Summary
        self.stdout.write('\n' + '=' * 50)
        self.stdout.write(self.style.SUCCESS(f'Analysis complete!'))
        self.stdout.write(f'Total systems analyzed: {total_systems}')
        self.stdout.write(self.style.SUCCESS(f'Successful: {success_count}'))
        if error_count > 0:
            self.stdout.write(self.style.ERROR(f'Errors: {error_count}'))
        self.stdout.write('=' * 50)
