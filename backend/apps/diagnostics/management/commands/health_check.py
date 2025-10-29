"""Health check command for monitoring system status."""

import sys
from django.core.management.base import BaseCommand
from django.db import connection
from django.core.cache import cache
from django.conf import settings
from celery import current_app
import redis
from apps.rag_assistant.models import RagSystem


class Command(BaseCommand):
    """Comprehensive health check for all system components."""
    
    help = "Perform health checks on all system components"
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--component',
            type=str,
            help='Check specific component: db, redis, timescale, celery, rag, all'
        )
        parser.add_argument(
            '--json',
            action='store_true',
            help='Output in JSON format'
        )
    
    def handle(self, *args, **options):
        component = options.get('component', 'all')
        json_output = options.get('json', False)
        
        checks = {
            'database': self.check_database,
            'redis': self.check_redis,
            'timescale': self.check_timescale,
            'celery': self.check_celery,
            'rag': self.check_rag_system,
        }
        
        results = {}
        overall_health = True
        
        if component == 'all':
            for name, check_func in checks.items():
                try:
                    results[name] = check_func()
                    if not results[name]['healthy']:
                        overall_health = False
                except Exception as e:
                    results[name] = {'healthy': False, 'error': str(e)}
                    overall_health = False
        elif component in checks:
            try:
                results[component] = checks[component]()
                overall_health = results[component]['healthy']
            except Exception as e:
                results[component] = {'healthy': False, 'error': str(e)}
                overall_health = False
        else:
            self.stdout.write(
                self.style.ERROR(f"Unknown component: {component}")
            )
            sys.exit(1)
        
        if json_output:
            import json
            output = {
                'healthy': overall_health,
                'components': results,
                'timestamp': self.get_timestamp()
            }
            self.stdout.write(json.dumps(output, indent=2))
        else:
            self.output_human_readable(results, overall_health)
        
        sys.exit(0 if overall_health else 1)
    
    def check_database(self):
        """Check database connectivity."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    return {'healthy': True, 'message': 'Database connected'}
                else:
                    return {'healthy': False, 'message': 'Database query failed'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def check_redis(self):
        """Check Redis connectivity."""
        try:
            cache.set('health_check', 'ok', 60)
            value = cache.get('health_check')
            if value == 'ok':
                cache.delete('health_check')
                return {'healthy': True, 'message': 'Redis connected'}
            else:
                return {'healthy': False, 'message': 'Redis set/get failed'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def check_timescale(self):
        """Check TimescaleDB extension."""
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT installed_version FROM pg_available_extensions WHERE name = 'timescaledb'"
                )
                result = cursor.fetchone()
                if result and result[0]:
                    return {'healthy': True, 'message': f'TimescaleDB {result[0]} available'}
                else:
                    return {'healthy': False, 'message': 'TimescaleDB extension not found'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def check_celery(self):
        """Check Celery workers status."""
        try:
            inspect = current_app.control.inspect()
            stats = inspect.stats()
            if stats:
                active_workers = len(stats)
                return {
                    'healthy': True, 
                    'message': f'{active_workers} Celery workers active',
                    'workers': list(stats.keys())
                }
            else:
                return {'healthy': False, 'message': 'No active Celery workers found'}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def check_rag_system(self):
        """Check RAG system readiness."""
        try:
            rag_systems = RagSystem.objects.count()
            if rag_systems > 0:
                return {
                    'healthy': True, 
                    'message': f'{rag_systems} RAG system(s) configured'
                }
            else:
                return {
                    'healthy': True, 
                    'message': 'RAG system ready (no systems configured yet)'
                }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def get_timestamp(self):
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'
    
    def output_human_readable(self, results, overall_health):
        """Output health check results in human-readable format."""
        self.stdout.write("\n🏥 Hydraulic Diagnostic SaaS - Health Check Report\n")
        self.stdout.write("=" * 50)
        
        for component, result in results.items():
            status_icon = "✅" if result['healthy'] else "❌"
            self.stdout.write(f"\n{status_icon} {component.upper()}:")
            
            if 'message' in result:
                self.stdout.write(f"   {result['message']}")
            
            if 'error' in result:
                self.stdout.write(
                    self.style.ERROR(f"   Error: {result['error']}")
                )
            
            if 'workers' in result:
                self.stdout.write(f"   Workers: {', '.join(result['workers'])}")
        
        self.stdout.write("\n" + "=" * 50)
        
        if overall_health:
            self.stdout.write(
                self.style.SUCCESS("\n🎉 Overall Status: HEALTHY\n")
            )
        else:
            self.stdout.write(
                self.style.ERROR("\n⚠️  Overall Status: UNHEALTHY\n")
            )