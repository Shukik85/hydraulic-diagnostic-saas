"""
Management Command to Setup Demo Sensor Configuration.

Creates a complete demo setup with:
- Local TLS Modbus node (localhost:1502)
- Hydraulic system sensors (pressure, temperature, flow, vibration, speed)
- Realistic validation ranges and scaling
"""

from django.core.management.base import BaseCommand
from django.db import transaction
from decimal import Decimal

from ...models import SensorNode, SensorConfig


class Command(BaseCommand):
    help = 'Setup demo sensor configuration for development testing'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--host',
            type=str,
            default='127.0.0.1',
            help='Modbus server host address (default: 127.0.0.1)'
        )
        parser.add_argument(
            '--port', 
            type=int,
            default=1502,
            help='Modbus server port (default: 1502)'
        )
        parser.add_argument(
            '--replace',
            action='store_true',
            help='Replace existing demo configuration'
        )
    
    def handle(self, *args, **options):
        host = options['host']
        port = options['port']
        replace = options['replace']
        
        self.stdout.write(
            self.style.SUCCESS(
                f"🚀 Setting up demo sensor configuration for {host}:{port}"
            )
        )
        
        try:
            with transaction.atomic():
                # Create or get sensor node
                node_name = f"Demo TLS Modbus ({host}:{port})"
                
                if replace:
                    # Remove existing demo node and configs
                    SensorNode.objects.filter(name=node_name).delete()
                    self.stdout.write(
                        self.style.WARNING("♾️ Removed existing demo configuration")
                    )
                
                node, created = SensorNode.objects.get_or_create(
                    name=node_name,
                    defaults={
                        'protocol': 'modbus_tcp',
                        'host_address': host,
                        'port': port,
                        'protocol_config': {
                            'unit_id': 1,
                            'read_timeout': 3.0,
                            'tls_enabled': True,  # Ready for TLS
                            'description': 'Demo hydraulic system with TLS Modbus TCP'
                        },
                        'is_active': True,
                        'connection_status': 'disconnected'
                    }
                )
                
                if created:
                    self.stdout.write(
                        self.style.SUCCESS(f"✅ Created sensor node: {node.name}")
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(f"♾️ Using existing sensor node: {node.name}")
                    )
                
                # Create sensor configurations
                sensor_configs = [
                    {
                        'name': 'System Pressure',
                        'register_address': 40001,
                        'data_type': 'uint16',
                        'unit': 'bar',
                        'scale_factor': Decimal('0.1'),  # Scale down by 10 (1500 -> 150.0)
                        'offset': Decimal('0'),
                        'validation_min': Decimal('50'),   # Min 50 bar
                        'validation_max': Decimal('300'),  # Max 300 bar
                        'description': 'Main hydraulic system pressure'
                    },
                    {
                        'name': 'Oil Temperature',
                        'register_address': 40002,
                        'data_type': 'int16',
                        'unit': '°C',
                        'scale_factor': Decimal('1'),
                        'offset': Decimal('0'),
                        'validation_min': Decimal('-10'),  # Min -10°C
                        'validation_max': Decimal('120'),  # Max 120°C
                        'description': 'Hydraulic oil temperature'
                    },
                    {
                        'name': 'Flow Rate',
                        'register_address': 40003,
                        'data_type': 'float32',
                        'unit': 'L/min',
                        'scale_factor': Decimal('1'),
                        'offset': Decimal('0'),
                        'validation_min': Decimal('0'),    # Min 0 L/min
                        'validation_max': Decimal('100'),  # Max 100 L/min
                        'description': 'Pump flow rate (32-bit float in registers 40003-40004)'
                    },
                    {
                        'name': 'Vibration Level',
                        'register_address': 40011,
                        'data_type': 'uint16',
                        'unit': 'mm/s',
                        'scale_factor': Decimal('0.001'),  # Scale down by 1000 (999 -> 0.999)
                        'offset': Decimal('0'),
                        'validation_min': Decimal('0'),     # Min 0 mm/s
                        'validation_max': Decimal('10'),    # Max 10 mm/s
                        'description': 'Pump vibration level'
                    },
                    {
                        'name': 'Motor Speed',
                        'register_address': 40012,
                        'data_type': 'uint16',
                        'unit': 'RPM',
                        'scale_factor': Decimal('1'),
                        'offset': Decimal('0'),
                        'validation_min': Decimal('0'),     # Min 0 RPM
                        'validation_max': Decimal('5000'),  # Max 5000 RPM
                        'description': 'Motor rotation speed'
                    }
                ]
                
                # Create sensor configs
                created_configs = 0
                for config_data in sensor_configs:
                    config, config_created = SensorConfig.objects.get_or_create(
                        node=node,
                        register_address=config_data['register_address'],
                        defaults={
                            **config_data,
                            'is_active': True
                        }
                    )
                    
                    if config_created:
                        created_configs += 1
                        self.stdout.write(
                            f"✅ Created sensor: {config.name} "
                            f"(reg: {config.register_address}, type: {config.data_type})"
                        )
                    else:
                        self.stdout.write(
                            f"♾️ Updated sensor: {config.name}"
                        )
                
                self.stdout.write(
                    self.style.SUCCESS(
                        f"🎉 Demo setup completed! "
                        f"Created {created_configs} new sensor configurations."
                    )
                )
                
                # Print summary
                self.stdout.write("\n📋 Demo Configuration Summary:")
                self.stdout.write(f"   Node: {node.name}")
                self.stdout.write(f"   Protocol: {node.protocol}")
                self.stdout.write(f"   Address: {node.host_address}:{node.port}")
                self.stdout.write(f"   Sensors: {node.sensors.count()} configured")
                self.stdout.write("\n💡 Expected Register Values:")
                self.stdout.write("   HR[0] (40001) = 1500 -> 150.0 bar (System Pressure)")
                self.stdout.write("   HR[1] (40002) = 65 -> 65°C (Oil Temperature)")
                self.stdout.write("   HR[2-3] (40003-40004) = float32 25.5 L/min (Flow Rate)")
                self.stdout.write("   HR[10] (40011) = 999 -> 0.999 mm/s (Vibration Level)")
                self.stdout.write("   HR[11] (40012) = 1500 -> 1500 RPM (Motor Speed)")
                
                self.stdout.write("\n🚀 Next Steps:")
                self.stdout.write("   1. Start Celery worker: celery -A project worker -l info")
                self.stdout.write("   2. Start Celery beat: celery -A project beat -l info")
                self.stdout.write("   3. Check API: GET /api/sensors/nodes/")
                self.stdout.write("   4. Monitor readings: GET /api/sensors/readings/latest/")
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"💥 Demo setup failed: {e}")
            )
            raise
