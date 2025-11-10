"""
Sensor Data Models for TimescaleDB.

Optimized for high-frequency industrial sensor data:
- HyperTable partitioning by timestamp (7-day chunks)
- Compression after 30 days
- 5-year retention policy
- Efficient indexing for time-series queries
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


class SensorNodeQuerySet(models.QuerySet["SensorNode"]):
    """Custom QuerySet for SensorNode model."""
    
    def active(self) -> "SensorNodeQuerySet":
        """Filter only active sensor nodes."""
        return self.filter(is_active=True)
    
    def by_protocol(self, protocol: str) -> "SensorNodeQuerySet":
        """Filter by communication protocol."""
        return self.filter(protocol=protocol)
    
    def with_recent_data(self, hours: int = 24) -> "SensorNodeQuerySet":
        """Filter nodes with recent data within specified hours."""
        cutoff = timezone.now() - timedelta(hours=hours)
        return self.filter(last_data_received__gte=cutoff)


class SensorNode(models.Model):
    """Industrial sensor node (PLC, gateway, device) configuration."""
    
    PROTOCOL_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('modbus_tcp', 'Modbus TCP'),
        ('modbus_rtu', 'Modbus RTU'),
        ('opcua', 'OPC UA'),
        ('http_json', 'HTTP JSON'),
        ('mqtt', 'MQTT'),
        ('ethernet_ip', 'Ethernet/IP'),
    ]
    
    STATUS_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('error', 'Error'),
        ('maintenance', 'Maintenance'),
    ]
    
    id: models.UUIDField = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False
    )
    
    # Basic identification
    name: models.CharField = models.CharField(
        max_length=255, 
        db_index=True,
        help_text="Human-readable sensor node name"
    )
    description: models.TextField = models.TextField(
        blank=True, 
        default="",
        help_text="Optional description of the sensor node"
    )
    
    # Network configuration
    protocol: models.CharField = models.CharField(
        max_length=32, 
        choices=PROTOCOL_CHOICES, 
        db_index=True
    )
    host_address: models.GenericIPAddressField = models.GenericIPAddressField(
        help_text="IP address of the sensor node"
    )
    port: models.PositiveIntegerField = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(65535)],
        help_text="Network port"
    )
    
    # Protocol-specific configuration (JSON)
    protocol_config: models.JSONField = models.JSONField(
        default=dict,
        blank=True,
        help_text="Protocol-specific configuration (timeouts, unit ID, etc.)"
    )
    
    # Status and monitoring
    is_active: models.BooleanField = models.BooleanField(
        default=True, 
        db_index=True,
        help_text="Enable/disable data collection from this node"
    )
    status: models.CharField = models.CharField(
        max_length=20, 
        choices=STATUS_CHOICES, 
        default='offline', 
        db_index=True
    )
    
    last_data_received: models.DateTimeField = models.DateTimeField(
        null=True, 
        blank=True, 
        db_index=True,
        help_text="Timestamp of last successful data reception"
    )
    last_error: models.TextField = models.TextField(
        blank=True, 
        default="",
        help_text="Last error message"
    )
    
    # Metadata
    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now, 
        db_index=True
    )
    updated_at: models.DateTimeField = models.DateTimeField(
        auto_now=True
    )
    
    objects = models.Manager()
    qs: "SensorNodeQuerySet" = SensorNodeQuerySet.as_manager()  # type: ignore[assignment]
    
    if TYPE_CHECKING:
        sensor_readings: RelatedManager["SensorReading"]
        sensor_configs: RelatedManager["SensorConfig"]
    
    class Meta:
        db_table = "sensors_node"
        ordering = ["name"]
        indexes = [
            BTreeIndex(fields=["protocol", "is_active"], name="idx_sn_protocol_active"),
            BTreeIndex(fields=["status", "last_data_received"], name="idx_sn_status_lastdata"),
        ]
    
    def __str__(self) -> str:
        return f"{self.name} ({self.protocol}@{self.host_address}:{self.port})"
    
    def get_connection_url(self) -> str:
        """Generate connection URL for this sensor node."""
        return f"{self.protocol}://{self.host_address}:{self.port}"


class SensorConfigQuerySet(models.QuerySet["SensorConfig"]):
    """Custom QuerySet for SensorConfig model."""
    
    def for_node(self, node_id: uuid.UUID) -> "SensorConfigQuerySet":
        """Filter configurations for specific node."""
        return self.filter(node_id=node_id)
    
    def active(self) -> "SensorConfigQuerySet":
        """Filter only active sensor configurations."""
        return self.filter(is_active=True)


class SensorConfig(models.Model):
    """Individual sensor configuration within a node."""
    
    SENSOR_TYPE_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('pressure', 'Pressure'),
        ('temperature', 'Temperature'),
        ('flow_rate', 'Flow Rate'),
        ('vibration', 'Vibration'),
        ('speed', 'Speed/RPM'),
        ('position', 'Position'),
        ('force', 'Force'),
        ('voltage', 'Voltage'),
        ('current', 'Current'),
        ('power', 'Power'),
    ]
    
    DATA_TYPE_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('float32', 'Float 32-bit'),
        ('float64', 'Float 64-bit'),
        ('int16', 'Integer 16-bit'),
        ('int32', 'Integer 32-bit'),
        ('uint16', 'Unsigned Integer 16-bit'),
        ('uint32', 'Unsigned Integer 32-bit'),
        ('bool', 'Boolean'),
    ]
    
    id: models.UUIDField = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False
    )
    
    node: models.ForeignKey = models.ForeignKey(
        SensorNode,
        on_delete=models.CASCADE,
        related_name='sensor_configs',
        db_index=True
    )
    
    # Sensor identification
    sensor_name: models.CharField = models.CharField(
        max_length=255,
        help_text="Human-readable sensor name"
    )
    sensor_type: models.CharField = models.CharField(
        max_length=32, 
        choices=SENSOR_TYPE_CHOICES,
        db_index=True
    )
    
    # Protocol-specific addressing
    register_address: models.PositiveIntegerField = models.PositiveIntegerField(
        help_text="Register address (Modbus) or node ID (OPC UA)"
    )
    data_type: models.CharField = models.CharField(
        max_length=16, 
        choices=DATA_TYPE_CHOICES,
        default='float32'
    )
    
    # Data processing
    scale_factor: models.FloatField = models.FloatField(
        default=1.0,
        help_text="Multiply raw value by this factor"
    )
    offset: models.FloatField = models.FloatField(
        default=0.0,
        help_text="Add this offset after scaling"
    )
    unit: models.CharField = models.CharField(
        max_length=32, 
        blank=True, 
        default="",
        help_text="Physical unit (bar, °C, L/min, etc.)"
    )
    
    # Sampling configuration
    sampling_interval_seconds: models.PositiveIntegerField = models.PositiveIntegerField(
        default=10,
        validators=[MinValueValidator(1), MaxValueValidator(3600)],
        help_text="Data collection interval in seconds"
    )
    
    # Quality control
    min_valid_value: models.FloatField = models.FloatField(
        null=True, 
        blank=True,
        help_text="Minimum valid sensor value (for range validation)"
    )
    max_valid_value: models.FloatField = models.FloatField(
        null=True, 
        blank=True,
        help_text="Maximum valid sensor value (for range validation)"
    )
    
    is_active: models.BooleanField = models.BooleanField(
        default=True, 
        db_index=True
    )
    
    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now
    )
    updated_at: models.DateTimeField = models.DateTimeField(
        auto_now=True
    )
    
    objects = models.Manager()
    qs: "SensorConfigQuerySet" = SensorConfigQuerySet.as_manager()  # type: ignore[assignment]
    
    if TYPE_CHECKING:
        sensor_readings: RelatedManager["SensorReading"]
    
    class Meta:
        db_table = "sensors_config"
        ordering = ["node", "register_address"]
        unique_together = ["node", "register_address"]
        indexes = [
            BTreeIndex(fields=["node", "sensor_type"], name="idx_sc_node_type"),
            BTreeIndex(fields=["is_active", "sampling_interval_seconds"], name="idx_sc_active_interval"),
        ]
    
    def __str__(self) -> str:
        return f"{self.node.name}::{self.sensor_name} ({self.sensor_type})"
    
    def validate_value(self, value: float) -> bool:
        """Validate sensor reading against configured min/max values."""
        if self.min_valid_value is not None and value < self.min_valid_value:
            return False
        if self.max_valid_value is not None and value > self.max_valid_value:
            return False
        return True
    
    def process_raw_value(self, raw_value: float) -> float:
        """Apply scale factor and offset to raw sensor value."""
        return (raw_value * self.scale_factor) + self.offset


class SensorReadingQuerySet(models.QuerySet["SensorReading"]):
    """Custom QuerySet for SensorReading model with TimescaleDB optimizations."""
    
    def for_node(self, node_id: uuid.UUID) -> "SensorReadingQuerySet":
        """Filter readings for specific node."""
        return self.filter(node_id=node_id)
    
    def for_sensor(self, sensor_config_id: uuid.UUID) -> "SensorReadingQuerySet":
        """Filter readings for specific sensor configuration."""
        return self.filter(sensor_config_id=sensor_config_id)
    
    def time_range(self, start: datetime, end: datetime) -> "SensorReadingQuerySet":
        """Filter readings within time range."""
        return self.filter(timestamp__gte=start, timestamp__lt=end)
    
    def recent(self, hours: int = 24) -> "SensorReadingQuerySet":
        """Get recent readings within specified hours."""
        cutoff = timezone.now() - timedelta(hours=hours)
        return self.filter(timestamp__gte=cutoff)
    
    def valid_only(self) -> "SensorReadingQuerySet":
        """Filter only valid readings (not quarantined)."""
        return self.filter(is_quarantined=False)


class SensorReading(models.Model):
    """High-frequency sensor reading data - TimescaleDB hypertable.
    
    This model is designed for TimescaleDB hypertable with:
    - Time-based partitioning (7-day chunks)
    - Compression after 30 days
    - 5-year retention policy
    """
    
    QUALITY_CHOICES: ClassVar[list[tuple[str, str]]] = [
        ('good', 'Good'),
        ('uncertain', 'Uncertain'),
        ('bad', 'Bad'),
    ]
    
    id: models.UUIDField = models.UUIDField(
        primary_key=True, 
        default=uuid.uuid4, 
        editable=False
    )
    
    # Foreign keys (optimized for TimescaleDB)
    node: models.ForeignKey = models.ForeignKey(
        SensorNode,
        on_delete=models.CASCADE,
        related_name='sensor_readings',
        db_index=True
    )
    sensor_config: models.ForeignKey = models.ForeignKey(
        SensorConfig,
        on_delete=models.CASCADE,
        related_name='sensor_readings',
        db_index=True
    )
    
    # Time-series data (partition key)
    timestamp: models.DateTimeField = models.DateTimeField(
        db_index=True,
        help_text="Sensor reading timestamp"
    )
    
    # Sensor values
    raw_value: models.FloatField = models.FloatField(
        help_text="Raw value from sensor before processing"
    )
    processed_value: models.FloatField = models.FloatField(
        help_text="Processed value (scaled and offset applied)"
    )
    
    # Data quality and validation
    quality: models.CharField = models.CharField(
        max_length=16, 
        choices=QUALITY_CHOICES,
        default='good',
        db_index=True
    )
    is_quarantined: models.BooleanField = models.BooleanField(
        default=False,
        db_index=True,
        help_text="True if reading failed validation and is quarantined"
    )
    quarantine_reason: models.CharField = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text="Reason for quarantine (out of range, communication error, etc.)"
    )
    
    # Collection metadata
    collection_latency_ms: models.PositiveIntegerField = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Time taken to collect this reading (milliseconds)"
    )
    
    created_at: models.DateTimeField = models.DateTimeField(
        default=timezone.now,
        db_index=True
    )
    
    objects = models.Manager()
    qs: "SensorReadingQuerySet" = SensorReadingQuerySet.as_manager()  # type: ignore[assignment]
    
    class Meta:
        db_table = "sensors_reading"
        ordering = ["-timestamp"]
        
        # TimescaleDB optimized indexes
        indexes = [
            # Primary time-based index (most important for TimescaleDB)
            BTreeIndex(fields=["timestamp", "node"], name="idx_sr_time_node"),
            BTreeIndex(fields=["timestamp", "sensor_config"], name="idx_sr_time_config"),
            
            # Quality and quarantine indexes
            BTreeIndex(fields=["is_quarantined", "timestamp"], name="idx_sr_quarantine_time"),
            BTreeIndex(fields=["quality", "timestamp"], name="idx_sr_quality_time"),
            
            # BRIN index for timestamp (efficient for TimescaleDB)
            BrinIndex(fields=["timestamp"], autosummarize=True, name="brin_sr_timestamp"),
        ]
    
    def clean(self) -> None:
        """Validate sensor reading data."""
        # Validate timestamp is not too far in the future
        if self.timestamp and self.timestamp > timezone.now() + timedelta(minutes=5):
            raise ValidationError("Timestamp cannot be more than 5 minutes in the future")
        
        # Validate sensor configuration matches node
        if self.sensor_config and self.node and self.sensor_config.node_id != self.node.id:
            raise ValidationError("Sensor configuration must belong to the specified node")
    
    def save(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Save with validation and automatic processing."""
        # Validate before saving
        self.full_clean()
        
        # Auto-process raw value if not already done
        if self.sensor_config and hasattr(self, '_state') and self._state.adding:
            self.processed_value = self.sensor_config.process_raw_value(self.raw_value)
            
            # Validate processed value
            if not self.sensor_config.validate_value(self.processed_value):
                self.is_quarantined = True
                self.quality = 'bad'
                self.quarantine_reason = f"Value {self.processed_value} outside valid range [{self.sensor_config.min_valid_value}, {self.sensor_config.max_valid_value}]"
        
        super().save(*args, **kwargs)
        
        # Update node's last data received timestamp
        if self.node:
            SensorNode.objects.filter(id=self.node.id).update(
                last_data_received=self.timestamp,
                status='online'
            )
    
    def __str__(self) -> str:
        return f"{self.sensor_config.sensor_name}@{self.timestamp}: {self.processed_value} {self.sensor_config.unit}"


__all__ = [
    'SensorNode',
    'SensorConfig',
    'SensorReading',
]
