"""Unit tests для Pydantic schemas.

Полное покрытие всех schemas, validators, computed fields.

Pytest fixtures в conftest.py.
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.schemas import (
    # Enums
    ComponentType,
    EdgeType,
    SensorType,
    ComponentStatus,
    AnomalyType,
    # Graph
    EdgeSpec,
    ComponentSpec,
    GraphTopology,
    # Metadata
    SensorConfig,
    EquipmentMetadata,
    SystemConfig,
    # Requests
    TimeWindow,
    InferenceRequest,
    BatchInferenceRequest,
    TrainingRequest,
    # Responses
    ComponentHealth,
    Anomaly,
    InferenceResponse,
    TrainingResponse,
)


# ==================== GRAPH SCHEMAS TESTS ====================

class TestEdgeSpec:
    """Tests для EdgeSpec."""
    
    def test_edge_spec_creation_valid(self):
        """Создание valid EdgeSpec."""
        edge = EdgeSpec(
            source_id="pump_001",
            target_id="valve_001",
            edge_type=EdgeType.HIGH_PRESSURE_HOSE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350
        )
        
        assert edge.source_id == "pump_001"
        assert edge.target_id == "valve_001"
        assert edge.edge_type == EdgeType.HIGH_PRESSURE_HOSE
        assert edge.diameter_mm == 16.0
        assert edge.length_m == 2.5
        assert edge.pressure_rating_bar == 350
        assert edge.flow_direction == "unidirectional"  # default
        assert edge.material == "steel"  # default
        assert not edge.has_quick_disconnect  # default
    
    def test_edge_spec_computed_cross_section_area(self):
        """Вычисление cross_section_area_mm2."""
        edge = EdgeSpec(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.HYDRAULIC_LINE,
            diameter_mm=10.0,
            length_m=1.0,
            pressure_rating_bar=200
        )
        
        # Площадь = π * (d/2)^2 = π * 25 ≈ 78.54
        assert pytest.approx(edge.cross_section_area_mm2, rel=0.01) == 78.54
    
    def test_edge_spec_computed_pressure_loss(self):
        """Вычисление pressure_loss_coefficient."""
        edge = EdgeSpec(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.HIGH_PRESSURE_HOSE,
            diameter_mm=16.0,
            length_m=2.5,
            pressure_rating_bar=350,
            material="steel"
        )
        
        # factor=1.0, length=2.5, diameter=16
        # coeff = 1.0 * 2.5 / 16^4 = 0.000038
        assert edge.pressure_loss_coefficient > 0
    
    def test_edge_spec_invalid_diameter(self):
        """Негативный диаметр должен вызывать ошибку."""
        with pytest.raises(ValidationError):
            EdgeSpec(
                source_id="a",
                target_id="b",
                edge_type=EdgeType.HYDRAULIC_LINE,
                diameter_mm=-5.0,  # Invalid
                length_m=1.0,
                pressure_rating_bar=200
            )
    
    def test_edge_spec_frozen(self):
        """Нельзя изменить frozen EdgeSpec."""
        edge = EdgeSpec(
            source_id="a",
            target_id="b",
            edge_type=EdgeType.HYDRAULIC_LINE,
            diameter_mm=10.0,
            length_m=1.0,
            pressure_rating_bar=200
        )
        
        with pytest.raises(ValidationError):
            edge.diameter_mm = 20.0  # Cannot modify frozen


class TestComponentSpec:
    """Tests для ComponentSpec."""
    
    def test_component_spec_creation_valid(self):
        """Создание valid ComponentSpec."""
        component = ComponentSpec(
            component_id="pump_001",
            component_type=ComponentType.PISTON_PUMP,
            sensors=["pressure_in", "pressure_out", "temperature"],
            feature_dim=12,
            nominal_pressure_bar=280,
            nominal_flow_lpm=120,
            rated_power_kw=45.0
        )
        
        assert component.component_id == "pump_001"
        assert component.component_type == ComponentType.PISTON_PUMP
        assert len(component.sensors) == 3
        assert component.feature_dim == 12
    
    def test_component_spec_computed_power_density(self):
        """Вычисление power_density."""
        component = ComponentSpec(
            component_id="pump_001",
            component_type=ComponentType.PISTON_PUMP,
            sensors=["pressure"],
            feature_dim=8,
            nominal_pressure_bar=280,
            nominal_flow_lpm=120,
            rated_power_kw=60.0
        )
        
        # power_density = 60 / 120 = 0.5 kW per L/min
        assert component.power_density == 0.5
    
    def test_component_spec_duplicate_sensors(self):
        """Дубликаты sensor IDs должны вызывать ошибку."""
        with pytest.raises(ValidationError, match="must be unique"):
            ComponentSpec(
                component_id="pump_001",
                component_type=ComponentType.PISTON_PUMP,
                sensors=["pressure", "pressure"],  # Duplicate!
                feature_dim=8,
                nominal_pressure_bar=280,
                nominal_flow_lpm=120
            )
    
    def test_component_spec_invalid_id_pattern(self):
        """Невалидный pattern component_id."""
        with pytest.raises(ValidationError):
            ComponentSpec(
                component_id="pump 001",  # Space not allowed
                component_type=ComponentType.PISTON_PUMP,
                sensors=["pressure"],
                feature_dim=8,
                nominal_pressure_bar=280,
                nominal_flow_lpm=120
            )


class TestGraphTopology:
    """Tests для GraphTopology."""
    
    def test_graph_topology_creation_valid(self):
        """Создание valid GraphTopology."""
        topology = GraphTopology(
            equipment_id="excavator_001",
            components={
                "pump_001": ComponentSpec(
                    component_id="pump_001",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["pressure"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "valve_001": ComponentSpec(
                    component_id="valve_001",
                    component_type=ComponentType.DIRECTIONAL_VALVE,
                    sensors=["position"],
                    feature_dim=6,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                )
            },
            edges=[
                EdgeSpec(
                    source_id="pump_001",
                    target_id="valve_001",
                    edge_type=EdgeType.HIGH_PRESSURE_HOSE,
                    diameter_mm=16.0,
                    length_m=2.5,
                    pressure_rating_bar=350
                )
            ]
        )
        
        assert topology.equipment_id == "excavator_001"
        assert topology.num_components == 2
        assert topology.num_edges == 1
        assert topology.avg_degree == 1.0
    
    def test_graph_topology_key_mismatch(self):
        """Ключ должен совпадать с component_id."""
        with pytest.raises(ValidationError, match="does not match component_id"):
            GraphTopology(
                equipment_id="excavator_001",
                components={
                    "wrong_key": ComponentSpec(
                        component_id="pump_001",  # Mismatch!
                        component_type=ComponentType.PISTON_PUMP,
                        sensors=["pressure"],
                        feature_dim=8,
                        nominal_pressure_bar=280,
                        nominal_flow_lpm=120
                    )
                },
                edges=[]
            )
    
    def test_graph_topology_edge_references_nonexistent_component(self):
        """Ребро не может ссылаться на несуществующий компонент."""
        with pytest.raises(ValidationError, match="not found in components"):
            GraphTopology(
                equipment_id="excavator_001",
                components={
                    "pump_001": ComponentSpec(
                        component_id="pump_001",
                        component_type=ComponentType.PISTON_PUMP,
                        sensors=["pressure"],
                        feature_dim=8,
                        nominal_pressure_bar=280,
                        nominal_flow_lpm=120
                    )
                },
                edges=[
                    EdgeSpec(
                        source_id="pump_001",
                        target_id="nonexistent",  # Does not exist!
                        edge_type=EdgeType.HYDRAULIC_LINE,
                        diameter_mm=10.0,
                        length_m=1.0,
                        pressure_rating_bar=200
                    )
                ]
            )
    
    def test_graph_topology_connectivity_check(self):
        """Проверка связности графа."""
        # Connected graph
        topology_connected = GraphTopology(
            equipment_id="exc_001",
            components={
                "pump": ComponentSpec(
                    component_id="pump",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["p"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "valve": ComponentSpec(
                    component_id="valve",
                    component_type=ComponentType.DIRECTIONAL_VALVE,
                    sensors=["pos"],
                    feature_dim=6,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                )
            },
            edges=[
                EdgeSpec(
                    source_id="pump",
                    target_id="valve",
                    edge_type=EdgeType.HYDRAULIC_LINE,
                    diameter_mm=10.0,
                    length_m=1.0,
                    pressure_rating_bar=200
                )
            ]
        )
        
        assert topology_connected.validate_connectivity() is True
        
        # Disconnected graph
        topology_disconnected = GraphTopology(
            equipment_id="exc_002",
            components={
                "pump": ComponentSpec(
                    component_id="pump",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["p"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "valve": ComponentSpec(
                    component_id="valve",
                    component_type=ComponentType.DIRECTIONAL_VALVE,
                    sensors=["pos"],
                    feature_dim=6,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                )
            },
            edges=[]  # No edges - disconnected!
        )
        
        assert topology_disconnected.validate_connectivity() is False
    
    def test_graph_topology_pyg_format_conversion(self):
        """Конвертация в PyTorch Geometric format."""
        topology = GraphTopology(
            equipment_id="exc_001",
            components={
                "pump": ComponentSpec(
                    component_id="pump",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["p"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "valve": ComponentSpec(
                    component_id="valve",
                    component_type=ComponentType.DIRECTIONAL_VALVE,
                    sensors=["pos"],
                    feature_dim=6,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                )
            },
            edges=[
                EdgeSpec(
                    source_id="pump",
                    target_id="valve",
                    edge_type=EdgeType.HYDRAULIC_LINE,
                    diameter_mm=10.0,
                    length_m=1.0,
                    pressure_rating_bar=200
                )
            ]
        )
        
        node_mapping, edge_list = topology.to_pyg_format()
        
        assert "pump" in node_mapping
        assert "valve" in node_mapping
        assert node_mapping["pump"] == 0
        assert node_mapping["valve"] == 1
        assert edge_list == [(0, 1)]
    
    def test_graph_topology_distributions(self):
        """Распределение типов."""
        topology = GraphTopology(
            equipment_id="exc_001",
            components={
                "pump1": ComponentSpec(
                    component_id="pump1",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["p"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "pump2": ComponentSpec(
                    component_id="pump2",
                    component_type=ComponentType.PISTON_PUMP,
                    sensors=["p"],
                    feature_dim=8,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                ),
                "valve": ComponentSpec(
                    component_id="valve",
                    component_type=ComponentType.DIRECTIONAL_VALVE,
                    sensors=["pos"],
                    feature_dim=6,
                    nominal_pressure_bar=280,
                    nominal_flow_lpm=120
                )
            },
            edges=[
                EdgeSpec(
                    source_id="pump1",
                    target_id="valve",
                    edge_type=EdgeType.HIGH_PRESSURE_HOSE,
                    diameter_mm=16.0,
                    length_m=2.5,
                    pressure_rating_bar=350
                ),
                EdgeSpec(
                    source_id="pump2",
                    target_id="valve",
                    edge_type=EdgeType.HIGH_PRESSURE_HOSE,
                    diameter_mm=16.0,
                    length_m=2.5,
                    pressure_rating_bar=350
                )
            ]
        )
        
        comp_dist = topology.get_component_types_distribution()
        assert comp_dist[ComponentType.PISTON_PUMP] == 2
        assert comp_dist[ComponentType.DIRECTIONAL_VALVE] == 1
        
        edge_dist = topology.get_edge_types_distribution()
        assert edge_dist[EdgeType.HIGH_PRESSURE_HOSE] == 2


# ==================== METADATA SCHEMAS TESTS ====================

class TestSensorConfig:
    """Tests для SensorConfig."""
    
    def test_sensor_config_creation_valid(self):
        """Создание valid SensorConfig."""
        sensor = SensorConfig(
            sensor_id="pressure_pump_out",
            sensor_type=SensorType.PRESSURE,
            component_id="pump_001",
            unit="bar",
            sampling_rate_hz=100.0,
            accuracy_percent=0.5,
            range_min=0.0,
            range_max=400.0
        )
        
        assert sensor.sensor_id == "pressure_pump_out"
        assert sensor.sensor_type == SensorType.PRESSURE
        assert sensor.measurement_range == 400.0
        assert sensor.absolute_accuracy == 2.0  # 0.5% of 400
    
    def test_sensor_config_invalid_range(self):
        """Некорректный диапазон (max < min)."""
        with pytest.raises(ValidationError, match="must be greater than"):
            SensorConfig(
                sensor_id="pressure",
                sensor_type=SensorType.PRESSURE,
                component_id="pump_001",
                unit="bar",
                sampling_rate_hz=100.0,
                accuracy_percent=0.5,
                range_min=100.0,
                range_max=50.0  # Less than min!
            )


class TestEquipmentMetadata:
    """Tests для EquipmentMetadata."""
    
    def test_equipment_metadata_creation_valid(self):
        """Создание valid EquipmentMetadata."""
        metadata = EquipmentMetadata(
            equipment_id="excavator_001",
            equipment_type="hydraulic_excavator",
            manufacturer="Caterpillar",
            model="320D",
            serial_number="CAT0320D123456",
            manufacture_year=2022,
            installation_date=datetime(2022, 3, 15),
            operating_hours=3450.5,
            fluid_type="ISO VG 46",
            tank_capacity_liters=180.0,
            max_working_pressure_bar=350
        )
        
        assert metadata.equipment_id == "excavator_001"
        assert metadata.age_years > 3.0  # At least 3 years old
    
    def test_equipment_metadata_invalid_year(self):
        """Некорректный год выпуска."""
        with pytest.raises(ValidationError):
            EquipmentMetadata(
                equipment_id="excavator_001",
                equipment_type="hydraulic_excavator",
                manufacturer="Cat",
                model="320D",
                serial_number="123",
                manufacture_year=1800,  # Too old!
                installation_date=datetime(2022, 3, 15),
                operating_hours=100.0,
                fluid_type="ISO VG 46",
                tank_capacity_liters=180.0,
                max_working_pressure_bar=350
            )


class TestSystemConfig:
    """Tests для SystemConfig."""
    
    def test_system_config_defaults(self):
        """Дефолтные значения."""
        config = SystemConfig()
        
        # Inference defaults
        assert config.inference_batch_size == 32
        assert config.use_gpu is True
        assert config.use_compile is True
        
        # Training defaults
        assert config.training_epochs == 100
        assert config.use_float8_training is False  # Requires A100/H100
        
        # Thresholds
        assert config.health_threshold_warning == 0.7
        assert config.health_threshold_critical == 0.5
    
    def test_system_config_threshold_validation(self):
        """Критический порог должен быть < warning."""
        with pytest.raises(ValidationError, match="must be less than"):
            SystemConfig(
                health_threshold_warning=0.5,
                health_threshold_critical=0.7  # Greater than warning!
            )


# ==================== REQUEST SCHEMAS TESTS ====================

class TestTimeWindow:
    """Tests для TimeWindow."""
    
    def test_time_window_valid(self):
        """Создание valid TimeWindow."""
        window = TimeWindow(
            start_time=datetime(2025, 11, 1, 0, 0, 0),
            end_time=datetime(2025, 11, 21, 0, 0, 0)
        )
        
        assert window.duration_hours == 480.0  # 20 days * 24 hours
        assert window.duration_seconds == 480.0 * 3600
    
    def test_time_window_end_before_start(self):
        """Конец до начала → ошибка."""
        with pytest.raises(ValidationError, match="must be after"):
            TimeWindow(
                start_time=datetime(2025, 11, 21),
                end_time=datetime(2025, 11, 1)  # Before start!
            )
    
    def test_time_window_exceeds_max_duration(self):
        """Превышение максимальной длительности (30 days)."""
        with pytest.raises(ValidationError, match="cannot exceed 30 days"):
            TimeWindow(
                start_time=datetime(2025, 1, 1),
                end_time=datetime(2025, 12, 31)  # 364 days!
            )


class TestInferenceRequest:
    """Tests для InferenceRequest."""
    
    def test_inference_request_valid(self):
        """Создание valid InferenceRequest."""
        request = InferenceRequest(
            equipment_id="excavator_001",
            time_window=TimeWindow(
                start_time=datetime(2025, 11, 1),
                end_time=datetime(2025, 11, 21)
            )
        )
        
        assert request.equipment_id == "excavator_001"
        assert request.include_attention_weights is False  # default
        assert request.include_recommendations is True  # default
        assert request.confidence_threshold == 0.7  # default


class TestBatchInferenceRequest:
    """Tests для BatchInferenceRequest."""
    
    def test_batch_inference_request_valid(self):
        """Создание valid batch request."""
        batch = BatchInferenceRequest(
            requests=[
                InferenceRequest(
                    equipment_id="exc_001",
                    time_window=TimeWindow(
                        start_time=datetime(2025, 11, 1),
                        end_time=datetime(2025, 11, 21)
                    )
                ),
                InferenceRequest(
                    equipment_id="exc_002",
                    time_window=TimeWindow(
                        start_time=datetime(2025, 11, 1),
                        end_time=datetime(2025, 11, 21)
                    )
                )
            ]
        )
        
        assert len(batch.requests) == 2
        assert batch.priority == "normal"  # default
    
    def test_batch_inference_request_duplicate_equipment_ids(self):
        """Дубликаты equipment_id в batch."""
        with pytest.raises(ValidationError, match="must be unique"):
            BatchInferenceRequest(
                requests=[
                    InferenceRequest(
                        equipment_id="exc_001",
                        time_window=TimeWindow(
                            start_time=datetime(2025, 11, 1),
                            end_time=datetime(2025, 11, 21)
                        )
                    ),
                    InferenceRequest(
                        equipment_id="exc_001",  # Duplicate!
                        time_window=TimeWindow(
                            start_time=datetime(2025, 11, 1),
                            end_time=datetime(2025, 11, 21)
                        )
                    )
                ]
            )


# ==================== RESPONSE SCHEMAS TESTS ====================

class TestComponentHealth:
    """Tests для ComponentHealth."""
    
    def test_component_health_creation_valid(self):
        """Создание valid ComponentHealth."""
        health = ComponentHealth(
            component_id="pump_001",
            component_type="piston_pump",
            health_score=0.87,
            degradation_rate=0.02,
            confidence=0.94,
            status=ComponentStatus.HEALTHY
        )
        
        assert health.component_id == "pump_001"
        assert health.health_score == 0.87
        assert health.requires_immediate_action is False  # healthy
    
    def test_component_health_critical_requires_action(self):
        """Критический статус требует действий."""
        health = ComponentHealth(
            component_id="valve_001",
            component_type="directional_valve",
            health_score=0.3,
            degradation_rate=0.15,
            confidence=0.91,
            status=ComponentStatus.CRITICAL
        )
        
        assert health.requires_immediate_action is True
    
    def test_component_health_low_ttf_requires_action(self):
        """Низкий TTF (< 24h) требует действий."""
        health = ComponentHealth(
            component_id="pump_001",
            component_type="piston_pump",
            health_score=0.65,
            degradation_rate=0.08,
            confidence=0.88,
            status=ComponentStatus.WARNING,
            time_to_failure_hours=12.0  # Less than 24!
        )
        
        assert health.requires_immediate_action is True


class TestAnomaly:
    """Tests для Anomaly."""
    
    def test_anomaly_creation_valid(self):
        """Создание valid Anomaly."""
        anomaly = Anomaly(
            anomaly_id="anom_001",
            anomaly_type=AnomalyType.PRESSURE_DROP,
            severity="medium",
            confidence=0.89,
            affected_components=["valve_001"],
            description="Unusual pressure fluctuation detected"
        )
        
        assert anomaly.anomaly_type == AnomalyType.PRESSURE_DROP
        assert anomaly.severity == "medium"
        assert len(anomaly.affected_components) == 1


class TestInferenceResponse:
    """Tests для InferenceResponse."""
    
    def test_inference_response_creation_valid(self):
        """Создание valid InferenceResponse."""
        response = InferenceResponse(
            request_id="req_123",
            equipment_id="excavator_001",
            overall_health_score=0.85,
            component_health=[
                ComponentHealth(
                    component_id="pump_001",
                    component_type="piston_pump",
                    health_score=0.92,
                    degradation_rate=0.02,
                    confidence=0.95,
                    status=ComponentStatus.HEALTHY
                )
            ]
        )
        
        assert response.overall_health_score == 0.85
        assert len(response.component_health) == 1
        assert response.has_critical_issues is False
        assert response.num_warnings == 0
    
    def test_inference_response_with_critical_issues(self):
        """Ответ с critical компонентами."""
        response = InferenceResponse(
            request_id="req_123",
            equipment_id="excavator_001",
            overall_health_score=0.45,
            component_health=[
                ComponentHealth(
                    component_id="valve_001",
                    component_type="directional_valve",
                    health_score=0.3,
                    degradation_rate=0.15,
                    confidence=0.91,
                    status=ComponentStatus.CRITICAL
                )
            ],
            anomalies=[
                Anomaly(
                    anomaly_id="anom_001",
                    anomaly_type=AnomalyType.PRESSURE_DROP,
                    severity="critical",
                    confidence=0.95,
                    affected_components=["valve_001"],
                    description="Critical pressure drop"
                )
            ]
        )
        
        assert response.has_critical_issues is True
        critical = response.get_critical_components()
        assert len(critical) == 1
        
        high_sev_anomalies = response.get_high_severity_anomalies()
        assert len(high_sev_anomalies) == 1


class TestTrainingResponse:
    """Tests для TrainingResponse."""
    
    def test_training_response_completed(self):
        """Успешное завершение training."""
        response = TrainingResponse(
            training_id="train_001",
            model_name="gnn_v2",
            status="completed",
            epochs_completed=100,
            best_validation_loss=0.0234,
            best_epoch=87,
            checkpoint_path="models/checkpoints/gnn_v2_epoch87.ckpt",
            training_time_seconds=14325.6,
            final_metrics={
                "health_mae": 0.045,
                "degradation_mae": 0.038,
                "anomaly_f1": 0.89
            }
        )
        
        assert response.status == "completed"
        assert response.epochs_completed == 100
        assert pytest.approx(response.training_time_hours, rel=0.01) == 3.98
        assert "health_mae" in response.final_metrics


# ==================== FIXTURES ====================

@pytest.fixture
def sample_edge_spec():
    """Sample EdgeSpec fixture."""
    return EdgeSpec(
        source_id="pump_001",
        target_id="valve_001",
        edge_type=EdgeType.HIGH_PRESSURE_HOSE,
        diameter_mm=16.0,
        length_m=2.5,
        pressure_rating_bar=350
    )


@pytest.fixture
def sample_component_spec():
    """Sample ComponentSpec fixture."""
    return ComponentSpec(
        component_id="pump_001",
        component_type=ComponentType.PISTON_PUMP,
        sensors=["pressure_in", "pressure_out", "temperature"],
        feature_dim=12,
        nominal_pressure_bar=280,
        nominal_flow_lpm=120,
        rated_power_kw=45.0
    )


@pytest.fixture
def sample_graph_topology(sample_component_spec):
    """Sample GraphTopology fixture."""
    valve = ComponentSpec(
        component_id="valve_001",
        component_type=ComponentType.DIRECTIONAL_VALVE,
        sensors=["position"],
        feature_dim=6,
        nominal_pressure_bar=280,
        nominal_flow_lpm=120
    )
    
    edge = EdgeSpec(
        source_id="pump_001",
        target_id="valve_001",
        edge_type=EdgeType.HIGH_PRESSURE_HOSE,
        diameter_mm=16.0,
        length_m=2.5,
        pressure_rating_bar=350
    )
    
    return GraphTopology(
        equipment_id="excavator_001",
        components={
            "pump_001": sample_component_spec,
            "valve_001": valve
        },
        edges=[edge]
    )


@pytest.fixture
def sample_time_window():
    """Sample TimeWindow fixture."""
    return TimeWindow(
        start_time=datetime(2025, 11, 1, 0, 0, 0),
        end_time=datetime(2025, 11, 21, 0, 0, 0)
    )


@pytest.fixture
def sample_inference_request(sample_time_window):
    """Sample InferenceRequest fixture."""
    return InferenceRequest(
        equipment_id="excavator_001",
        time_window=sample_time_window,
        include_recommendations=True
    )
