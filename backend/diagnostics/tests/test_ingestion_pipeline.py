"""
Comprehensive tests for sensor data ingestion pipeline.

Covers:
- API endpoints (POST /data/ingest, GET /jobs/{job_id})
- Celery tasks
- Validation logic
- Quarantine workflow
- TimescaleDB integration
"""

import pytest
from datetime import timedelta
from uuid import uuid4
from django.utils import timezone
from django.urls import reverse
from rest_framework.test import APIClient
from rest_framework import status

from diagnostics.models import SensorData, HydraulicSystem
from diagnostics.models_ingestion import IngestionJob
from diagnostics.models_quarantine import QuarantinedReading
from diagnostics.tasks_ingest import validate_reading, chunked_bulk_create
from users.models import User


@pytest.fixture
def api_client():
    """API client with JWT authentication."""
    return APIClient()


@pytest.fixture
def authenticated_user(db):
    """Create test user."""
    user = User.objects.create_user(
        username='testuser',
        email='test@example.com',
        password='testpass123'
    )
    return user


@pytest.fixture
def test_system(db, authenticated_user):
    """Create test hydraulic system."""
    return HydraulicSystem.objects.create(
        name='Test System',
        system_type='industrial',
        status='active',
        owner=authenticated_user,
    )


@pytest.fixture
def valid_reading():
    """Valid sensor reading data."""
    return {
        'sensor_id': str(uuid4()),
        'timestamp': timezone.now().isoformat(),
        'value': 125.5,
        'unit': 'bar',
        'quality': 95,
    }


@pytest.fixture
def invalid_reading_out_of_range():
    """Invalid reading - value out of range."""
    return {
        'sensor_id': str(uuid4()),
        'timestamp': timezone.now().isoformat(),
        'value': 800.0,  # Exceeds max 700 for bar
        'unit': 'bar',
        'quality': 95,
    }


@pytest.fixture
def invalid_reading_future_timestamp():
    """Invalid reading - timestamp in future."""
    return {
        'sensor_id': str(uuid4()),
        'timestamp': (timezone.now() + timedelta(hours=1)).isoformat(),
        'value': 125.5,
        'unit': 'bar',
        'quality': 95,
    }


class TestValidationLogic:
    """Test sensor reading validation."""
    
    def test_validate_valid_reading(self, test_system, valid_reading):
        """Test validation passes for valid reading."""
        # Convert ISO string to datetime
        valid_reading['timestamp'] = timezone.now()
        
        is_valid, reason, details = validate_reading(valid_reading, test_system)
        
        assert is_valid is True
        assert reason == ''
        assert details == ''
    
    def test_validate_out_of_range(self, test_system, invalid_reading_out_of_range):
        """Test validation fails for out of range value."""
        invalid_reading_out_of_range['timestamp'] = timezone.now()
        
        is_valid, reason, details = validate_reading(invalid_reading_out_of_range, test_system)
        
        assert is_valid is False
        assert reason == 'out_of_range'
        assert '800' in details
        assert '700' in details
    
    def test_validate_future_timestamp(self, test_system, invalid_reading_future_timestamp):
        """Test validation fails for future timestamp."""
        invalid_reading_future_timestamp['timestamp'] = timezone.now() + timedelta(hours=1)
        
        is_valid, reason, details = validate_reading(invalid_reading_future_timestamp, test_system)
        
        assert is_valid is False
        assert reason == 'invalid_timestamp'
        assert 'future' in details.lower()
    
    def test_validate_low_quality(self, test_system, valid_reading):
        """Test validation fails for low quality score."""
        valid_reading['timestamp'] = timezone.now()
        valid_reading['quality'] = 30  # Below threshold (50)
        
        is_valid, reason, details = validate_reading(valid_reading, test_system)
        
        assert is_valid is False
        assert reason == 'out_of_range'
        assert 'quality' in details.lower()


class TestChunkedBulkCreate:
    """Test performance optimization helper."""
    
    def test_chunked_bulk_create_single_chunk(self, test_system):
        """Test bulk create with single chunk."""
        readings = [
            SensorData(
                system=test_system,
                timestamp=timezone.now(),
                sensor_type='pressure',
                value=100.0 + i,
                unit='bar',
            )
            for i in range(500)
        ]
        
        created_count = chunked_bulk_create(SensorData, readings, batch_size=1000)
        
        assert created_count == 500
        assert SensorData.objects.count() == 500
    
    def test_chunked_bulk_create_multiple_chunks(self, test_system):
        """Test bulk create with multiple chunks."""
        readings = [
            SensorData(
                system=test_system,
                timestamp=timezone.now(),
                sensor_type='pressure',
                value=100.0 + i,
                unit='bar',
            )
            for i in range(2500)
        ]
        
        created_count = chunked_bulk_create(SensorData, readings, batch_size=1000)
        
        assert created_count == 2500
        assert SensorData.objects.count() == 2500


class TestIngestionModels:
    """Test IngestionJob and QuarantinedReading models."""
    
    def test_ingestion_job_creation(self, test_system, authenticated_user):
        """Test IngestionJob model creation."""
        job = IngestionJob.objects.create(
            system_id=test_system.id,
            status='queued',
            total_readings=100,
            created_by=authenticated_user,
        )
        
        assert job.id is not None
        assert job.status == 'queued'
        assert job.total_readings == 100
        assert job.inserted_readings == 0
        assert job.quarantined_readings == 0
        assert job.success_rate == 0.0
        assert job.is_active is True
        assert job.is_completed is False
    
    def test_ingestion_job_success_rate(self, test_system, authenticated_user):
        """Test success_rate property calculation."""
        job = IngestionJob.objects.create(
            system_id=test_system.id,
            status='completed',
            total_readings=1000,
            inserted_readings=987,
            quarantined_readings=13,
            created_by=authenticated_user,
        )
        
        assert job.success_rate == 98.7
    
    def test_quarantined_reading_creation(self, test_system):
        """Test QuarantinedReading model creation."""
        job_id = uuid4()
        reading = QuarantinedReading.objects.create(
            job_id=job_id,
            sensor_id=uuid4(),
            timestamp=timezone.now(),
            value=800.0,
            unit='bar',
            quality=95,
            system_id=test_system.id,
            reason='out_of_range',
            reason_details='Value 800 > max 700',
        )
        
        assert reading.id is not None
        assert reading.reason == 'out_of_range'
        assert reading.review_status == 'pending'


class TestBulkIngestAPI:
    """Test POST /api/v1/data/ingest endpoint."""
    
    def test_bulk_ingest_success(self, api_client, authenticated_user, test_system, valid_reading):
        """Test successful bulk ingestion."""
        # Authenticate
        api_client.force_authenticate(user=authenticated_user)
        
        # Prepare request
        url = reverse('api-sensor-bulk-ingest')
        data = {
            'system_id': str(test_system.id),
            'readings': [valid_reading],
        }
        
        # Make request
        response = api_client.post(url, data, format='json')
        
        # Assertions
        assert response.status_code == status.HTTP_202_ACCEPTED
        assert 'job_id' in response.data
        assert response.data['status'] in ['queued', 'processing']
        
        # Verify job created
        job_id = response.data['job_id']
        job = IngestionJob.objects.get(id=job_id)
        assert job.system_id == test_system.id
    
    def test_bulk_ingest_authentication_required(self, api_client, test_system, valid_reading):
        """Test authentication is required."""
        url = reverse('api-sensor-bulk-ingest')
        data = {
            'system_id': str(test_system.id),
            'readings': [valid_reading],
        }
        
        response = api_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_bulk_ingest_validation_error(self, api_client, authenticated_user, test_system):
        """Test validation error response."""
        api_client.force_authenticate(user=authenticated_user)
        
        url = reverse('api-sensor-bulk-ingest')
        data = {
            'system_id': 'invalid-uuid',  # Invalid UUID
            'readings': [],
        }
        
        response = api_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_bulk_ingest_empty_readings(self, api_client, authenticated_user, test_system):
        """Test empty readings array rejected."""
        api_client.force_authenticate(user=authenticated_user)
        
        url = reverse('api-sensor-bulk-ingest')
        data = {
            'system_id': str(test_system.id),
            'readings': [],  # Empty array
        }
        
        response = api_client.post(url, data, format='json')
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestJobStatusAPI:
    """Test GET /api/v1/jobs/{job_id}/ endpoint."""
    
    def test_get_job_status_success(self, api_client, authenticated_user, test_system):
        """Test successful job status retrieval."""
        # Create job
        job = IngestionJob.objects.create(
            system_id=test_system.id,
            status='completed',
            total_readings=100,
            inserted_readings=95,
            quarantined_readings=5,
            processing_time_ms=1234,
            created_by=authenticated_user,
        )
        
        # Authenticate and request
        api_client.force_authenticate(user=authenticated_user)
        url = reverse('api-job-status', kwargs={'job_id': str(job.id)})
        response = api_client.get(url)
        
        # Assertions
        assert response.status_code == status.HTTP_200_OK
        assert response.data['job_id'] == str(job.id)
        assert response.data['status'] == 'completed'
        assert response.data['total_readings'] == 100
        assert response.data['inserted_readings'] == 95
        assert response.data['quarantined_readings'] == 5
        assert response.data['success_rate'] == 95.0
    
    def test_get_job_status_not_found(self, api_client, authenticated_user):
        """Test 404 for non-existent job."""
        api_client.force_authenticate(user=authenticated_user)
        
        url = reverse('api-job-status', kwargs={'job_id': str(uuid4())})
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_get_job_status_invalid_uuid(self, api_client, authenticated_user):
        """Test 400 for invalid UUID."""
        api_client.force_authenticate(user=authenticated_user)
        
        url = '/api/v1/jobs/invalid-uuid/'
        response = api_client.get(url)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestCeleryIngestionTask:
    """Test Celery ingestion task."""
    
    @pytest.mark.django_db
    def test_ingest_task_all_valid(self, test_system, authenticated_user):
        """Test ingestion with all valid readings."""
        from diagnostics.tasks_ingest import ingest_sensor_data_bulk
        
        job_id = str(uuid4())
        readings = [
            {
                'sensor_id': str(uuid4()),
                'timestamp': timezone.now() - timedelta(seconds=i),
                'value': 100.0 + i,
                'unit': 'bar',
                'quality': 95,
            }
            for i in range(100)
        ]
        
        # Create job
        job = IngestionJob.objects.create(
            id=job_id,
            system_id=test_system.id,
            total_readings=len(readings),
            created_by=authenticated_user,
        )
        
        # Run task
        result = ingest_sensor_data_bulk(
            system_id=str(test_system.id),
            readings=readings,
            job_id=job_id,
        )
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['inserted'] == 100
        assert result['quarantined'] == 0
        
        # Verify database
        job.refresh_from_db()
        assert job.status == 'completed'
        assert job.inserted_readings == 100
        assert job.quarantined_readings == 0
        assert SensorData.objects.count() == 100
    
    @pytest.mark.django_db
    def test_ingest_task_with_quarantine(self, test_system, authenticated_user):
        """Test ingestion with some invalid readings."""
        from diagnostics.tasks_ingest import ingest_sensor_data_bulk
        
        job_id = str(uuid4())
        readings = [
            # Valid readings
            {
                'sensor_id': str(uuid4()),
                'timestamp': timezone.now() - timedelta(seconds=i),
                'value': 100.0,
                'unit': 'bar',
                'quality': 95,
            }
            for i in range(50)
        ] + [
            # Invalid readings (out of range)
            {
                'sensor_id': str(uuid4()),
                'timestamp': timezone.now() - timedelta(seconds=i),
                'value': 800.0,  # Out of range
                'unit': 'bar',
                'quality': 95,
            }
            for i in range(50, 100)
        ]
        
        # Create job
        job = IngestionJob.objects.create(
            id=job_id,
            system_id=test_system.id,
            total_readings=len(readings),
            created_by=authenticated_user,
        )
        
        # Run task
        result = ingest_sensor_data_bulk(
            system_id=str(test_system.id),
            readings=readings,
            job_id=job_id,
        )
        
        # Assertions
        assert result['status'] == 'completed'
        assert result['inserted'] == 50
        assert result['quarantined'] == 50
        
        # Verify database
        job.refresh_from_db()
        assert job.inserted_readings == 50
        assert job.quarantined_readings == 50
        assert job.success_rate == 50.0
        assert SensorData.objects.count() == 50
        assert QuarantinedReading.objects.count() == 50
    
    @pytest.mark.django_db
    def test_ingest_task_system_not_found(self, authenticated_user):
        """Test ingestion with non-existent system."""
        from diagnostics.tasks_ingest import ingest_sensor_data_bulk
        
        job_id = str(uuid4())
        fake_system_id = str(uuid4())
        readings = [
            {
                'sensor_id': str(uuid4()),
                'timestamp': timezone.now().isoformat(),
                'value': 100.0,
                'unit': 'bar',
                'quality': 95,
            }
        ]
        
        # Create job
        job = IngestionJob.objects.create(
            id=job_id,
            system_id=fake_system_id,
            total_readings=len(readings),
            created_by=authenticated_user,
        )
        
        # Run task
        result = ingest_sensor_data_bulk(
            system_id=fake_system_id,
            readings=readings,
            job_id=job_id,
        )
        
        # Assertions
        assert result['status'] == 'failed'
        assert 'not found' in result['error'].lower()
        
        # Verify all readings quarantined
        job.refresh_from_db()
        assert job.status == 'failed'
        assert job.quarantined_readings == 1
        assert QuarantinedReading.objects.filter(reason='system_not_found').count() == 1


class TestE2EIngestionPipeline:
    """End-to-end ingestion pipeline tests."""
    
    @pytest.mark.django_db
    def test_e2e_successful_ingestion(self, api_client, authenticated_user, test_system):
        """Test complete E2E flow: API -> Celery -> Database."""
        # Authenticate
        api_client.force_authenticate(user=authenticated_user)
        
        # POST ingestion request
        url = reverse('api-sensor-bulk-ingest')
        data = {
            'system_id': str(test_system.id),
            'readings': [
                {
                    'sensor_id': str(uuid4()),
                    'timestamp': (timezone.now() - timedelta(seconds=i)).isoformat(),
                    'value': 100.0 + i,
                    'unit': 'bar',
                    'quality': 95,
                }
                for i in range(10)
            ],
        }
        
        response = api_client.post(url, data, format='json')
        
        # Verify API response
        assert response.status_code == status.HTTP_202_ACCEPTED
        job_id = response.data['job_id']
        
        # Verify job created
        job = IngestionJob.objects.get(id=job_id)
        assert job.total_readings == 10
        
        # Run Celery task synchronously (for testing)
        from diagnostics.tasks_ingest import ingest_sensor_data_bulk
        result = ingest_sensor_data_bulk(
            system_id=str(test_system.id),
            readings=data['readings'],
            job_id=job_id,
        )
        
        # Verify task result
        assert result['status'] == 'completed'
        assert result['inserted'] == 10
        
        # Verify database
        job.refresh_from_db()
        assert job.status == 'completed'
        assert job.inserted_readings == 10
        assert SensorData.objects.filter(system=test_system).count() == 10
        
        # GET job status
        status_url = reverse('api-job-status', kwargs={'job_id': job_id})
        status_response = api_client.get(status_url)
        
        # Verify status endpoint
        assert status_response.status_code == status.HTTP_200_OK
        assert status_response.data['status'] == 'completed'
        assert status_response.data['success_rate'] == 100.0


@pytest.mark.django_db
class TestPerformanceBenchmark:
    """Performance benchmark tests."""
    
    def test_bulk_insert_10k_rows(self, test_system):
        """Test >10K rows/second insertion target."""
        import time
        
        # Prepare 10K readings
        readings = [
            SensorData(
                system=test_system,
                timestamp=timezone.now() - timedelta(seconds=i),
                sensor_type='pressure',
                value=100.0 + (i % 100),
                unit='bar',
            )
            for i in range(10000)
        ]
        
        # Benchmark insertion
        start = time.time()
        chunked_bulk_create(SensorData, readings, batch_size=1000)
        elapsed = time.time() - start
        
        # Calculate throughput
        throughput = 10000 / elapsed
        
        # Assertions
        assert SensorData.objects.count() == 10000
        assert throughput > 10000, f"Throughput {throughput:.0f} rows/sec < 10K target"
        assert elapsed < 1.0, f"Elapsed {elapsed:.2f}s > 1s target"


__all__ = [
    'TestValidationLogic',
    'TestChunkedBulkCreate',
    'TestIngestionModels',
    'TestBulkIngestAPI',
    'TestJobStatusAPI',
    'TestE2EIngestionPipeline',
    'TestPerformanceBenchmark',
]
