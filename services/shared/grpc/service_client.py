# services/shared/grpc/service_client.py
"""
Secure gRPC client для inter-service communication с mTLS.
"""
import grpc
import asyncio
from typing import Optional, Callable, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SecureServiceClient:
    """
    mTLS-enabled gRPC client для zero-trust communication.
    
    Istio автоматически инъектит certificates в /var/run/secrets/tls/
    """
    
    def __init__(
        self,
        service_name: str,
        namespace: str = "hydraulic-prod",
        cert_path: str = "/var/run/secrets/tls",
        port: int = 50051
    ):
        self.service_name = service_name
        self.namespace = namespace
        self.cert_path = Path(cert_path)
        self.port = port
        
        # Service mesh DNS: service.namespace.svc.cluster.local:port
        self.endpoint = f"{service_name}.{namespace}.svc.cluster.local:{port}"
        logger.info(f"Initialized gRPC client for {self.endpoint}")
    
    def create_channel(self) -> grpc.Channel:
        """
        Create mTLS-secured gRPC channel.
        
        Returns:
            grpc.Channel: Secure channel with mTLS
        """
        try:
            # Load certificates issued by Istio
            with open(self.cert_path / "tls.crt", 'rb') as f:
                client_cert = f.read()
            with open(self.cert_path / "tls.key", 'rb') as f:
                client_key = f.read()
            with open(self.cert_path / "ca.crt", 'rb') as f:
                ca_cert = f.read()
            
            # Create SSL credentials
            credentials = grpc.ssl_channel_credentials(
                root_certificates=ca_cert,
                private_key=client_key,
                certificate_chain=client_cert
            )
            
            # Channel options for production
            options = [
                ('grpc.keepalive_time_ms', 30000),
                ('grpc.keepalive_timeout_ms', 10000),
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.keepalive_permit_without_calls', 1),
                ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                ('grpc.max_receive_message_length', 100 * 1024 * 1024)
            ]
            
            # Create secure channel
            channel = grpc.secure_channel(
                self.endpoint,
                credentials,
                options=options
            )
            
            logger.info(f"Created secure channel to {self.endpoint}")
            return channel
            
        except FileNotFoundError as e:
            logger.error(f"Certificate files not found: {e}")
            logger.warning("Falling back to insecure channel (development only)")
            return grpc.insecure_channel(self.endpoint)
    
    async def call_with_retry(
        self,
        stub_method: Callable,
        request: Any,
        max_retries: int = 3,
        timeout: float = 10.0
    ) -> Any:
        """
        Call gRPC method with exponential backoff retry.
        
        Args:
            stub_method: gRPC stub method to call
            request: Request message
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            Response from gRPC service
            
        Raises:
            grpc.RpcError: If all retries fail
        """
        from grpc import StatusCode
        
        for attempt in range(max_retries):
            try:
                channel = self.create_channel()
                
                # Call stub method
                response = await stub_method(
                    request,
                    timeout=timeout,
                    metadata=self._get_metadata()
                )
                
                channel.close()
                logger.info(f"gRPC call to {self.service_name} succeeded")
                return response
                
            except grpc.RpcError as e:
                logger.error(
                    f"gRPC error (attempt {attempt + 1}/{max_retries}): "
                    f"{e.code()} - {e.details()}"
                )
                
                # Retry only on specific errors
                if e.code() in [
                    StatusCode.UNAVAILABLE,
                    StatusCode.DEADLINE_EXCEEDED,
                    StatusCode.RESOURCE_EXHAUSTED
                ]:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 1s, 2s, 4s
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Don't retry on client errors
                raise
        
        raise Exception(f"Failed to call {self.service_name} after {max_retries} attempts")
    
    def _get_metadata(self) -> list:
        """
        Generate gRPC metadata for request.
        
        Returns:
            list: gRPC metadata tuples
        """
        import uuid
        
        return [
            ('x-request-id', str(uuid.uuid4())),
            ('x-client-service', 'unknown'),  # TODO: Get from env
        ]


class ServiceDiscovery:
    """
    Service discovery helper для Kubernetes.
    """
    
    SERVICES = {
        'auth': {'name': 'auth-service', 'port': 8001},
        'equipment': {'name': 'equipment-service', 'port': 8002},
        'diagnosis': {'name': 'diagnosis-service', 'port': 8003},
        'gnn': {'name': 'gnn-service', 'port': 50051},
    }
    
    @classmethod
    def get_endpoint(cls, service_key: str, namespace: str = "hydraulic-prod") -> str:
        """
        Get service endpoint URL.
        
        Args:
            service_key: Service identifier ('auth', 'equipment', etc.)
            namespace: Kubernetes namespace
            
        Returns:
            str: Service endpoint
        """
        service = cls.SERVICES.get(service_key)
        if not service:
            raise ValueError(f"Unknown service: {service_key}")
        
        return f"{service['name']}.{namespace}.svc.cluster.local:{service['port']}"
    
    @classmethod
    def create_client(cls, service_key: str, namespace: str = "hydraulic-prod") -> SecureServiceClient:
        """
        Create gRPC client for service.
        
        Args:
            service_key: Service identifier
            namespace: Kubernetes namespace
            
        Returns:
            SecureServiceClient: Configured gRPC client
        """
        service = cls.SERVICES.get(service_key)
        if not service:
            raise ValueError(f"Unknown service: {service_key}")
        
        return SecureServiceClient(
            service_name=service['name'],
            namespace=namespace,
            port=service['port']
        )
