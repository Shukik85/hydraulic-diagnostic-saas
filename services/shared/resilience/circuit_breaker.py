# services/shared/resilience/circuit_breaker.py
"""
Circuit breaker pattern для предотвращения cascade failures.
"""
import asyncio
from enum import Enum
from datetime import datetime, timedelta
from typing import Callable, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing recovery


class CircuitBreakerError(Exception):
    """Exception raised when circuit is open."""
    pass


class CircuitBreaker:
    """
    Enterprise circuit breaker с exponential backoff.
    
    Используется для защиты от cascade failures в microservices.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception,
        name: str = "circuit_breaker"
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
            expected_exception: Exception type to catch
            name: Circuit breaker name for logging
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
        
        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={timeout}s"
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original exception from function
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit '{self.name}' transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
            else:
                logger.warning(
                    f"Circuit '{self.name}' is OPEN, rejecting request. "
                    f"Last failure: {self.last_failure_time}"
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            logger.error(
                f"Circuit '{self.name}' caught exception: {type(e).__name__} - {e}"
            )
            raise
    
    def _on_success(self):
        """
        Handle successful call.
        """
        if self.failure_count > 0:
            logger.info(
                f"Circuit '{self.name}' recovered, resetting failure count "
                f"from {self.failure_count}"
            )
        
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit '{self.name}' transitioning to CLOSED")
            self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """
        Handle failed call.
        """
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        logger.warning(
            f"Circuit '{self.name}' failure count: {self.failure_count}/{self.failure_threshold}"
        )
        
        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit '{self.name}' threshold reached, transitioning to OPEN"
            )
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """
        Check if enough time has passed to attempt reset.
        
        Returns:
            bool: True if should attempt reset
        """
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.utcnow() - self.last_failure_time
        return time_since_failure > timedelta(seconds=self.timeout)
    
    def get_state(self) -> dict:
        """
        Get current circuit breaker state.
        
        Returns:
            dict: State information
        """
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'failure_threshold': self.failure_threshold,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class CircuitBreakerRegistry:
    """
    Global registry для circuit breakers.
    """
    
    _breakers: dict = {}
    
    @classmethod
    def get_or_create(
        cls,
        name: str,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ) -> CircuitBreaker:
        """
        Get existing or create new circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Failure threshold
            timeout: Reset timeout
            expected_exception: Exception to catch
            
        Returns:
            CircuitBreaker: Circuit breaker instance
        """
        if name not in cls._breakers:
            cls._breakers[name] = CircuitBreaker(
                failure_threshold=failure_threshold,
                timeout=timeout,
                expected_exception=expected_exception,
                name=name
            )
        return cls._breakers[name]
    
    @classmethod
    def get_all_states(cls) -> dict:
        """
        Get states of all circuit breakers.
        
        Returns:
            dict: All circuit breaker states
        """
        return {
            name: breaker.get_state()
            for name, breaker in cls._breakers.items()
        }
