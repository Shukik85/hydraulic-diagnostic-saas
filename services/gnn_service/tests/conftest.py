import asyncio
from typing import AsyncGenerator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """Create an event loop for pytest-asyncio (session scoped)."""
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()
