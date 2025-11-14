"""
Demo hydraulic systems generator for testing and showcase.

Based on commit ea2cd5d45d232a088822b2742d22b998c2de45c8

Includes:
1. Excavator hydraulic system
2. Injection molding machine
3. CNC machine hydraulic system
4. Industrial robot manipulator

Each system includes:
- Realistic component topology
- Sensor configurations
- Normal and failure modes
- Synthetic data generation
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ... (demo systems logic here) ... Код из файла 59 (demo-systems.py)...