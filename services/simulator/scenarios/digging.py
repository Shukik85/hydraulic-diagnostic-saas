"""
Digging Scenario - Realistic excavator operation
"""


class DiggingScenario:
    """Simulates realistic digging operations"""
    
    def __init__(self, duration: float = 30.0):
        self.duration = duration  # seconds
        self.phase = "approach"
        self.phase_time = 0.0
    
    def get_controls(self, time: float) -> dict[str, float]:
        """
        Get control commands for digging cycle
        
        Phases:
        1. Approach (0-5s): Move boom down, extend stick
        2. Dig (5-10s): Boom up + stick curl + bucket curl (combined!)
        3. Lift (10-15s): Boom up to max
        4. Swing (15-20s): Rotate to dump position
        5. Dump (20-25s): Bucket dump
        6. Return (25-30s): Swing back, lower boom
        """
        t = time % self.duration
        
        boom_cmd = 0.0
        stick_cmd = 0.0
        bucket_cmd = 0.0
        swing_cmd = 0.0
        
        # Phase 1: Approach (0-5s)
        if 0 <= t < 5:
            self.phase = "approach"
            boom_cmd = -0.6  # Lower boom
            stick_cmd = 0.7  # Extend stick
            bucket_cmd = 0.0
            swing_cmd = 0.0
        
        # Phase 2: Dig (5-10s) - COMBINED OPERATIONS!
        elif 5 <= t < 10:
            self.phase = "dig"
            boom_cmd = 0.8  # Lift boom (main force)
            stick_cmd = -0.9  # Curl stick (tear action)
            bucket_cmd = 0.7  # Curl bucket (fill)
            swing_cmd = 0.0
        
        # Phase 3: Lift (10-15s)
        elif 10 <= t < 15:
            self.phase = "lift"
            boom_cmd = 0.9  # Max lift
            stick_cmd = -0.3  # Slight curl
            bucket_cmd = 0.2  # Hold bucket
            swing_cmd = 0.0
        
        # Phase 4: Swing to dump (15-20s)
        elif 15 <= t < 20:
            self.phase = "swing_to_dump"
            boom_cmd = 0.3  # Maintain height
            stick_cmd = 0.0
            bucket_cmd = 0.0
            swing_cmd = 0.8  # Swing left
        
        # Phase 5: Dump (20-25s)
        elif 20 <= t < 25:
            self.phase = "dump"
            boom_cmd = 0.0
            stick_cmd = 0.0
            bucket_cmd = -1.0  # Open bucket
            swing_cmd = 0.0
        
        # Phase 6: Return (25-30s)
        else:
            self.phase = "return"
            boom_cmd = -0.5  # Lower boom
            stick_cmd = 0.0
            bucket_cmd = 0.5  # Close bucket
            swing_cmd = -0.8  # Swing back
        
        return {
            'boom': boom_cmd,
            'stick': stick_cmd,
            'bucket': bucket_cmd,
            'swing': swing_cmd
        }
    
    def get_load_mass(self, time: float) -> float:
        """Get bucket load mass (kg)"""
        t = time % self.duration
        
        # Load increases during dig phase
        if 5 <= t < 10:
            # Gradual fill (0 → 1500 kg over 5 seconds)
            fill_progress = (t - 5) / 5
            return 1500 * fill_progress
        
        # Full load during lift and swing
        elif 10 <= t < 25:
            return 1500  # Full bucket (1.5 tons)
        
        # Empty after dump
        else:
            return 0
    
    def get_soil_resistance(self, time: float) -> str:
        """Get soil type for digging resistance"""
        t = time % self.duration
        
        if 5 <= t < 10:
            return "hard"  # Digging into soil
        else:
            return "soft"  # No resistance


class LoadingScenario:
    """Simulates loading a truck - repetitive cycles"""
    
    def __init__(self, cycles: int = 5):
        self.cycles = cycles
        self.cycle_duration = 25.0  # seconds per load
    
    def get_controls(self, time: float) -> dict[str, float]:
        """Faster cycles for loading operations"""
        t = time % self.cycle_duration
        
        # Similar to digging but faster
        if t < 3:  # Quick approach
            return {'boom': -0.8, 'stick': 0.9, 'bucket': 0.0, 'swing': 0.0}
        elif t < 6:  # Fast dig
            return {'boom': 1.0, 'stick': -1.0, 'bucket': 0.9, 'swing': 0.0}
        elif t < 10:  # Lift and swing simultaneously
            return {'boom': 0.9, 'stick': -0.2, 'bucket': 0.2, 'swing': 0.9}
        elif t < 14:  # Swing to truck
            return {'boom': 0.3, 'stick': 0.0, 'bucket': 0.0, 'swing': 1.0}
        elif t < 17:  # Dump
            return {'boom': 0.0, 'stick': 0.0, 'bucket': -1.0, 'swing': 0.0}
        else:  # Quick return
            return {'boom': -0.6, 'stick': 0.0, 'bucket': 0.7, 'swing': -1.0}
    
    def get_load_mass(self, time: float) -> float:
        """Load profile"""
        t = time % self.cycle_duration
        
        if 3 <= t < 6:
            return 1800 * ((t - 3) / 3)  # Heavy loading
        elif 6 <= t < 17:
            return 1800
        else:
            return 0


class EmergencyStopScenario:
    """Simulates emergency stop - sudden deceleration"""
    
    def __init__(self, stop_time: float = 5.0):
        self.stop_time = stop_time
    
    def get_controls(self, time: float) -> dict[str, float]:
        """Full speed → sudden stop"""
        if time < 3:
            # Full speed operation
            return {'boom': 0.9, 'stick': -0.8, 'bucket': 0.7, 'swing': 0.6}
        elif time < 3.5:
            # Emergency stop! (pressure surge expected)
            return {'boom': 0.0, 'stick': 0.0, 'bucket': 0.0, 'swing': 0.0}
        else:
            # Hold position
            return {'boom': 0.0, 'stick': 0.0, 'bucket': 0.0, 'swing': 0.0}
