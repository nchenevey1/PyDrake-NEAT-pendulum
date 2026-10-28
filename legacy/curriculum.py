import numpy as np
from typing import Dict, List, Tuple
from config import Config

class CurriculumManager:
    """
    Manages the evolution of simulation physics parameters based on agent performance
    
    Implements a two phase curriculum:
    1. Gravity Ramp: Increases gravity from G_MIN to G_MAX
    2. Drag Reduction: Decreases drag from DRAG_MAX to DRAG_MIN
    """

    def __init__(self, difficulty: float = 0.0):
        """
        Args:
            difficulty: Initial difficulty level
        """
        self._difficulty = np.clip(difficulty, 0.0, 1.0)

    @property
    def difficulty(self) -> float:
        """Current difficulty index"""
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float):
        self._difficulty = np.clip(value, 0.0, 1.0)

    def get_params(self) -> Dict[str, float]:
        """
        Calculates physics parameters for the current difficulty level
        
        Returns:
            Dictionary containing 'g', 'drag', 'max_force', and 'difficulty'
        """
        # Phase boundaries
        PHASE_SPLIT = 0.5
        
        # --- Phase 1: Gravity ---
        if self._difficulty <= PHASE_SPLIT:
            # Normalize progress within phase 
            phase_progress = self._difficulty / PHASE_SPLIT
            
            # Quadratic ramp for gravity; hold max drag
            g = self._interp_quadratic(Config.G_MIN, Config.G_MAX, phase_progress)
            d = Config.DRAG_MAX
            max_f = Config.FORCE_MAX_START

        # --- Phase 2: Drag ---
        else:
            # Normalize progress within phase [0, 1]
            phase_progress = (self._difficulty - PHASE_SPLIT) / (1.0 - PHASE_SPLIT)
            
            # Hold max gravity; linear drop for drag
            g = Config.G_MAX
            d = self._interp_linear(Config.DRAG_MAX, Config.DRAG_MIN, phase_progress)
            max_f = self._interp_linear(Config.FORCE_MAX_START, Config.FORCE_MAX_END, phase_progress)

        return {
            'g': float(g),
            'drag': float(d),
            'max_force': float(max_f),
            'difficulty': float(self._difficulty)
        }

    def update(self, worst_elite_performance: float) -> str:
        """
        Evaluates population performance and increments difficulty if criteria met

        Args:
            upright_ratios: List of normalized uptime scores [0.0, 1.0] for the population

        Returns:
            Status string for logging (e.g., "UP >>", "MAX  ", "     ")
        """
        if self._difficulty >= 1.0:
            return "MAX  "
        
        # Advance if the N-th best performance meets criteria
        if worst_elite_performance >= Config.PASS_TIME_RATIO:
            self._difficulty += Config.DIFFICULTY_STEP
            self._difficulty = np.clip(self._difficulty, 0.0, 1.0)
            return "UP"
        
        return "  "

    # =========================================================================
    # Math Helpers
    # =========================================================================

    @staticmethod
    def _interp_linear(start: float, end: float, t: float) -> float:
        """Linear Interpolation: start + (end - start) * t"""
        return start + (end - start) * t

    @staticmethod
    def _interp_quadratic(start: float, end: float, t: float) -> float:
        """Quadratic Interpolation: start + (end - start) * t^2"""
        return start + (end - start) * (t ** 2)