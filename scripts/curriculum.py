import os
import sys
from dataclasses import dataclass
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# Allow importing random_track_generator from the sibling directory
_GENERATOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'random-track-generator')
if _GENERATOR_PATH not in sys.path:
    sys.path.insert(0, _GENERATOR_PATH)

from random_track_generator import generate_track
from random_track_generator import SimType


@dataclass
class DifficultyConfig:
    n_points: int
    n_regions: int
    max_bound: float
    mode: str   # "expand" or "extend"
    label: str  # human-readable name


DIFFICULTY_LEVELS = [
    DifficultyConfig(15,  4,  30.0, "expand", "novice"),
    DifficultyConfig(20,  5,  45.0, "expand", "easy"),
    DifficultyConfig(28,  7,  60.0, "expand", "medium"),
    DifficultyConfig(38, 10,  80.0, "extend", "hard"),
    DifficultyConfig(55, 14, 100.0, "extend", "expert"),
]


def generate_track_for_difficulty(cfg: DifficultyConfig):
    """Generate a random Track object for the given difficulty config."""
    return generate_track(
        n_points=cfg.n_points,
        n_regions=cfg.n_regions,
        min_bound=0.0,
        max_bound=cfg.max_bound,
        mode=cfg.mode,
        seed=None,
    )


class CurriculumCallback(BaseCallback):
    """
    Monitors average laps completed per episode across all envs.
    When the rolling average >= lap_threshold over the last `window` episodes,
    advances the difficulty level on all training envs.

    Args:
        train_env: The VecEnv used for training (before VecNormalize wrappers).
        window: Number of recent episodes used to compute the rolling average.
        lap_threshold: Average laps per episode required to advance.
        verbose: Verbosity level.
    """

    def __init__(self, train_env, window: int = 20, lap_threshold: float = 1.0, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.window = window
        self.lap_threshold = lap_threshold
        self.episode_laps: deque = deque(maxlen=window)
        self.current_level = 0
        self.max_level = len(DIFFICULTY_LEVELS) - 1

    def _on_step(self) -> bool:
        # Collect lap counts from completed episodes
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episode_laps.append(info.get("lap_count", 0))

        # Check whether to advance difficulty
        if (
            len(self.episode_laps) >= self.window
            and np.mean(self.episode_laps) >= self.lap_threshold
            and self.current_level < self.max_level
        ):
            self.current_level += 1
            if self.verbose:
                avg = np.mean(self.episode_laps)
                print(
                    f"\n[Curriculum] Advancing to difficulty level {self.current_level} "
                    f"({DIFFICULTY_LEVELS[self.current_level].label}) — "
                    f"avg laps over last {self.window} episodes: {avg:.2f}"
                )
            self.episode_laps.clear()

            # Signal all training envs to regenerate their track on next reset
            self.train_env.set_attr("difficulty_level", self.current_level)
            self.train_env.set_attr("_pending_track_reload", True)

        return True
