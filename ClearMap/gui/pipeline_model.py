from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import yaml
from pathlib import Path


@dataclass(frozen=True)
class StepSpec:
    """Static definition of one binarization post-processing step."""
    name: str          # matches worker method suffix: binarize_channel, smooth_channel...
    label: str         # display label
    color: str = '#2d6a4f'
    description: str = ''


@dataclass
class PipelineStep:
    spec_name: str
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enabled: bool = True
    locked: bool = False    # cannot be moved or displaced
    keep_intermediate: bool = True  # keep output for inspection


@dataclass
class LinearPipeline:
    """
    Ordered, reorderable sequence of binarization steps.
    Locked steps cannot be moved and cannot be displaced by adjacent moves.
    """
    steps: list[PipelineStep] = field(default_factory=list)

    @property
    def enabled_steps(self) -> list[str]:
        """Ordered list of spec_names for enabled steps only."""
        return [s.spec_name for s in self.steps if s.enabled]

    def toggle(self, step_id: str) -> None:
        """Toggle the enabled state of a step. No-op for locked steps."""
        for s in self.steps:
            if s.step_id == step_id:
                if not s.locked:
                    s.enabled = not s.enabled
                return

    def move(self, step_id: str, direction: int) -> None:
        """
        Move a step up (-1) or down (+1) in the pipeline.

        Constraints
        -----------
        - Locked steps cannot move.
        - No step can displace a locked step.

        Parameters
        ----------
        step_id : str
            ID of the step to move.
        direction : int
            -1 to move up, +1 to move down.
        """
        idx = next((i for i, s in enumerate(self.steps)
                    if s.step_id == step_id), None)
        if idx is None:
            return
        if self.steps[idx].locked:
            return
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(self.steps):
            return
        if self.steps[new_idx].locked:
            return
        self.steps[idx], self.steps[new_idx] = (
            self.steps[new_idx], self.steps[idx])

    # # ---- serialization ----------------------------------------------------
    #
    # def to_dict(self) -> dict:
    #     return {'steps': [{'spec_name': s.spec_name,
    #                        'step_id': s.step_id,
    #                        'enabled': s.enabled,
    #                        'locked': s.locked,
    #                        'keep_intermediate': s.keep_intermediate}
    #                       for s in self.steps]}
    #
    # def save(self, path: Path) -> None:
    #     path.write_text(yaml.safe_dump(self.to_dict()))  # FIXME: move IO to ConfigHandler
    #
    # @classmethod
    # def from_dict(cls, d: dict) -> 'LinearPipeline':
    #     return cls(steps=[PipelineStep(**s) for s in d.get('steps', [])])
    #
    # @classmethod
    # def load(cls, path: Path) -> 'LinearPipeline':
    #     return cls.from_dict(yaml.safe_load(path.read_text()))  # FIXME: move IO to ConfigHandler


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

# Steps driven by run_* flags in VesselBinarizationParams.
# Insertion order = default execution order.
BINARIZATION_STEPS: dict[str, StepSpec] = {
    'binarize': StepSpec(
        name='binarize',
        label='Binarize',
        color='#1b4332',
        description='Threshold raw channel to produce a binary mask',
    ),
    'smooth': StepSpec(
        name='smooth',
        label='Smooth',
        color='#2d6a4f',
        description='Morphological smoothing of the binary mask',
    ),
    'binary_fill': StepSpec(
        name='binary_fill',
        label='Fill',
        color='#40916c',
        description='Fill holes in the binary mask',
    ),
    'deep_fill': StepSpec(
        name='deep_fill',
        label='Deep fill',
        color='#52b788',
        description='CNN-based vessel lumen filling (requires GPU)',
    ),
}


def default_binarization_pipeline() -> LinearPipeline:
    """
    Build the default pipeline from BINARIZATION_STEPS.
    The first step (binarize) is locked — it must always run first.
    """
    return LinearPipeline(steps=[
        PipelineStep(spec_name=name, locked=(i == 0))
        for i, name in enumerate(BINARIZATION_STEPS)
    ])
