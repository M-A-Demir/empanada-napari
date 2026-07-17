from abc import ABC, abstractmethod


class Executor(ABC):
    r"""The distribution axis of the Executor x InferenceStrategy bridge
    (report-2026-07-16-segmentation-architecture.md): how many regions a
    workflow runs on, and how results across those regions are assembled.

    An Executor never implements a segmentation algorithm itself -- what
    actually runs on the pixels lives entirely behind self.strategy (see
    strategy.py). This is what lets SingleRegionExecutor and
    ChunkedExecutor share the same four InferenceStrategy subclasses
    without duplicating any inference logic, and what let the previous
    SerialExecutor/ParallelExecutor split (which mixed both concerns
    into one hierarchy) be replaced without touching engine code at all.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    @abstractmethod
    def run_workflow(self, engine, image, **kwargs):
        r"""Entrypoint for running the full segmentation workflow and
        returning a result in the shape historically expected by
        callers (SliceSegPipeline/VolumeSegPipeline)."""
