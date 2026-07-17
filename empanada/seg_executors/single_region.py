from .executor import Executor


class SingleRegionExecutor(Executor):
    r"""Runs the strategy on the whole image/volume as a single region --
    there is no cross-region identity to resolve, so finalize() runs
    immediately after run().

    Renamed from SerialExecutor (report-2026-07-16-segmentation-architecture.md):
    the old name described concurrency, an orthogonal internal detail --
    a strategy could run "serially" over one single, huge region and
    still belong here. What actually varies between this class and
    ChunkedExecutor is the number of regions, which is what the new name
    describes instead.
    """

    def run_workflow(self, engine, image, **kwargs):
        result = self.strategy.run(engine, image, **kwargs)
        result = self.strategy.finalize(result, engine=engine, **kwargs)
        return result.to_return_value()
