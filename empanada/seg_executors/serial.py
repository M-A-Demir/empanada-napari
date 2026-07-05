from .executor import Executor

class SerialExecutor(Executor):
    def __init__(self,
                 fill_holes_in_segmentation=False
                 ):
        super().__init__(fill_holes_in_segmentation)


    def run_workflow(self, engine, image=None, axis=None, plane=None, y=None, x=None):
        return self._get_segmentation(engine, image, axis, plane, y, x)