class UnionFind:
    r"""Minimal union-find for collapsing chunk-local instance/label ids
    that ChunkReconciler.record_matches has determined refer to the same
    real object (report-2026-07-15-orthoplane-strategy.md, section 4).

    Built and resolved entirely by the driver (ChunkedExecutor), after
    every wave of a chunked workflow has returned its edges -- never
    mutated from inside a parallel worker. joblib's default backend
    (loky) uses separate processes, so there is no shared memory for a
    worker to mutate a live union-find into; edges must be returned by
    each worker and folded in here, in the driver process, same as
    trackers/segmentations already are.

    Recording every match above threshold as an edge (rather than
    immediately relabeling to the single best match) is what lets a
    multi-way merge collapse correctly: an object that a boundary strip
    reveals is actually the same as two separately-labeled panel
    fragments needs all three ids in one connected component, not one
    fragment adopted and the other silently orphaned.
    """

    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb
