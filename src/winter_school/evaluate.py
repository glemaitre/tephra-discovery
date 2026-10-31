"""Unused splitter stub.

``skore.evaluate`` and ``project.put`` live in
``experiments/NN_*.py``, not here. The locked cross-validator sits
on the DataOp. Do not pass ``splitter=`` from the experiment
script, except ``splitter="prefit"`` when the locked folds cell
is ``predefined``: the learner is already fitted on the training
table and the call receives only the test table. ``None`` here
is not a holdout by itself: a holdout is a marker with no
``cv``. See ``evaluate-ml-pipeline/references/metadata-routing.md``.
"""

from __future__ import annotations

splitter = None
