"""wbc - Hybrid ViT-ECA-CF with Adaptive Loss Reweighting for imbalanced
white blood cell classification.

The package is organised so that every quantity reported in the manuscript
has exactly one implementation:

* ``wbc.config``     - the single source of truth for hyper-parameters
* ``wbc.data``       - leakage-safe train / validation / test protocol
* ``wbc.models``     - ECA (adaptive kernel), colour-feature branch, hybrid
                       CNN-Transformer and ViT-B/16 variants, CNN baselines
* ``wbc.losses``     - CE, weighted CE (Eq. 13), focal loss and the corrected
                       adaptive loss reweighting (Eq. 14-15)
* ``wbc.metrics``    - all metrics with documented aggregation rules
* ``wbc.stats``      - exact paired tests and their small-n limits
* ``wbc.benchmark``  - parameters, FLOPs and a controlled latency protocol
* ``wbc.explain``    - Grad-CAM with an explicit selection protocol
"""

__version__ = "1.0.0"
