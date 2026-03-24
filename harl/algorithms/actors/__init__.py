"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hadt import HADT
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.rulebased import RULEBASED
from harl.algorithms.actors.genetic import GENETIC
from harl.algorithms.actors.mat.transformer_policy import TransformerPolicy

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hadt": HADT,
    "mappo": MAPPO,
    "rulebased": RULEBASED,
    "genetic": GENETIC,
    "mat": TransformerPolicy
}
