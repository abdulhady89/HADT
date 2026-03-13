from absl import flags
from harl.envs.bsk.clusterbsk_logger import ClusterbskLogger


FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "bsk": ClusterbskLogger,
    "bsk_single_sat": ClusterbskLogger
}
