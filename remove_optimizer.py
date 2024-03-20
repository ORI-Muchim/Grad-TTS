import utils
from utils import *

hps = utils.get_hparams()

utils.remove_optimizer_latest_checkpoint(hps.model_dir, hps.model_dir, "G_*.pth")
