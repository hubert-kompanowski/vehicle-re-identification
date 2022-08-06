import sys

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    config.name = [x for x in sys.argv if 'experiment' in x][0].split('=')[-1]
    # print(OmegaConf.to_yaml(config))

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    hash_cv = str(uuid.uuid4())

    # Applies optional utilities
    utils.extras(config)

    if config.get("run_cv"):
        config.hash_cv = hash_cv

        for fold in range(4):
            config.datamodule.val_fold = [fold]
            train(config)
    else:
        config.hash_cv = "single_" + hash_cv
        return train(config)


if __name__ == "__main__":
    main()
