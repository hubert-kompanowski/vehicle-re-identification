import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    # print(OmegaConf.to_yaml(config))

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    hash_cv = str(uuid.uuid4())
    config.hash_cv = "single_"+hash_cv

    # Applies optional utilities
    utils.extras(config)

    if config.get("run_cv"):

        for fold in range(4):
            config.datamodule.val_fold = [fold]
            train(config)
    else:
        return train(config)


if __name__ == "__main__":
    main()
