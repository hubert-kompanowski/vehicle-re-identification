import glob
import sys

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
import uuid

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def get_ckpt_by_fold(saved_checkpoints, fold):
    for path in saved_checkpoints:
        if path.split('/')[-1].split('_')[2][-1] == str(fold):
            return path
    raise Exception('no checkpoint path found')


def run_stage(config, stage, hash_cv):
    from src.training_pipeline import train

    config.hash_cv = f'{stage}_{hash_cv}'
    config.datamodule.stage = stage
    config.model.stage = stage
    for fold in range(4):
        if stage == 'second':
            saved_checkpoints = glob.glob(f"{config.output_dir}/checkpoints/epoch_*_fold*")
            config.trainer.resume_from_checkpoint = get_ckpt_by_fold(saved_checkpoints, fold)
        config.datamodule.val_fold = [fold]
        config.callbacks.model_checkpoint.filename = "epoch_{epoch:03d}_fold" + f"{fold}_stage_{stage}"
        train(config)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    config.name = [x for x in sys.argv if 'experiment' in x][0].split('=')[-1]
    # print(OmegaConf.to_yaml(config))

    from src import utils
    from src.training_pipeline import train

    hash_cv = str(uuid.uuid4())

    utils.extras(config)

    if config.get("run_mdl"):
        run_stage(config, stage='first', hash_cv=hash_cv)
        run_stage(config, stage='second', hash_cv=hash_cv)

    elif config.get("run_cv"):
        config.hash_cv = hash_cv

        for fold in range(4):
            config.datamodule.val_fold = [fold]
            config.callbacks.model_checkpoint.filename = "epoch_{epoch:03d}_fold" + str(fold)
            train(config)

    else:
        config.hash_cv = "single_" + hash_cv
        return train(config)


if __name__ == "__main__":
    main()
