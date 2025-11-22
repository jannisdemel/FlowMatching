# train.py
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()