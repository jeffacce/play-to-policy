import hydra
import joblib
import logging
import numpy as np


@hydra.main(version_base="1.2", config_path="configs", config_name="env_config")
def main(cfg):
    # Needs _recursive_: False since we have more objects within that we are instantiating
    # without using nested instantiation from hydra
    workspace = hydra.utils.instantiate(cfg.env.workspace, cfg=cfg, _recursive_=False)
    rewards, infos, results = workspace.run()
    logging.info("==== Summary ====")
    logging.info(rewards)
    logging.info(infos)
    logging.info(results)
    logging.info(f"Average reward: {np.mean(rewards)}")
    logging.info(f"Std: {np.std(rewards)}")
    logging.info(f"Average result: {np.mean(results)}")
    logging.info(f"Std: {np.std(results)}")


if __name__ == "__main__":
    main()
