import hydra
from omegaconf import DictConfig, OmegaConf
import scribe
import pandas as pd
import jax.numpy as jnp
import pickle
import os
from scribe.data_loader import load_and_preprocess_anndata


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Running with config:\n", OmegaConf.to_yaml(cfg))

    # Load data
    data_path = hydra.utils.to_absolute_path(cfg.data.path)
    counts = load_and_preprocess_anndata(
        data_path, cfg.data.get("preprocessing")
    )

    # Prepare arguments for run_scribe from the config
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # Move inference-specific args to top level and rename 'method' to
    # 'inference_method'
    inference_kwargs = kwargs.pop("inference")
    kwargs["inference_method"] = inference_kwargs.pop("method")
    kwargs.update(inference_kwargs)

    # Remove keys that are not arguments to run_scribe
    del kwargs["data"]
    if "hydra" in kwargs:
        del kwargs["hydra"]

    # Run the inference
    results = scribe.run_scribe(counts=counts, **kwargs)

    print("Inference complete.")

    # Save the results
    output_file = "scribe_results.pkl"
    print(f"Saving results to {os.getcwd()}/{output_file}")
    with open(output_file, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
