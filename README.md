# NOTE

- This branch contains code and experiments for an ongoing research submission.
- Model checkpoints and configs are released on [huggingface_link](https://huggingface.co/Yixuan/codec-add)
  
# OneBit

A training codebase for **Audio Deepfake Detection**. It supports the following architecture:

* **AudioDataset / AudioCollator** – variable-length inputs, data augmentation, and preprocessing via Huggingface Transformer’s `AutoFeatureExtractor`.
* **Front End** – Huggingface audio foundation models used to extract deep-learning features.
* **Back End** – Models that operate on extracted features.

Compared with the earlier [Tomato](https://github.com/XIAOYixuan/tomatoDD) library, OneBit adds stricter **type checking**, replaces dictionaries objects with `dataclasses` for module I/O, adopts a **factory pattern** that automatically discovers/instantiates new classes, and uses flexible **callbacks** to control checkpointing, logging, and more. These choices make the codebase both easier to extend and less error-prone.

> Note: parts of this codebase (`ConfigManager` for reading/writing config files and unit tests) were initially generated with claude-4-sonnet(cursor) and then refined by the developer.

> Note: This repository is still under development. We expect to migrate the model architecture from tomatoDD within the next two months (2025-08-23). 

# Repository Layout

```
onebit/
  configs/        # example configs for training and inference
  data/           # dataset design, collator functions, augmentation methods
  model/
    frontend/     # wrappers of Transformer lib for feature extraction
    backend/      # custom backend models (factory pattern)
    operations/   # reusable operators
  loss/
    builtin/      # losses provided by PyTorch (factory pattern)
    ...           # custom loss implementations (factory pattern)
  metrics/
    # currently only supports EER
  task/           # Training/Evaluation Pipelines
    trainer/      
    evaluator/     
    callbacks/    # hooks around run lifecycle (e.g., before/after train, per-epoch, per-batch)
      # examples:
      # - early_stop: decide whether to stop after each evaluation
      # - lr_scheduler: adjust LR after each epoch as needed
      # - tensorboard_monitor: write metrics to tensorboard as needed 
      # - write_infer_output: save predictions after inference
      # - write_train_output: save corresponding config file, checkpoints and final training summary (time, best EER, etc.)
  utils/
    # currently: logging helpers
  config.py       # core config management
                  # most components accept this object at init time
                  # includes a small self-test.
tests/
  # unit tests 
train.py          # entrypoint for training
infer.py          # entrypoint for inference
```

# Usage

The only required argument is a path to a yaml **config file**.
Any parameter in that config can be overridden from the command line. **CLI values take precedence** over yaml.


## Train

```bash
python train.py --config onebit/configs/your_config.yaml \
  exp.name=your_epic_exp_name \
  data.dataset.name=the_dataset_dir_base_name \
  ...
```

## Inference

```bash
python infer.py --config onebit/configs/your_config.yaml \
  exp.name=your_epic_exp_name \
  data.dataset.name=the_dataset_dir_base_name \
  ...
```

# Outputs

* **Training outputs (default):**

  ```
  <project_root>/output/<exp.name>/<exp.seed>/
  ```

* **Inference outputs (default):**

  ```
  <project_root>/output/<exp.name>/<exp.seed>/infer/
  ```

To change the root output directory, set `exp.root_dir` in your config or via the CLI override.
