## Configs

---

<aside>
💡 contains files related to various configurations

</aside>


## `dataset`
Defines where the data lives and how it is handled.

- **`name`**: Dataset identifier (string label used in code paths, metadata, or conditional logic).
- **`path`**: Absolute or relative path to the dataset root.
- **`saved_images`**: Whether the path is tensor file or just image paths. Set to false if using image paths (raw images)

---

## `model`
Defines the architecture and optional modules that can be toggled on/off.

### Core model settings
- **`name`**: Model identifier 
- **`pretrained`**: Whether to load pretrained weights (if supported by the chosen model).
- **`kernel_size`**: Convolution kernel size used by the model blocks (commonly `3`).
- **`base_filters`**: Base channel width (scales the whole network capacity).
- **`activation`**: Activation function used in blocks.
  - Current values: `relu`, `GLU` 
- **`dropout`**: Dropout probability for regularization.
- **`segmentation_type`**: Task head mode.
  - Current values: `semantic`, `instance`.

### LiquidNN settings
- **`use_lnn`**: Enables the LiquidNN module.

### Hopfield / memory settings
- **`use_hopfield`**: Enables the Hopfield-based retrieval 

### Quantization
- **`use_quantizer`**: Enables residual quantization modules 

### Fusion
- **`use_fusion`**: Enables an additional fusion mechanism 

---

## `training`
Defines optimization and schedule settings.

- **`epochs`**: Total number of training epochs.
- **`batch_size`**: Mini-batch size.
- **`learning_rate`**: Base learning rate 

---

## `other`
Miscellaneous runtime controls.

- **`log_print`**: Whether to print logs/progress to stdout in addition to file logging.
- **`wandb`**: Whether to enable Weights & Biases logging.
  - `false` disables it.
  - If a string/project name is used (e.g., `"liquid_nuclei"`), code may treat it as a project identifier.
- **`debug`**: Enables debug mode (commonly reduces dataset size, disables heavy logging, or adds extra checks—depends on implementation).

---

## Notes on extending safely
- To add new options, keep them grouped under existing sections (`dataset`, `model`, `training`, `other`) so Hydra overrides stay clean.
- If you introduce a new model variant, prefer adding a new `model.name` option and keeping the interface consistent (inputs/outputs), so the training loop does not change.