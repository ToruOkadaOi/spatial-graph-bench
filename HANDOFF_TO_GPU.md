# GPU Training Handoff

Use [docs/RETRAINING.md](docs/RETRAINING.md) for the complete training workflow. It covers frozen inputs, code transfer, instance setup, smoke and pilot runs, the full sweep, checkpoint resume, and result verification.

## Run overview

1. Prepare or verify the frozen inputs on the laptop.
2. Transfer the code bundle and input archive to a Linux x86-64 instance with an NVIDIA GPU.
3. Verify the checksums, install the locked environment, and run the GPU check.
4. Run the smoke test and pilot before starting the full sweep in `tmux`.
5. Package results, transfer them back, and verify the returned metrics against the frozen inputs.

The commands and required paths are maintained in [docs/RETRAINING.md](docs/RETRAINING.md).
