# Google Colab training

The supported Colab workflow is documented in [README.md](README.md#train-and-validate-in-google-colab). Upload `MedScript_Colab_Training_fixed.zip` to Drive, select a GPU runtime, then run the notebook cells there to install the Colab dependencies, create data splits, train TrOCR, and evaluate on validation data.

The archive contains one split directory: `data/splits`. It includes the original partition manifests (`*_manifest.csv`) and receives the generated `train.csv`, `validation.csv`, `test.csv`, and duplicate-resolution report. Keeping manifests separate from generated split files lets the script preserve original train/validation/test membership when rerun.

The split script validates CSV fields and image files, resolves exact duplicate images to avoid leakage, and excludes identical image bytes with conflicting labels. Review `data/splits/duplicate_resolution.csv` after it completes. Do not start training unless splitting exits successfully. Training selects the model using validation exact-match accuracy, with CER as the tie-breaker, and saves epoch metrics to `training_metrics.csv`.

Use `validation.csv` to review predictions or compare decoding settings. Run `test.csv` evaluation only after model and decoding choices are finalized; do not use the test set for tuning. The annotation data does not contain writer or prescription IDs, so the split cannot guarantee writer-independent evaluation.

The trained model, metrics, validation predictions, and generated splits can be copied to Drive using the final notebook cell in the README. Keep the entire `trocr-medicine` folder, including its model and processor/tokenizer files.
