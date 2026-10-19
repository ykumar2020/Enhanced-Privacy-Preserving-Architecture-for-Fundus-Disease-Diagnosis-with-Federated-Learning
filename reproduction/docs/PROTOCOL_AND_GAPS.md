# Protocol map, corrections, and expansion path

## Published protocol mapped to code

| Published description | Reproduction implementation |
|---|---|
| ODIR; normal, glaucoma, cataract; 3,402 images | `manifest.py` derives labels from eye-specific keywords and reports whether the undocumented count is recovered. |
| 64/16/20 split | `assign_patient_splits`, with patients rather than images as the indivisible unit. |
| RGB CLAHE, bounding crop, 224x224 | `data.py`, in that order. |
| 50% horizontal and vertical flips | Applied only to training loaders. |
| ImageNet normalization | Canonical ImageNet channel mean and standard deviation. |
| ImageNet VGG-19, frozen features | `model.py`; the two 4096/ReLU/dropout blocks are retained and the output is replaced with three logits. |
| Four clients; two selected; one local epoch; five rounds | `paper_hyperparameters_patient_safe.yaml`. |
| Batch 16, Adam, learning rate 0.001 | Both centralized and federated runners. |
| Dataset-size weighted aggregation | `aggregate_sample_weighted`; denominator contains selected clients only and is asserted to sum to one. |

## Released-code findings

The journal's supplementary-material link resolves to the public repository owned by `rayzjiang` (formerly referenced as `coolraycode`). The audit used commit `6a2c21ce4b6dfbb3f2f9cb170041c198ef8ee3c8`; future comparisons should pin that commit rather than silently following the repository head. Its executable semantics differ from parts of the prose:

- Labels come from an untracked `labels_perEye.csv`, not directly from the official patient workbook.
- The task is three-output multi-label classification using `BCEWithLogitsLoss`, sigmoid, and a 0.55 threshold—not softmax multiclass classification.
- CLAHE is applied to the LAB luminance channel with clip limit 10, despite being described as RGB CLAHE.
- The ImageNet red-channel standard deviation is coded as 0.228 rather than torchvision's canonical 0.229.
- Random horizontal and vertical flips are included in the single transform used for training, validation, and testing.
- `random_split` is unseeded and operates on images, so patient separation is not established.
- Flower's standard `FedAvg` strategy is used. No custom aggregator is present.
- The client returns `len(trainloader)` (number of batches) instead of number of examples to FedAvg.
- Client evaluation references `self.testloader`, which is never defined.
- The centralized and federated evaluation denominators assume every batch is full, biasing the reported accuracy when the final batch is partial.
- No differential-privacy implementation is present.

Compatibility mode retains the released label semantics, threshold, preprocessing constants, and training hyperparameters while correcting the execution defects above and enforcing patient-separated evidence.

## Unrecoverable details

- Exact 3,402-image selection and exclusion ledger.
- Original train/validation/test patient assignments.
- Original client assignments and round-by-round selection schedule.
- Random seed and library-level determinism settings.
- Exact CLAHE clip limit, tile size, and background threshold.
- Whether validation influenced training or checkpoint selection.
- Differential-privacy clipping, noise, sampling, protected unit, delta, epsilon, and accountant.
- Per-run predictions and number of statistical repetitions.

These are parameters or evidence gaps, not opportunities to guess. Every new run records its actual choices.

## Corrected interpretation

- Dataset-size weighting is standard FedAvg behavior.
- The direct reported difference is 1.75 percentage points: 86.63% minus 84.88%.
- The relative reduction is approximately 2.02% of centralized accuracy.
- With four equal clients, two selected per round, five rounds, and one local epoch, the federated exposure is approximately 2.5 full-training-set passes, not five.
- Federated learning keeps raw images local but does not itself provide differential privacy.

## Expansion milestones

1. **ODIR recovery:** obtain the original annotations/images and produce the patient-safe selection ledger.
2. **Matched baselines:** run local-only, pooled centralized, and canonical FedAvg with repeated seeds and equalized exposure budgets.
3. **Source shift:** introduce documented camera/source domains and an external cohort rather than calling random clients hospitals.
4. **Patient-level DP:** add explicit per-patient clipping, calibrated noise, and an accountant; report epsilon and delta from the executed run.
5. **Compact adaptation:** compare classifier-only training with a small adapter on the same frozen retinal encoder.
6. **External evaluation:** freeze thresholds before testing and report each source, the weakest source, calibration, communication, and uncertainty by patient.

The framework deliberately separates executed evidence from proposed experiments. No table or figure should contain a numerical result unless it is regenerated from a saved prediction ledger.

