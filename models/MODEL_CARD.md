# BrainIAC Phase 2 Assets

BrainIAC is used for non-commercial academic research only, under
`models/licenses/brainiac.txt`.

## Runtime

The BrainIAC backbone must run inside the `brainiac-svc` image with
`monai==1.3.2`. Later MONAI versions add inactive cross-attention parameters to
`ViT`, which breaks the strict checkpoint load used by BrainIAC.

## Assets

| File | Purpose | SHA256 |
|---|---|---|
| `models/brainiac-v1-simclr.pt` | SimCLR BrainIAC ViT-B checkpoint | `f22bdbcae26823a9d9e8aee883c6f24386ba4617339c12269848b6666cc62693` |
| `models/brainiac/atlases/nihpd_asym_13.0-18.5_t1w.nii` | BrainIAC NIHPD T1 atlas for raw future cohorts | `f10804664000688f0ddc124b39ce3ae27f2c339a40583bd6ff916727e97b77d0` |
| `models/brainiac/hdbet/0.model` | HD-BET fast-mode parameter file | `6f75233753c4750672815e2b7a86db754995ae44b8f1cd77bccfc37becd2d83c` |
| `models/licenses/brainiac.txt` | Research-only license note | `41d4cfcc2e98b9a8f04e0cbde37986a4ec88c45d587beecdd255046151dac320` |

## Feature Contract

`brainiac-svc` stores the 768-d CLS token from `features[0][:, 0]`. It must not
mean-pool the 216 patch tokens.
