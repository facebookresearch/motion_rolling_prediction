
# Running FlowMDM

## 💾 Pretrained models

```bash
TO ADD
```

## 🗂️ Data preparation

<!-- <details> -->

**AMASS Protocol 1 (P1)**:

1. Download the `SMPL-X N` versions of the BMLrub, CMU and the HDM05 datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage.
2. Run the following script:

```bash
python prepare_data.py --support_dir PATH_TO_SMPL_MODELS --save_dir PATH_TO_AMASS_DATASET --root_dir WHERE_TO_STORE_PROCESSED_DATASET --splits_dir prepare_data/amass_p1 --out_fps 60
```

> [!NOTE]
> Add the `--cpu` argument if you get a `RuntimeError: CUDA error: out of memory`. Note that the preprocessing will take a bit longer.

**AMASS Protocol 2 (P2)**:

1. Download the `SMPL-X N` versions of the ACCAD, BMLmovi, BMLrub, CMU, EKUT, EyesJapanDataset, HDM05, HumanEva, KIT, MoSh, PosePrior, SFU, TotalCapture, and Transitions datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage.
2. Run the following script:

```bash
python prepare_data.py --support_dir PATH_TO_SMPL_MODELS --save_dir PATH_TO_AMASS_DATASET --root_dir WHERE_TO_STORE_PROCESSED_DATASET --splits_dir prepare_data/amass_p2 --out_fps 30
```

**GORP**:

**------ TO BE RELEASED ------**

```bash
python prepare_data_gorp.py --support_dir PATH_TO_SMPL_MODELS --save_dir PATH_TO_AMASS_DATASET --root_dir WHERE_TO_STORE_PROCESSED_DATASET --splits_dir prepare_data/gorp --out_fps 30
```

<!-- </details> -->

## 🎬 Visualization

To generate examples of human motion compositions with Babel model run:

```bash
...
```

To generate examples of human motion compositions with HumanML3D model run:

```bash
...
```


### Render SMPL meshes in Unity

TO BE ADDED



## 📊 Evaluation

To reproduce the Babel evaluation over the motion and transition run:

```bash
...
```

To reproduce the HumanML3D evaluation over the motion and transition run:

```bash
...
```


## 🏋️‍♂️ Training

To retrain our model with AMASS on P1, run:

```bash
...
```

To retrain our model with AMASS on P2, run:

```bash
...
```

To retrain our model with GORP, run:

```bash
...
```
