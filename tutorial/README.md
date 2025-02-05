
# Running FlowMDM

## 💾 Pretrained models

Download them from **TO BE ADDED** and extract its content to the repo root.
The repo should look like
```
RPM
├── checkpoints
├──── amass_p1/
├────── reactive
├────── smooth
├──── amass_p2/
├────── reactive
├────── smooth
├──── ...
├── body_visualizer/
├── human_body_prior/
├── SMPL/
└── ...
```

## 🗂️ Data preparation

<!-- <details> -->

**AMASS Protocol 1 (P1)**:

1. Download the `SMPL-X N` versions of the BMLrub, CMU and the HDM05 datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage, and uncompress them.
2. Run the following script:

```bash
python prepare_data.py --save_dir ./datasets_processed/amass_p1 --root_dir PATH_TO_AMASS_DATASET --splits_dir prepare_data/amass_p1 --out_fps 60
```

> [!NOTE]
> Add the `--cpu` argument if you get a `RuntimeError: CUDA error: out of memory`. Note that the preprocessing will take a bit longer.

**AMASS Protocol 2 (P2)**:

1. Download the `SMPL-X N` versions of the ACCAD, BMLmovi, BMLrub, CMU, EKUT, EyesJapanDataset, HDM05, HumanEva, KIT, MoSh, PosePrior, SFU, TotalCapture, and Transitions datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage, and uncompress them.
2. Run the following script:

```bash
python prepare_data.py --save_dir ./datasets_processed/amass_p2 --root_dir PATH_TO_AMASS_DATASET --splits_dir prepare_data/amass_p2 --out_fps 30
```

**GORP**:

1. Download the GORP dataset from **TO BE RELEASED** and uncompress it.
2. Run the following script:

```bash
python prepare_data_gorp.py --root_dir PATH_TO_GORP_DATASET
```

<!-- </details> -->


## 📊 Evaluation

To evaluate any of the models, make sure you followed the dataset preparation steps from above and run these commands with DATASET_FOLDER replaced with `amass_p1` or `amass_p2`, and MODEL_NAME with `reactive` or `smooth`:

**[AMASS-P1/P2 MC setup]**
```bash
python test.py --model_path ./checkpoints/<DATASET_FOLDER>/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16
```

**[AMASS-P1/P2 HT setup]**
```bash
python test.py --model_path ./checkpoints/<DATASET_FOLDER>/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config hand_tracking
```

**[GORP Synthetic Inputs]**

For the GORP dataset, we can evaluate either on synthetic tracking inputs (using SMPL GT head/wrists), or using the real input from the headset IMU and wrists IMU/hand-tracking signals. You can specify the `--test_split`: `test_controllers` (MC setup) or `test_tracking` (HT setup).

```bash
python test.py --model_path ./checkpoints/gorp/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config real_input --test_split TEST_SPLIT
```

**[GORP Real Inputs]**

The paper includes a synth-to-real and a real-to-real evaluation. To reproduce them, run the following command replacing DATASET_FOLDER with `gorp` or `gorp_real_inputs`, respectively:

```bash
python test.py --model_path ./checkpoints/<DATASET_FOLDER>/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config real_input --test_split TEST_SPLIT --use_real_input --input_conf_threshold 0.8
```

> [!NOTE]
> If you get an `RuntimeError: CUDA error: out of memory` error, try decreasing the `--eval_batch_size`, or running the evaluation in CPU with the `--cpu` flag.


## 🎬 Visualization

To generate a visualization of a model in a particular dataset and setup, use the evaluation commands above replacing:
`--eval --eval_batch_size 16`
with
`--vis --vis_overwrite`.

> [!WARNING]
> Comment all `os.environ['PYOPENGL_PLATFORM'] = "egl"` in the project if you get an `ImportError: ('Unable to load EGL library', "Could not find module 'EGL' (or one of its dependencies). Try using the full path with constructor syntax.", 'EGL', None)` error.


### Render SMPL meshes in Unity

If you add `--vis_export`, the visualizer will store an `.obj` file for each frame containing the SMPL mesh. It will also generate `.json` files with information on the skeleton predictions (W skeletons per frame, coordinates in world space), the tracking input (0, 1, 2 are headset, left and right wrists) world coordinates (x, y, z) and orientation (rw, rx, ry, rz). These can be used to generate visualizations such as the ones in the paper.


## 🏋️‍♂️ Training

**[AMASS-P1]** To retrain our `RPM - Reactive` model with A-P1, run:

```bash
python train.py --results_dir ./results/amass_p1_retrained --dataset amass_p1 --train_dataset_repeat_times 100 --batch_size 512 --input_motion_length 10 --exp_name reactive --rolling_fr_frames 60 --rolling_motion_ctx 10 --rolling_sparse_ctx 10 --loss_velocity 1 --loss_fk 1 --loss_fk_vel 1 --overwrite
```

Set `--input_motion_length 20 --exp_name smooth` to train the `RPM - Smooth` version reported in the paper.

**[AMASS-P2]** To retrain our `RPM - Reactive` model with A-P2, run:

```bash
python train.py --results_dir ./results/amass_p2_retrained --dataset amass_p2 --train_dataset_repeat_times 100 --batch_size 512 --input_motion_length 10 --exp_name reactive --rolling_fr_frames 30 --rolling_motion_ctx 10 --rolling_sparse_ctx 10 --loss_velocity 1 --loss_fk 1 --loss_fk_vel 1 --overwrite
```
Set `--input_motion_length 10 --exp_name smooth` to train the `RPM - Smooth` version reported in the paper.

**[GORP]** To retrain our `RPM - Reactive` model with GORP, run:

```bash
python train.py --results_dir ./results/gorp --dataset gorp --train_dataset_repeat_times 100 --batch_size 512 --input_motion_length 10 --exp_name reactive --rolling_fr_frames 30 --rolling_motion_ctx 10 --rolling_sparse_ctx 10 --loss_velocity 1 --loss_fk 1 --loss_fk_vel 1 --overwrite
```
Set `--input_motion_length 10 --exp_name smooth` to train the `RPM - Smooth` version reported in the paper.
