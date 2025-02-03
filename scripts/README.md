
# Running FlowMDM

## 💾 Pretrained models

Download the pretrained models from our repository:

```bash
TO ADD
```

## 🗂️ Data preparation

<!-- <details> -->

**AMASS Protocol 1 (P1)**:

1. Download the `SMPL-X N` versions of the BMLrub, CMU and the HDM05 datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage.
2. Run the following script:

```bash
python prepare_data.py --support_dir PATH_TO_SMPL_MODELS --save_dir ./datasets_processed/amass_p1 --root_dir PATH_TO_AMASS_DATASET --splits_dir prepare_data/amass_p1 --out_fps 60
```

3. Download the mean/std of the dataset from **TO-BE-ADDED**
4. Download the json files describing the random gaps from [here (TO BE ADDED)](), and move them to `./datasets_processed/amass_p1/eval_gap_configs/`.

> [!NOTE]
> Add the `--cpu` argument if you get a `RuntimeError: CUDA error: out of memory`. Note that the preprocessing will take a bit longer.

**AMASS Protocol 2 (P2)**:

1. Download the `SMPL-X N` versions of the ACCAD, BMLmovi, BMLrub, CMU, EKUT, EyesJapanDataset, HDM05, HumanEva, KIT, MoSh, PosePrior, SFU, TotalCapture, and Transitions datasets from the [AMASS](https://amass.is.tue.mpg.de/download.php) downloads webpage.
2. Run the following script:

```bash
python prepare_data.py --support_dir PATH_TO_SMPL_MODELS --save_dir ./datasets_processed/amass_p2 --root_dir PATH_TO_AMASS_DATASET --splits_dir prepare_data/amass_p2 --out_fps 30
```

3. Download the mean/std of the dataset from **TO-BE-ADDED**
4. Download the json files describing the random gaps from [here (TO BE ADDED)](), and move them to `./datasets_processed/amass_p2/eval_gap_configs/`.

**GORP**:

**------ TO BE RELEASED ------**

```bash
python prepare_data_gorp.py --support_dir PATH_TO_SMPL_MODELS --save_dir PATH_TO_AMASS_DATASET --root_dir WHERE_TO_STORE_PROCESSED_DATASET --splits_dir prepare_data/gorp --out_fps 30
```

<!-- </details> -->


## 📊 Evaluation

To evaluate any of the models, make sure you followed the dataset preparation steps from above and run:

**[AMASS-P1/P2 MC setup]**
```bash
python test.py --model_path ./checkpoints/amass_p1/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16
```

**[AMASS-P1/P2 HT setup]**
```bash
python test.py --model_path ./checkpoints/amass_p1/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config hand_tracking
```

**[GORP dataset]**

For the GORP dataset, we can evaluate either on synthetic tracking inputs (using SMPL GT head/wrists), or using the real input from the headset IMU and wrists IMU/hand-tracking signals. You can specify the `--test_split`: `test_controllers` (MC setup) or `test_tracking` (HT setup).

**[GORP Synthetic Inputs]**
```bash
python test.py --model_path ./checkpoints/gorp/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config real_input --test_split TEST_SPLIT
```

**[GORP Real Inputs]**
```bash
python test.py --model_path ./checkpoints/gorp/<MODEL_NAME>/model_latest.pt --eval --eval_batch_size 16 --eval_gap_config real_input --test_split TEST_SPLIT --use_real_input --input_conf_threshold 0.8
```

> [!NOTE]
> If you get an `RuntimeError: CUDA error: out of memory` error, try decreasing the `--eval_batch_size`, or running the evaluation in CPU with the `--cpu` flag.


## 🎬 Visualization

To generate a visualization of a model in a particular dataset and setup, use the evaluation command replacing:
`--eval --eval_batch_size 16`
with
`--vis --vis_overwrite`.

> [!WARNING]
> Comment all `os.environ['PYOPENGL_PLATFORM'] = "egl"` in the project if you get the `ImportError: ('Unable to load EGL library', "Could not find module 'EGL' (or one of its dependencies). Try using the full path with constructor syntax.", 'EGL', None)` error.


### Render SMPL meshes in Unity

If you use `--vis_export`, the visualizer will store an .obj file for each frame containing the SMPL mesh. It will also generate `.json` files with information on the skeleton predictions (W skeletons per frame, coordinates in world space), the tracking input (0, 1, 2 are headset, left and right wrists) world coordinates (x, y, z) and orientation (rw, rx, ry, rz). These can be used to generate visualizations like in the paper.


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
