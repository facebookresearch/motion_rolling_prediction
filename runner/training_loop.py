# MIT License
# Copyright (c) 2021 OpenAI
#
# This code is based on https://github.com/openai/guided-diffusion
# MIT License
# Copyright (c) 2022 Guy Tevet
#
# This code is based on https://github.com/GuyTevet/motion-diffusion-model
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import functools

import os
import tempfile
import time

import torch

from data_loaders.dataloader import TestDataset

from diffusion import logger
from diffusion.diffusion_model import RollingDiffusionModel
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import create_named_schedule_sampler
from evaluation.evaluation import EvaluatorWrapper
from evaluation.generators import create_generator
from evaluation.utils import BodyModelsWrapper
from evaluation.visualization import VisualizerWrapper
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW

from tqdm import tqdm
from utils import dist_util
from utils.constants import DataTypeGT, MotionLossType, RollingType

layout = {
    "quartiles": {
        MotionLossType.LOSS: ["Multiline", [f"loss_q{i}" for i in range(4)]],
        MotionLossType.ROT_MSE: ["Multiline", [f"rot_mse_q{i}" for i in range(4)]],
    },
}

class TrainLoop:
    def __init__(self, args, model, diffusion, data, device="cuda"):
        self.args = args
        self.seq_len = args.input_motion_length
        self.dataset = args.dataset
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.load_optimizer = args.load_optimizer
        self.use_fp16 = False
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.resume_epoch = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.results_dir / "checkpoints" / args.exp_name
        self.overwrite = args.overwrite

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step and self.load_optimizer:
            logger.log("loading optimizer state...")
            self._load_optimizer_state()

        self.device = torch.device("cpu")
        if device != "cpu" and torch.cuda.is_available() and dist_util.dev() != "cpu":
            self.device = torch.device(dist_util.dev())

        self.schedule_sampler_type = RollingType.UNIFORM
        if isinstance(diffusion, RollingDiffusionModel):
            self.schedule_sampler_type = diffusion.rolling_type
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type, diffusion
        )

        self.eval_during_training = args.eval_during_training
        self.vis_during_training = args.vis_during_training
        if self.eval_during_training or self.vis_during_training:
            self.test_evaluators = []
            self.test_visualizers = []
            self.test_suffixs = []
            body_model = BodyModelsWrapper(args.support_dir)
            to_evaluate_list = [
                ("", None, args.input_motion_length),
                ("_medHandsGaps", "medium_hands_idp", args.input_motion_length),
            ]
            for suffix, eval_gap_config, rolling_horizon in to_evaluate_list:
                self.test_suffixs.append(suffix)
                args.rolling_horizon = rolling_horizon
                test_dataset = TestDataset(
                    args.dataset,
                    args.dataset_path,
                    args.no_normalization,
                    min_frames=args.min_frames,
                    max_frames=args.max_frames,
                    eval_gap_config=eval_gap_config,
                    use_real_input=args.use_real_input,
                    input_conf_threshold=args.input_conf_threshold,
                )
                test_generator = create_generator(
                    args, model, diffusion, test_dataset, device, body_model
                )
                if self.eval_during_training:
                    evaluator = EvaluatorWrapper(
                        args,
                        test_generator,
                        test_dataset,
                        body_model,
                        self.device,
                        batch_size=min(args.batch_size, 64),
                    )
                    (self.save_dir / ("eval" + suffix)).mkdir(parents=True, exist_ok=True)
                    self.test_evaluators.append(evaluator)
                if self.vis_during_training:
                    visualizer = VisualizerWrapper(
                        args,
                        test_generator,
                        test_dataset,
                        body_model,
                        self.device,
                    )
                    (self.save_dir / ("vis" + suffix)).mkdir(parents=True, exist_ok=True)
                    self.test_visualizers.append(visualizer)

        self.use_ddp = False
        self.ddp_model = self.model

        # LOGGING STUFF
        logger.log(args)
        train_log_dir = args.results_dir / "logging" / "tensorboard" / args.exp_name / "train"
        self.tb_writer = SummaryWriter(log_dir=train_log_dir)
        self.tb_writer.add_custom_scalars(layout)

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            self.resume_epoch = self.resume_step // len(self.data) + 1
            logger.log(
                f"loading model from checkpoint: {resume_checkpoint} at epoch {self.resume_epoch}..."
            )
            assert resume_checkpoint.exists(), "resume_checkpoint does not exist."
            self.model.load_state_dict(
                dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            )

    def _load_optimizer_state(self):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = main_checkpoint.parents[0] / f"opt{self.resume_step:09}.pt"

        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        assert opt_checkpoint.exists(), "resume_checkpoint does not exist."
        state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        self.opt.load_state_dict(state_dict)

    def run_loop(self):
        for epoch in range(self.resume_epoch, self.num_epochs):
            logger.log(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            for gt_data, cond in tqdm(self.data):
                gt_data = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in gt_data.items()
                }
                cond = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in cond.items()
                }
                self.run_step(gt_data, cond)
                self.step += 1
            if epoch % self.save_interval == 0:
                self.save(save_latest=True)
                self.evaluate(epoch)
                self.visualize()
            if epoch % self.log_interval == 0:
                for k, v in logger.get_current().name2val.items():
                    if k == MotionLossType.LOSS:
                        logger.log("epoch[{}]: loss[{:0.5f}]".format(epoch, v))
                        logger.log("lr:", self.lr)
                        self.tb_writer.add_scalar("lr", self.lr, epoch)
                    self.tb_writer.add_scalar(k, v, epoch)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save(save_latest=True)
            self.evaluate(epoch)
            self.visualize()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._step_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        bs = cond[DataTypeGT.SPARSE].shape[0]
        sched = self.schedule_sampler.sample(
            bs, self.seq_len, self.device, train=True
        )  # dist_util.dev())
        t = sched.timesteps
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            batch,
            t,
            cond,
            dataset=self.data.dataset,
            model_kwargs={"padding_mask": sched.padding_mask},
        )

        losses = compute_losses()
        weights = sched.weights
        if (
            len(weights.shape) > 1
            and weights.shape[1] != losses[MotionLossType.LOSS].shape[1]
        ):
            # dynamically pad weights when using training strategies like freerunning + BPTT
            w_pad = torch.zeros(
                weights.shape[0],
                losses[MotionLossType.LOSS].shape[1] - weights.shape[1],
                device=weights.device,
                dtype=weights.dtype,
            )
            w_pad += weights[:, 0].unsqueeze(1)
            weights = torch.cat([w_pad, weights], dim=1)

        loss = (losses[MotionLossType.LOSS] * weights).mean()
        log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _step_lr(self):
        # One-step learning rate decay if needed.
        if not self.lr_anneal_steps:
            return
        if (self.step + self.resume_step) > self.lr_anneal_steps:
            self.lr = self.lr / 30.0
            self.lr_anneal_steps = False
        else:
            self.lr = self.lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = self.lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self, save_latest=False):
        filename_ckpt = self.ckpt_file_name()
        filename_opt = filename_ckpt.replace("model", "opt")
        
        state_dict = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        logger.log("saving model...")
        with open(self.save_dir / filename_ckpt, "wb") as f:
            torch.save(state_dict, f)
        with open(self.save_dir / filename_opt, "wb") as f:
            torch.save(self.opt.state_dict(), f)
            
        if save_latest:
            with open(self.save_dir / "model_latest.pt", "wb") as f:
                torch.save(state_dict, f)
            with open(self.save_dir / "opt_latest.pt", "wb") as f:
                torch.save(self.opt.state_dict(), f)
            
    def evaluate(self, epoch):
        if not self.args.eval_during_training:
            return
        self.model.eval()
        for evaluator, suffix in zip(self.test_evaluators, self.test_suffixs):
            eval_name = f"eval{suffix}"
            csv_path = self.save_dir, eval_name, f"results_{self.step+self.resume_step}.csv"
            start_eval = time.time()
            log, all_results_df, _ = evaluator.evaluate_all()
            evaluator.store_all_results(all_results_df, csv_path)
            evaluator.push_to_tb(log, self.tb_writer, epoch, suffix=suffix)
            end_eval = time.time()
            logger.info(
                f"[{eval_name}] Evaluation time: {round(end_eval-start_eval)/60}min"
            )
        self.model.train()

    def visualize(self):
        if not self.args.vis_during_training:
            return
        self.model.eval()
        for visualizer, suffix in zip(self.test_evaluators, self.test_suffixs):
            vis_name = f"vis{suffix}"
            output_dir = self.save_dir / vis_name / f"step_{self.step+self.resume_step}"
            output_dir.mkdir(parents=True, exist_ok=True)
            start_vis = time.time()
            visualizer.visualize_subset(output_dir)
            end_vis = time.time()
            logger.info(
                f"[{vis_name}] Visualization time: {round(end_vis-start_vis)/60}min"
            )
        self.model.train()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if ts.ndim == 1:
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
