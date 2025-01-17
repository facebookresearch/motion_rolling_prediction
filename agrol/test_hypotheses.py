# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import os
import random

import numpy as np
import pandas as pd
import torch
from data_loaders.dataloader import load_data_from_manifold, TestDataset
from human_body_prior.body_model.body_model import BodyModel as BM

from loguru import logger

from tqdm import tqdm

from utils import utils_transform
from utils.config import pathmgr
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import sample_args


def compute_div(samples, p=2):
    # samples: (num_samples, num_frames, feats)
    samples = samples.reshape(samples.shape[0], -1)
    return torch.nn.functional.pdist(samples, p=p).mean(-1)


class BodyModel(torch.nn.Module):
    def __init__(self, support_dir, device="cuda"):
        super().__init__()
        subject_gender = "male"
        bm_fname = pathmgr.get_local_path(
            os.path.join(support_dir, "smplh/{}/model.npz".format(subject_gender))
        )
        dmpl_fname = pathmgr.get_local_path(
            os.path.join(support_dir, "dmpls/{}/model.npz".format(subject_gender))
        )
        num_betas = 16  # number of body parameters
        num_dmpls = 8  # number of DMPL parameters
        body_model = BM(
            bm_fname=bm_fname,
            num_betas=num_betas,
            num_dmpls=num_dmpls,
            dmpl_fname=dmpl_fname,
        )
        logger.info("Body model loaded! Now moving to device:", device)
        device = torch.device(device)
        body_model = body_model.to(device)
        self.body_model = body_model.eval()

    def forward(self, body_params):
        with torch.no_grad():
            body_pose = self.body_model(
                **{
                    k: v
                    for k, v in body_params.items()
                    if k in ["pose_body", "trans", "root_orient"]
                }
            )
        return body_pose


def get_joints(sample, body_model, head_motion=None, device="cuda"):
    # it should receive: (num_frames, num_feats)
    motion_pred = sample.squeeze().to(device)
    # Get the  prediction from the model
    model_rot_input = (
        utils_transform.sixd2aa(motion_pred.reshape(-1, 6).detach())
        .reshape(motion_pred.shape[0], -1)
        .float()
    )

    # Get the offset between the head and other joints using forward kinematic model
    body_pose_local = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
        }
    ).Jtr

    # Get the offset in global coordiante system between head and body_world.
    t_head2root = -body_pose_local[:, 15, :]
    if head_motion is not None:
        T_head2world = head_motion.clone().to(device)
        t_head2world = T_head2world[..., :3, 3].clone()
        t_root2world = t_head2root + t_head2world.to(device)
    else:
        t_root2world = t_head2root

    predicted_body = body_model(
        {
            "pose_body": model_rot_input[..., 3:66],
            "root_orient": model_rot_input[..., :3],
            "trans": t_root2world,
        }
    )
    predicted_position = predicted_body.Jtr[:, :22, :]
    return predicted_position


def plot_features(intermediates, num_feat=-1):
    # param intermediates: list of intermediates from the diffusion process (dump_steps=True)
    import matplotlib.pyplot as plt
    import numpy as np

    # Plot denoised 'num_feat' for each sample in batch
    for i, sample_i in enumerate(intermediates[-1]["sample"]):
        plt.plot(sample_i.cpu().squeeze()[:, num_feat], label=f"sample #{i}")
    plt.legend()
    plt.savefig("/tmp/denoising_vis_5_samples.png")
    plt.close()

    # seismic colormap --> 5 colors
    colors = plt.cm.copper_r(np.linspace(0, 1, len(intermediates)))

    # Plot progressively denoised 'num_feat' for the first sample of the batch
    for i, sample_i in enumerate(intermediates):
        plt.plot(
            sample_i["sample"].cpu()[0, :, num_feat],
            label=f"{i}th step",
            color=colors[i],
        )
    plt.legend()
    plt.savefig("/tmp/denoising_vis.png")
    plt.close()

    # Plot fully denoised x0 from each denoising timestep for 'num_feat' for the first sample of the batch
    for i, sample_i in enumerate(intermediates):
        plt.plot(
            sample_i["pred_xstart"].cpu()[0, :, num_feat],
            label=f"{i}th step",
            color=colors[i],
        )
    plt.legend()
    plt.savefig("/tmp/denoising_vis_pred_xstart.png")
    plt.close()


def get_hypotheses_statistics(
    args,
    data,
    sample_fn,
    dataset,
    model,
    body_model,
    sld_wind_size=70,
    model_type="diffusion",
    div_reps=5,
    device="cuda",
):
    assert (
        model_type == "diffusion"
    ), "currently only diffusion model supports overlapping test!!!"

    gt_data, sparse_original, _, head_motion, filename = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[4],
    )
    gt_data = gt_data.to(device).float()
    sparse_original = sparse_original.to(device).float()
    head_motion = head_motion.to(device).float()
    num_frames = head_motion.shape[0]

    output_samples = []
    seq_final = torch.zeros_like(gt_data, device=device)
    # seq_final_joints = torch.zeros((gt_data.shape[0], 66), device=device)
    last_idx_predicted = -1
    count = 0
    sparse_splits = []
    flag_index = None

    if (
        num_frames < args.input_motion_length
    ):  # if the sequence is shorter than 196, we pad it with the first frame of the sparse signals sequence
        flag_index = args.input_motion_length - num_frames
        tmp_init = sparse_original[:, :1].repeat(1, flag_index, 1).clone()
        sub_sparse = torch.concat([tmp_init, sparse_original], dim=1)
        segment = (-flag_index, args.input_motion_length)
        sparse_splits = [
            [sub_sparse, 0, segment],
        ]

    else:  # if it is longer than 196, we split the sequence into subsequences that are overlapping 196-W frames (W=sld_wind_size)
        while count + args.input_motion_length <= num_frames:
            if count == 0:  # first iteration --> needs to go from 0 to 196
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = 0  # where the subsequence starts
            else:
                sub_sparse = sparse_original[
                    :, count : count + args.input_motion_length
                ]
                tmp_idx = args.input_motion_length - sld_wind_size
            segment = (count, count + args.input_motion_length)
            sparse_splits.append([sub_sparse, tmp_idx, segment])
            count += sld_wind_size

        if count < num_frames:  # add the last subsequence of length 196
            sub_sparse = sparse_original[:, -args.input_motion_length :]
            tmp_idx = args.input_motion_length - (
                num_frames - (count - sld_wind_size + args.input_motion_length)
            )
            segment = (num_frames - args.input_motion_length, num_frames)
            sparse_splits.append([sub_sparse, tmp_idx, segment])

    memory = None  # init memory

    if args.fix_noise:
        logger.warning("WARNING: fixing noise!!")
        # fix noise seed for every frame
        noise = torch.randn(1, 1, 1, device=device)
        noise = noise.repeat(1, args.input_motion_length, args.motion_nfeat)
    else:
        noise = None

    data_h1, data_h2 = [], []
    for step_index in range(len(sparse_splits)):
        sparse_per_batch = sparse_splits[step_index][0]
        memory_end_index = sparse_splits[step_index][1]
        segment = sparse_splits[step_index][2]

        new_batch_size = div_reps
        shape = (new_batch_size, args.input_motion_length, args.motion_nfeat)

        if memory is not None:
            model_kwargs = {}
            model_kwargs["y"] = {}
            model_kwargs["y"]["inpainting_mask"] = torch.zeros(
                shape,
                device=device,
            )
            model_kwargs["y"]["inpainting_mask"][:, :memory_end_index, :] = 1
            model_kwargs["y"]["inpainted_motion"] = torch.zeros(
                shape,
                device=device,
            )
            model_kwargs["y"]["inpainted_motion"][:, :memory_end_index, :] = memory[
                :, -memory_end_index:, :
            ].expand(new_batch_size, -1, -1)
        else:  # there is no previous step, so we simply generate the first 0-196 frames
            model_kwargs = None

        deactivate_last_inpainting = True
        intermediates = sample_fn(
            model,
            (new_batch_size, args.input_motion_length, args.motion_nfeat),
            sparse=sparse_per_batch.expand(div_reps, -1, -1),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=False,
            dump_steps=True,
            noise=noise,
            const_noise=False,
            deactivate_last_inpainting=deactivate_last_inpainting,
        )
        all_samples = intermediates[-1]["sample"]

        sample = all_samples[0].unsqueeze(0)
        memory = (
            sample.clone().detach()
        )  # we keep this iteration as the memory for the next step
        if deactivate_last_inpainting and step_index == 0:
            logger.warning(
                "WARNING: AR-AGRoL only works properly when deactivate_last_inpainting=False, otherwise it will keep changing the inpainting context of the next iteration --> unstabilities."
            )

        if (
            flag_index is not None
        ):  # only for sequences shorter than 196 --> not relevant for autoregressive approach
            sample = sample[:, flag_index:].cpu().reshape(-1, args.motion_nfeat)
        else:  # get the generated frames (last frames inside window size)
            sample = sample[:, memory_end_index:].reshape(-1, args.motion_nfeat)

        if step_index > 0:
            # convert to joints
            num_reps = all_samples.shape[0]
            num_frames = all_samples[0].shape[-2]
            samples_j = torch.stack(
                [
                    get_joints(
                        (
                            dataset.inv_transform(s.squeeze(1).cpu()).to(device)
                            if not args.no_normalization
                            else s.squeeze(1)
                        ),
                        body_model,
                        device=device,
                    )
                    for s in all_samples
                ],
                dim=0,
            )
            samples_j = samples_j.reshape(num_reps, num_frames, -1)

            # h1
            div_context, div_predicted = compute_statistics_H1(
                samples_j, memory_end_index
            )
            data_h1.append(
                (filename, segment[0], segment[1], div_context, div_predicted)
            )

            # h2
            current_idces = torch.arange(segment[0], segment[1], device=device)

            diff, dist = compute_statistics_H2(
                all_samples[0].squeeze(),
                seq_final[: last_idx_predicted + 1],
                current_idces,
                last_idx_predicted,
            )

            """
            # TODO: implement H2 for joints, which would be more descriptive
            diff_joints, dist_joints = compute_statistics_H2(
                samples_j[0].squeeze(),
                seq_final_joints[: last_idx_predicted + 1],
                current_idces,
                last_idx_predicted,
            )
            """

            for difference, distance in zip(diff, dist):
                data_h2.append(
                    (
                        filename,
                        segment[0],
                        segment[1],
                        distance,
                        difference,
                    )
                )

        if not args.no_normalization:
            output_samples.append(dataset.inv_transform(sample.cpu().float()))
        else:
            output_samples.append(sample.cpu().float())
        seq_final[last_idx_predicted + 1 : last_idx_predicted + sample.shape[0] + 1] = (
            sample.cpu().float()
        )
        last_idx_predicted += sample.shape[0]

        # if last_idx_predicted > 197:
        #     plot_features(intermediates)
        #     exit()

    return data_h1, data_h2, filename


def load_diffusion_model(args, device="cuda"):
    logger.info("Creating model and diffusion...")
    args.arch = args.arch[len("diffusion_") :]
    model, diffusion = create_model_and_diffusion(args)

    logger.info(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(pathmgr.get_local_path(args.model_path), map_location="cpu")
    load_model_wo_clip(model, state_dict)

    model.to(device)  # dist_util.dev())
    model.eval()  # disable random masking
    return model, diffusion


def compute_statistics_H1(samples_j, memory_end_index):
    # samples_j is [num_reps, num_frames, joints*feats] --> [5, 196, 66]
    # ============== Hypothesis 1 --> stochasticity (diversity) ==============
    inpainted = samples_j[:, :memory_end_index, :]
    predicted = samples_j[:, memory_end_index:, :]
    return compute_div(inpainted).item(), compute_div(predicted).item()


def compute_statistics_H2(sample, sample_ref, current_idces, last_idx_predicted):
    # get memory idx in curent sample/idces. Alternatively we could use the memory idx.
    memory_idx = torch.where(current_idces == last_idx_predicted)[0]
    memory_idces = current_idces[: memory_idx + 1]
    dist_to_present = current_idces[-1] - last_idx_predicted
    diff = sample[: memory_idx + 1] - sample_ref[memory_idces]

    distances = memory_idces[-1] - memory_idces + dist_to_present
    diff_per_dist = diff.abs().mean(axis=-1)
    return diff_per_dist.cpu().numpy(), distances.cpu().numpy()


def test_hypotheses_dummy(args):
    LENGTH = 100
    W = 2
    MAX_SEQ = 10

    statistics = {
        "div_context": [],  # H1
        "div_predicted": [],  # H1
        "err_dist": {distance: [] for distance in range(W, MAX_SEQ)},  # H2
    }

    all_samples = [torch.rand((MAX_SEQ, 132), dtype=torch.float) for _ in range(5)]
    body_model = BodyModel(args.support_dir, device="cpu")
    memory_end_index = MAX_SEQ - W

    num_reps = len(all_samples)
    num_frames = all_samples[0].shape[-2]
    all_samples_joints = torch.stack(
        [get_joints(s.squeeze(1), body_model, device="cpu") for s in all_samples],
        dim=0,
    )
    all_samples_joints = all_samples_joints.reshape(num_reps, num_frames, -1)
    div_ctx, div_pred = compute_statistics_H1(all_samples_joints, memory_end_index)
    statistics["div_context"].append(div_ctx)
    statistics["div_predicted"].append(div_pred)

    seq = torch.rand((LENGTH, 132), dtype=torch.float)
    seq_final = torch.zeros_like(seq)
    # sample_ref --> keeps updating with latest predicted results
    last_idx_predicted = MAX_SEQ - 1

    for i0 in range(W, LENGTH - MAX_SEQ, W):
        i = i0 + MAX_SEQ
        logger.info(f"{i0=} to {i}")
        current_idces = torch.arange(i0, i)
        sample = seq[i0:i]
        diff, dist = compute_statistics_H2(
            sample,
            seq_final[: last_idx_predicted + 1],
            current_idces,
            last_idx_predicted,
        )
        last_idx_predicted = i - 1

        for difference, distance in zip(diff, dist):
            statistics["err_dist"][distance].append(difference)

    logger.info(statistics)
    exit()


def main():
    args = sample_args()
    device = f"cuda:{args.device}" if not args.cpu else "cpu"
    logger.info(f"Device: {device}")
    # test_hypotheses_dummy(args)

    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("Loading dataset...")
    filename_list, all_info, mean, std = load_data_from_manifold(
        args.dataset,
        args.dataset_path,
        "test",
    )
    dataset = TestDataset(
        args.dataset,
        mean,
        std,
        all_info,
        filename_list,
    )

    model_type = args.arch.split("_")[0]
    model, diffusion = load_diffusion_model(args, device=device)
    sample_fn = diffusion.ddim_sample_loop

    # sliding window size in case of overlapping testing
    n_testframe = args.sld_wind_size

    if args.random:
        order = np.random.permutation(len(dataset))
    else:
        order = np.arange(len(dataset))

    logger.info("Loading body model...")
    body_model = BodyModel(args.support_dir, device=device)

    logger.info("Overlapping testing...")
    # all_h1, all_h2 = [], []
    is_first = True
    for i in tqdm(range(len(dataset))):
        sample_index = order[i]
        num_frames = dataset[sample_index][1].shape[1]
        if num_frames < args.min_frames:
            continue

        data_h1, data_h2, filename = get_hypotheses_statistics(
            args,
            dataset[sample_index],
            sample_fn,
            dataset,
            model,
            body_model,
            n_testframe,
            model_type=model_type,
            div_reps=5,
            device=device,
        )
        if data_h1 is None and data_h2 is None:
            continue

        # Write data to CSV files
        df_h1 = pd.DataFrame(
            data_h1,
            columns=["filename", "start", "end", "div_context", "div_predicted"],
        )
        df_h2 = pd.DataFrame(
            data_h2,
            columns=[
                "filename",
                "start",
                "end",
                "distance",
                "difference",
            ],
        )
        suffix = f"{args.timestep_respacing}_{args.sld_wind_size}_minframes_{args.min_frames}"
        df_h1.to_csv(
            f"/home/germanbarquero/persistent/private-90d/results_hypotheses_v2/h1_analysis_{suffix}.csv",
            index=False,
            mode="a",
            header=is_first,
        )
        df_h2.to_csv(
            f"/home/germanbarquero/persistent/private-90d/results_hypotheses_v2/h2_analysis_{suffix}.csv",
            index=False,
            mode="a",
            header=is_first,
        )
        is_first = False


if __name__ == "__main__":
    main()
