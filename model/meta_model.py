# Copyright (c) Meta Platforms, Inc. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
from model.networks import DiffMLP
from utils.constants import DataTypeGT, ModelOutputType


class MetaModel(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        mask_cond_fn,
        latent_dim=256,
        num_layers=8,
        dropout=0.1,
        dataset="amass",
        sparse_dim=54,
        **kargs,
    ):
        super().__init__()

        self.arch = arch
        self.dataset = dataset

        self.input_feats = nfeats
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparse_dim = sparse_dim

        self.mask_cond_fn = mask_cond_fn
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)

        self.mlp = eval(self.arch)(
            self.latent_dim, seq=kargs.get("input_motion_length"), num_layers=num_layers
        )
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        self.sparse_process = nn.Linear(self.sparse_dim, self.latent_dim)
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, x, timesteps, cond, force_mask=False, y=None, **kwargs):
        """
        x: [batch_size, nfeats, nframes], denoted x_t in the paper
        sparse: [batch_size, nframes, sparse_dim], the sparse features
        timesteps: [batch_size] (int)
        """
        sparse_emb = cond[DataTypeGT.SPARSE]

        emb = self.embed_timestep(timesteps)  # time step embedding : [1, bs, d]

        # Pass the sparse signal to a FC
        sparse_emb = self.sparse_process(
            self.mask_cond_fn(sparse_emb, self.training, force_mask=force_mask)
        )

        # Pass the input to a FC
        x = self.input_process(x)

        # Concat the sparse feature with input
        x = torch.cat((sparse_emb, x), axis=-1)
        output = self.mlp(x, emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        return {
            ModelOutputType.RELATIVE_ROTS: output,
            ModelOutputType.SHAPE_PARAMS: None,
        }


class MetaModelRolling(nn.Module):
    def __init__(
        self,
        arch,
        nfeats,
        mask_cond_fn,
        latent_dim=256,
        num_layers=8,
        dropout=0.1,
        dataset="amass",
        sparse_dim=54,
        **kargs,
    ):
        super().__init__()

        self.arch = arch
        self.dataset = dataset

        self.input_feats = nfeats
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sparse_dim = sparse_dim

        self.framewise_time_emb = kargs.get("framewise_time_emb", False)
        self.rolling_motion_ctx = kargs.get("rolling_motion_ctx", 0)
        self.rolling_sparse_ctx = kargs.get("rolling_sparse_ctx", 0)
        self.total_seq_len = kargs.get("input_motion_length") + self.rolling_motion_ctx
        assert (
            self.rolling_sparse_ctx <= self.total_seq_len
        ), "sparse context can't be larger than total sequence length for this model"

        self.mask_cond_fn = mask_cond_fn
        self.input_process = nn.Linear(self.input_feats, self.latent_dim)

        self.mlp = eval(self.arch)(
            self.latent_dim, seq=self.total_seq_len, num_layers=num_layers
        )
        self.embed_timestep = TimestepEmbeding(self.latent_dim)
        self.sparse_process = nn.Linear(self.sparse_dim, self.latent_dim)
        self.output_process = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, x, timesteps, cond, force_mask=False, y=None, **kwargs):
        """
        x: [batch_size, nframes, nfeats], denoted x_t in the paper
        sparse: [batch_size, nframes, sparse_dim], the sparse features
        motion_ctx: [batch_size, nframes, nfeats], the contextual information
        timesteps: [batch_size, nframes] (int)
        """
        bs, sl = x.shape[:2]
        sparse_emb = cond[DataTypeGT.SPARSE]
        motion_ctx = cond[DataTypeGT.MOTION_CTX]

        # expand sparse with 0's to match self.total_seq_len
        padding = torch.zeros(
            (bs, self.total_seq_len - sparse_emb.shape[1], sparse_emb.shape[2]),
            device=sparse_emb.device,
        )
        sparse_emb = torch.cat([sparse_emb, padding], dim=1)

        # concat motion ctx with motion
        x = torch.cat([motion_ctx, x], dim=1)

        if not self.framewise_time_emb:
            # average per batch element
            timesteps = timesteps.float().mean(axis=-1).round().long()
            assert (
                len(timesteps.shape) == 1
            ), f"timesteps should be [batch_size] and are {timesteps.shape}"
        else:
            t_padding = torch.zeros(
                (bs, self.rolling_motion_ctx),
                device=sparse_emb.device,
            )
            timesteps = (
                torch.cat(
                    [t_padding, timesteps],
                    dim=1,
                )
                .reshape(-1)
                .long()
            )
        emb = self.embed_timestep(timesteps)  # time step embedding : [bs, 1/sl, d]
        emb = (
            emb.squeeze().reshape(bs, self.total_seq_len, -1)
            if self.framewise_time_emb
            else emb
        )

        # Pass the sparse signal to a FC
        sparse_emb = self.sparse_process(
            self.mask_cond_fn(sparse_emb, self.training, force_mask=force_mask)
        )

        # Pass the input to a FC
        x = self.input_process(x)

        # Concat the sparse feature with input
        x = torch.cat((sparse_emb, x), axis=-1)
        output = self.mlp(x, emb)

        # Pass the output to a FC and reshape the output
        output = self.output_process(output)
        return {
            ModelOutputType.RELATIVE_ROTS: output[:, motion_ctx.shape[1] :],
            ModelOutputType.SHAPE_PARAMS: None,
        }


class TimestepEmbeding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, timesteps):
        return self.pe[timesteps]
