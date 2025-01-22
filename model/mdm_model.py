import numpy as np
import torch
import torch.nn as nn
import utils.constants as Constants
from utils.constants import DataTypeGT, ModelOutputType


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


def inverted_mask(size, dev):
    """Create an upper triangular matrix to mask past tokens."""
    mask = (
        torch.tril(torch.ones(size, size, device=dev))
        - torch.diag(torch.ones(size, device=dev))
    ).bool()
    return mask


class FrameDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        """
        Initializes the FrameDropout module.

        Parameters:
            dropout_prob (float): The probability of dropping a frame. Should be between 0 and 1.
        """
        super(FrameDropout, self).__init__()
        if not (0 <= dropout_prob <= 1):
            raise ValueError("dropout_prob must be between 0 and 1.")
        self.dropout_prob = dropout_prob

    def forward(self, x):
        """
        Applies dropout to whole frames in the sequence.

        Parameters:
            x (torch.Tensor): Input tensor of shape [seqlen, bs, d]

        Returns:
            torch.Tensor: Output tensor with the same shape as input, with some frames dropped.
        """
        if self.training and self.dropout_prob > 0.0:
            # Generate a mask for dropping frames
            frame_mask = torch.rand(x.size(0), device=x.device) > self.dropout_prob
            # Expand mask to cover all batches and features
            frame_mask = frame_mask.unsqueeze(1).unsqueeze(2).expand_as(x)
            return x * frame_mask
        else:
            # During evaluation, return the input without changes
            return x


class RollingMDM(nn.Module):
    """
    Based on original MDM code: https://github.com/GuyTevet/motion-diffusion-model/blob/main/model/mdm.py
    and adapted so that can be used for rolling diffusion (https://arxiv.org/abs/2402.09470)
    """

    def __init__(
        self,
        nfeats,
        mask_cond_fn,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        dropout_framewise=0.0,
        dataset="amass",
        activation="gelu",
        sparse_dim=54,
        **kargs,
    ):
        super().__init__()

        self.nfeats = nfeats
        self.dataset = dataset

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.framedropout = FrameDropout(dropout_framewise)
        self.sparse_dim = sparse_dim

        self.rolling_motion_ctx = kargs.get("rolling_motion_ctx", 0)
        self.rolling_sparse_ctx = kargs.get("rolling_sparse_ctx", 0)
        self.total_seq_len = kargs.get("input_motion_length") + self.rolling_motion_ctx

        self.activation = activation

        self.mask_cond_fn = mask_cond_fn

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.lookahead = kargs.get("lookahead", False)
        self.cond = kargs.get("cond", "concat")
        self.use_timestep_emb = kargs.get("mdm_timestep_emb", False)

        time_emb = 0
        if self.use_timestep_emb:
            self.embed_timestep = TimestepEmbeding(self.latent_dim)
            time_emb = self.latent_dim

        sparse_dim_concat = 0
        if self.cond == "concat":
            assert (
                self.rolling_sparse_ctx <= self.total_seq_len
            ), "sparse context can't be larger than total sequence length for this model with concat conditioning"
            sparse_dim_concat = self.sparse_dim
        elif self.cond == "xatt":
            self.sparse_process = InputProcess(self.sparse_dim, self.latent_dim)
            self.xatt = nn.MultiheadAttention(
                embed_dim=self.latent_dim, num_heads=self.num_heads
            )
            self.xatt_norm = nn.LayerNorm(self.latent_dim)
            self.xatt_ff = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            raise NotImplementedError

        self.input_process = InputProcess(
            self.nfeats + sparse_dim_concat + time_emb, self.latent_dim
        )

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )

        self.output_process = OutputProcess(self.nfeats, self.latent_dim)

        self.inv_mask = None
        if self.lookahead:
            # True are ignored, False are attended
            self.inv_mask = inverted_mask(self.total_seq_len, "cpu")
            # context is attended always
            self.inv_mask[: self.rolling_motion_ctx] = False
            self.inv_mask[:, : self.rolling_motion_ctx] = False

        self.use_shape_head = kargs.get("use_shape_head", False)
        if self.use_shape_head:
            self.shape_enc = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=2)
            self.shape_head = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, Constants.NUM_BETAS_SMPL),
            )

    def forward(
        self, x, timesteps, cond, force_mask=False, y=None, padding_mask=None, **kwargs
    ):
        """
        x: [batch_size, nframes, nfeats], denoted x_t in the paper
        sparse: [batch_size, nframes, sparse_dim], the sparse features
        motion_ctx: [batch_size, nframes, nfeats], the contextual information
        timesteps: [batch_size, nframes] (int)
        """
        bs, sl, nfeats = x.shape
        sparse_emb = cond[DataTypeGT.SPARSE]
        motion_ctx = cond[DataTypeGT.MOTION_CTX]

        # concat motion ctx with motion
        x = torch.cat([motion_ctx, x], dim=1)

        sparse_emb = self.mask_cond_fn(sparse_emb, self.training, force_mask=force_mask)
        if self.cond == "concat":
            # expand sparse with 0's to match self.total_seq_len
            padding = torch.zeros(
                (bs, self.total_seq_len - sparse_emb.shape[1], sparse_emb.shape[2]),
                device=sparse_emb.device,
            )
            # mask sparse feats
            sparse_emb = torch.cat([sparse_emb, padding], dim=1)

            x = torch.cat([x, sparse_emb], dim=2)

        if self.use_timestep_emb:
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
            timestep_emb = self.embed_timestep(timesteps)
            timestep_emb = timestep_emb.squeeze().reshape(bs, self.total_seq_len, -1)
            x = torch.cat([x, timestep_emb], dim=2)

        xseq = self.input_process(x)  # --> [seqlen, bs, d]
        xseq = self.framedropout(xseq)
        # adding the positional embed
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]

        if self.cond == "xatt":
            sparse_emb = self.sparse_process(sparse_emb)
            sparse_emb = self.sequence_pos_encoder(sparse_emb)
            xseq_xatt = self.xatt(
                xseq,
                sparse_emb,
                sparse_emb,
            )[0]
            xseq = self.xatt_norm(xseq_xatt + xseq)

        if padding_mask is not None and self.rolling_motion_ctx > 0:
            t_padding = torch.zeros(
                (bs, self.rolling_motion_ctx),
                device=padding_mask.device,
            )
            padding_mask = torch.cat([t_padding, padding_mask], dim=1)

        output_feats = self.seqTransEncoder(
            xseq,
            mask=self.inv_mask.to(xseq.device) if self.inv_mask is not None else None,
            src_key_padding_mask=padding_mask,
        )  # [seqlen, bs, d]

        output = self.output_process(output_feats)  # --> [bs, seqlen, nfeats]
        out_dict = {
            ModelOutputType.RELATIVE_ROTS: output[:, motion_ctx.shape[1] :],
        }
        if self.use_shape_head:
            feats = self.shape_enc(output_feats).mean(axis=0)  # mean across seqlen
            out_dict[ModelOutputType.SHAPE_PARAMS] = self.shape_head(feats)
        else:
            out_dict[ModelOutputType.SHAPE_PARAMS] = None
        return out_dict



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.nfeats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        bs, nframes, nfeats = x.shape
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)  # --> [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        nframes, bs, nfeats = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, seqlen, nfeats]
        return output
