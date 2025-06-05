# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the CC BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import utils.constants as Constants
from utils.constants import DataTypeGT, ModelOutputType



class RollingMDM(nn.Module):
    """
    Based MDM architecture, adapted so that can be used as a rolling prediction model
    """

    def __init__(
        self,
        nfeats,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
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
        self.sparse_dim = sparse_dim

        self.rolling_motion_ctx = kargs.get("rolling_motion_ctx", 0)
        self.rolling_sparse_ctx = kargs.get("rolling_sparse_ctx", 0)
        self.total_seq_len = kargs.get("input_motion_length") + self.rolling_motion_ctx
        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        time_emb = 0
        self.sparse_process = InputProcess(self.sparse_dim, self.latent_dim)
        self.xatt = nn.MultiheadAttention(
            embed_dim=self.latent_dim, num_heads=self.num_heads
        )
        self.xatt_norm = nn.LayerNorm(self.latent_dim)
        self.xatt_ff = nn.Linear(self.latent_dim, self.latent_dim)

        self.input_process = InputProcess(
            self.nfeats + time_emb, self.latent_dim
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

    def forward(
        self, x, cond, **kwargs
    ):
        """
        x: [batch_size, nframes, nfeats], denoted x_t in the paper
        cond[DataTypeGT.SPARSE]: [batch_size, nframes_sparse, sparse_dim], the sparse features
        cond[DataTypeGT.MOTION_CTX]: [batch_size, nframes_ctx, nfeats], the motion context feats
        """
        sparse_emb = cond[DataTypeGT.SPARSE]
        motion_ctx = cond[DataTypeGT.MOTION_CTX]

        # concat motion ctx with motion
        x = torch.cat([motion_ctx, x], dim=1)

        xseq = self.input_process(x)  # --> [seqlen, bs, d]
        # adding the positional embed
        xseq = self.sequence_pos_encoder(xseq)  # [seqlen, bs, d]

        sparse_emb = self.sparse_process(sparse_emb)
        sparse_emb = self.sequence_pos_encoder(sparse_emb)
        xseq_xatt = self.xatt(
            xseq,
            sparse_emb,
            sparse_emb,
        )[0]
        xseq = self.xatt_norm(xseq_xatt + xseq)

        output_feats = self.seqTransEncoder(xseq)  # [seqlen, bs, d]

        output = self.output_process(output_feats)  # --> [bs, seqlen, nfeats]
        return {
            ModelOutputType.RELATIVE_ROTS: output[:, motion_ctx.shape[1] :],
            # logic is ready to handle shape params prediction, but not used in the current training
            ModelOutputType.SHAPE_PARAMS: None,
        }



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
        x = x.permute((1, 0, 2))
        x = self.poseEmbedding(x)  # --> [seqlen, bs, d]
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        output = self.poseFinal(output)  # [seqlen, bs, nfeats]
        output = output.permute(1, 0, 2)  # [bs, seqlen, nfeats]
        return output
