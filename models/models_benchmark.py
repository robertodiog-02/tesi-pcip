"""
BenchmarkSingleRNN — replica esatta del SingleRNN di PedestrianActionBenchmark
==============================================================================
Corrisponde a questo modello Keras:

  input_box   (None, 15, 4)  -.
                               |- concatenate (None,15,5) - GRU(256) - Dense(1)
  input_speed (None, 15, 1)  -'

  - Concatenate degli input sull'asse feature (early fusion)
  - UNA GRU singola (256), ritorna solo l'ultimo hidden state
  - NIENTE proiezione d'ingresso, NIENTE attention, NIENTE decoder
  - Output = 1 logit (usare BCEWithLogitsLoss)
  - ~202k parametri (201984 GRU + 257 Dense)

forward(bbox, bbox_delta, ego_speed) resta identico al tuo BaselineGRU,
cosi' funziona col tuo train.py e collate_fn.
"""

import torch
import torch.nn as nn


class BenchmarkSingleRNN(nn.Module):
    def __init__(
        self,
        hidden_dim:     int  = 256,
        num_layers:     int  = 1,
        dropout:        float = 0.0,
        use_bbox:       bool = True,
        use_bbox_displacement: bool = False,
        use_bbox_delta: bool = False,
        use_ego_speed:  bool = True,
    ):
        super().__init__()
        self.use_bbox = use_bbox
        self.use_bbox_displacement = use_bbox_displacement
        self.use_bbox_delta = use_bbox_delta
        self.use_ego_speed  = use_ego_speed
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = (4 if use_bbox else 0) + (4 if use_bbox_displacement else 0) + (4 if use_bbox_delta else 0) + (1 if use_ego_speed else 0)
        self.input_dim = input_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, 1)   # Dense(1)

    def forward(self, bbox, bbox_displacement=None, bbox_delta=None, ego_speed=None):
        feats = []
        if self.use_bbox:
            feats.append(bbox)
        if self.use_bbox_displacement:
            feats.append(bbox_displacement)
        if self.use_bbox_delta:
            feats.append(bbox_delta)
        if self.use_ego_speed:
            feats.append(ego_speed)
        x = torch.cat(feats, dim=-1)          # [B, T, input_dim]

        _, h_n = self.gru(x)                  # h_n: [num_layers, B, hidden]
        last = h_n[-1]                        # [B, hidden]
        logit = self.output(last)            # [B, 1]
        return logit
