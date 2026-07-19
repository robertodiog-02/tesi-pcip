"""
Modelli per PIE PCIP
====================

Gerarchia:
  BaselineGRU     - lower bound: solo bbox + ego_speed → GRU → FC
  BranchA_GCN     - Branch A: skeleton GCN (richiede pose cache)
  BranchB_GAT     - Branch B: scene GAT (richiede detection objects)
  CrossAttnDualGraph - architettura completa con CGAM
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ─── Baseline GRU ────────────────────────────────────────────────────────────

class BaselineGRU(nn.Module):
    """
    Baseline GRU.

    Feature di input modulari, decise tramite flag:
        bbox             : 4 dim   se use_bbox=True
        bbox_displacement: +4 dim  se use_bbox_displacement=True
        bbox_delta       : +4 dim  se use_bbox_delta=True
        ego_speed        : +1 dim  se use_ego_speed=True

    input_dim effettivo viene calcolato automaticamente in base ai flag.
    """

    def __init__(
        self,
        hidden_dim:     int   = 256,
        num_layers:     int   = 2,
        dropout:        float = 0.0,
        use_bbox:       bool  = True,
        use_bbox_displacement: bool = False,
        use_bbox_delta: bool  = True,
        use_ego_speed:  bool  = False,
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


        self.decoder = nn.Linear(hidden_dim, 2)
        

    def forward(
        self,
        bbox:              torch.Tensor,            # [B, T, 4]
        bbox_displacement: torch.Tensor = None,     # [B, T, 4]
        bbox_delta:        torch.Tensor = None,     # [B, T, 4]
        ego_speed:         torch.Tensor = None,     # [B, T, 1]
    ) -> torch.Tensor:                       # [B, 2]

        features = []
        if self.use_bbox:
            features.append(bbox)
        if self.use_bbox_displacement:
            features.append(bbox_displacement)
        if self.use_bbox_delta:
            features.append(bbox_delta)
        if self.use_ego_speed:
            features.append(ego_speed)
        x = torch.cat(features, dim=-1)      # [B, T, input_dim]

        
        gru_out, _ = self.gru(x)



        logits = self.decoder(gru_out[:, -1, :])  # [B, 2]
        return logits


class TransformerModalityNet(nn.Module):
    """
    Transformer Modality Network.

    FUSIONE — due modalita', controllate da `separate_encoder_speed_kinematics`:

    False (default, comportamento originale — FUSIONE PIATTA):
        tutte le feature (bbox, displacement, delta, speed) vengono proiettate
        singolarmente, concatenate e riproiettate insieme:
            x = FC_joint( Concat[ FC_box(b), FC_disp(d), FC_delta(v), FC_speed(s) ] )

    True (stile GTransPDM — FUSIONE GERARCHICA A DUE LIVELLI):
        Livello 1 — due encoder separati per gruppo semantico:
            X_pe = FC_pe( Concat[ FC_box(b), FC_disp(d), FC_delta(v) ] )   (eq. 3)
            X_ev = FC_ev( Concat[ FC_speed(s) ] )                          (eq. 5)
        Livello 2 — fusione tra encoder:
            x    = FC_fusion( Concat[ X_pe, X_ev ] )                       (eq. 8)

        La cinematica del pedone e il moto dell'ego-veicolo restano separati
        finche' non sono entrambi nello stesso spazio di rappresentazione.

    POOLING — come si collassa la dimensione temporale prima della testa:
        "cls"     : token CLS appreso (stile BERT/ViT)
        "last"    : ultimo token = frame piu' vicino all'evento
        "mean"    : global average pooling su tutti i token
        "flatten" : concatena tutti i T token -> FC   (GTransPDM eq. 9)

    NOTA su "flatten": la testa dipende da T fisso, quindi va passato
    `obs_len` corretto (16, oppure 15 con drop_first_frame).

    STABILITA' — due opzioni indipendenti, entrambe disattivate di default:

    `norm_first` (Pre-LN): sposta la LayerNorm PRIMA di attention e FFN dentro
        ogni layer del Transformer, invece che dopo (Post-LN, default PyTorch).
        Pre-LN e' generalmente piu' stabile su dataset piccoli e reti
        addestrate da zero, perche' il segnale residuo passa senza essere
        normalizzato e i gradienti si propagano meglio.

    `use_input_layernorm`: applica una LayerNorm alla sequenza subito prima
        del Transformer (dopo il positional encoding). Utile quando le feature
        in ingresso hanno scale molto diverse tra loro (es. bbox in [0,1],
        displacement in pixel, speed in km/h): normalizza le attivazioni e
        rende la scala dell'input irrilevante.
    """

    def __init__(
        self,
        hidden_dim:     int   = 128,          # d_model
        num_layers:     int   = 4,
        nhead:          int   = 8,
        dropout:        float = 0.1,
        pooling:        str   = "cls",        # "cls" | "last" | "mean" | "flatten"
        max_len:        int   = 512,
        obs_len:        int   = 16,           # serve solo a pooling="flatten"
        separate_encoder_speed_kinematics: bool = False,
        norm_first:     bool  = False,        # Pre-LN dentro il Transformer
        use_input_layernorm: bool = False,    # LayerNorm prima del Transformer
        use_bbox:       bool  = True,
        use_bbox_displacement: bool = False,
        use_bbox_delta: bool  = False,
        use_ego_speed:  bool  = True,
    ):
        super().__init__()
        assert pooling in ("cls", "last", "mean", "flatten"), \
            f"pooling non valido: {pooling}"

        self.d_model = hidden_dim
        self.pooling = pooling
        self.obs_len = obs_len
        self.separate_encoder_speed_kinematics = separate_encoder_speed_kinematics
        self.norm_first = norm_first
        self.use_input_layernorm = use_input_layernorm
        self.use_bbox = use_bbox
        self.use_bbox_displacement = use_bbox_displacement
        self.use_bbox_delta = use_bbox_delta
        self.use_ego_speed = use_ego_speed

        # 1. Proiezioni lineari separate (una per feature, pesi distinti)
        self.projections = nn.ModuleDict()
        if use_bbox:
            self.projections["box"] = nn.Linear(4, hidden_dim)
        if use_bbox_displacement:
            self.projections["box_displacement"] = nn.Linear(4, hidden_dim)
        if use_bbox_delta:
            self.projections["box_delta"] = nn.Linear(4, hidden_dim)
        if use_ego_speed:
            self.projections["speed"] = nn.Linear(1, hidden_dim)

        n_modalities = len(self.projections)
        assert n_modalities > 0, "Almeno una modalita' di input deve essere attiva!"

        # 2. Fusione
        if separate_encoder_speed_kinematics:
            # Livello 1: encoder separati
            n_kin = sum([use_bbox, use_bbox_displacement, use_bbox_delta])
            self.n_kin = n_kin

            # ramo cinematico (GTransPDM eq. 3)
            self.proj_kinematics = (
                nn.Linear(n_kin * hidden_dim, hidden_dim) if n_kin > 0 else None
            )
            # ramo ego-motion (GTransPDM eq. 5)
            self.proj_ego = nn.Linear(hidden_dim, hidden_dim) if use_ego_speed else None

            # Livello 2: fusione tra encoder (GTransPDM eq. 8)
            n_branches = int(n_kin > 0) + int(use_ego_speed)
            self.proj_fusion = nn.Linear(n_branches * hidden_dim, hidden_dim)
            self.proj_joint = None
        else:
            # Fusione piatta (comportamento originale)
            self.proj_joint = nn.Linear(n_modalities * hidden_dim, hidden_dim)
            self.proj_kinematics = None
            self.proj_ego = None
            self.proj_fusion = None

        # 3. CLS token
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)

        # 4. Positional Encoding (apprendibile)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # 4b. LayerNorm opzionale prima del Transformer
        self.input_ln = nn.LayerNorm(hidden_dim) if use_input_layernorm else None

        # 5. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 6. Classification Head (1 logit per BCEWithLogitsLoss)
        if pooling == "flatten":
            self.decoder = nn.Linear(obs_len * hidden_dim, 1)
        else:
            self.decoder = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        bbox:              torch.Tensor,
        bbox_displacement: torch.Tensor = None,
        bbox_delta:        torch.Tensor = None,
        ego_speed:         torch.Tensor = None,
    ) -> torch.Tensor:
        B, T = bbox.shape[0], bbox.shape[1]

        if self.separate_encoder_speed_kinematics:
            # ── Livello 1a: ramo cinematico (bbox / displacement / delta) ──
            kin_feats = []
            if self.use_bbox:
                kin_feats.append(self.projections["box"](bbox))
            if self.use_bbox_displacement:
                kin_feats.append(self.projections["box_displacement"](bbox_displacement))
            if self.use_bbox_delta:
                kin_feats.append(self.projections["box_delta"](bbox_delta))

            branches = []
            if kin_feats:
                x_pe = self.proj_kinematics(torch.cat(kin_feats, dim=-1))
                branches.append(x_pe)                       # [B, T, d_model]

            # ── Livello 1b: ramo ego-motion (speed), percorso separato ──
            if self.use_ego_speed:
                x_ev = self.proj_ego(self.projections["speed"](ego_speed))
                branches.append(x_ev)                       # [B, T, d_model]

            # ── Livello 2: fusione tra encoder ──
            x = self.proj_fusion(torch.cat(branches, dim=-1))  # [B, T, d_model]
        else:
            # ── Fusione piatta: tutto insieme ──
            proj_feats = []
            if self.use_bbox:
                proj_feats.append(self.projections["box"](bbox))
            if self.use_bbox_displacement:
                proj_feats.append(self.projections["box_displacement"](bbox_displacement))
            if self.use_bbox_delta:
                proj_feats.append(self.projections["box_delta"](bbox_delta))
            if self.use_ego_speed:
                proj_feats.append(self.projections["speed"](ego_speed))

            x = torch.cat(proj_feats, dim=-1)   # [B, T, n_modalities * d_model]
            x = self.proj_joint(x)              # [B, T, d_model]

        # ── Positional encoding, poi CLS ──
        x = x + self.pos_embedding[:, :x.size(1), :]

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)   # [B, T+1, d_model]

        # ── LayerNorm opzionale prima del Transformer ──
        if self.input_ln is not None:
            x = self.input_ln(x)

        # ── Transformer Encoder ──
        x = self.transformer_encoder(x)     # [B, seq_len, d_model]

        # ── Pooling ──
        if self.pooling == "cls":
            feat = x[:, 0]                  # [B, d_model]
        elif self.pooling == "last":
            feat = x[:, -1]                 # [B, d_model]
        elif self.pooling == "mean":
            feat = x.mean(dim=1)            # [B, d_model]
        else:  # flatten (GTransPDM eq. 9)
            feat = x.reshape(B, -1)         # [B, T * d_model]

        # ── Decode logit ──
        logit = self.decoder(feat)          # [B, 1]
        return logit
