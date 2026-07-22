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
        use_pdm:        bool  = False,
        use_polar:      bool  = False,
        use_ego_speed:  bool  = False,
    ):
        super().__init__()
        self.use_bbox = use_bbox
        self.use_pdm = use_pdm
        self.use_polar = use_polar
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




# ─── Skeleton Pose Encoder (GTransPDM, eq. 6-7) ──────────────────────────────

# Scheletro COCO-17: connessioni anatomiche naturali
_COCO17_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # testa
    (5, 6),                                     # spalle
    (5, 7), (7, 9), (6, 8), (8, 10),           # braccia
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # gambe
]
# Collo derivato (idx 17) — usato sia in OpenPose-18 sia in GTransPDM-20.
# Archi come nella topologia OpenPose: collo <-> naso e collo <-> spalle.
_NECK_EDGES = [(17, 0), (17, 5), (17, 6)]

# Punti extra GTransPDM (17=neck, 18=hip, 19=body center) e i loro archi
_EXTRA_EDGES = _NECK_EDGES + [
    (18, 11), (18, 12),             # hip: anche
    (19, 17), (19, 18),             # body center: neck e hip
]


def _edges_for(n_joints: int):
    """Archi dello scheletro secondo il numero di giunti."""
    e = list(_COCO17_EDGES)
    if n_joints == 18:
        e += _NECK_EDGES            # COCO-17 + collo (topologia OpenPose-18)
    elif n_joints == 20:
        e += _EXTRA_EDGES           # + hip e body center (GTransPDM)
    elif n_joints != 17:
        raise ValueError(f"n_joints non supportato: {n_joints} (17, 18 o 20)")
    return e


def _build_adjacency(n_joints: int) -> torch.Tensor:
    """
    A_hat = D^{-1/2} (A + I) D^{-1/2}   (GTransPDM eq. 6)
    Adiacenza anatomica + self-connections, normalizzazione simmetrica.
    """
    A = torch.zeros(n_joints, n_joints)
    for i, j in _edges_for(n_joints):
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(n_joints)
    deg = A.sum(-1)
    d = deg.pow(-0.5)
    d[torch.isinf(d)] = 0.0
    D = torch.diag(d)
    return D @ A @ D


class SkeletonEncoder(nn.Module):
    """
    Skeleton Pose Encoder di GTransPDM (eq. 6-7), fedele al paper:

        A_hat = D^{-1/2} A D^{-1/2}
        X_i   = BatchNorm(K_i)
        H_0   = sigma( (E_0 (*) A_hat) X_i W_0 )
        H_l   = sigma( (E_l (*) A_hat) H_{l-1} W_l + Conv2D(H_{l-1}) )   l=1,2,3
        X_ke  = FC( Flatten(H_3) )

    dove (*) e' il prodotto di Hadamard ed E_l e' la matrice di edge
    importance APPRENDIBILE di ciascun layer (una per layer, come nel paper:
    "learnable edge importance was also applied"). Il primo blocco non ha
    residuo; i blocchi 1-3 hanno il residuo Conv2D 1x1 esplicito.

    Il Flatten e' sui GIUNTI (N x d_hid), NON sul tempo: l'output e'
    [B, T, C_d] e la dimensione temporale resta intatta per il Transformer.
    """

    def __init__(
        self,
        n_joints: int = 20,
        in_channels: int = 3,        # (x, y, confidence)
        hidden_channels: int = 64,   # d_hid (GTransPDM: 64)
        out_dim: int = 64,           # C_d
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_joints = n_joints
        adj = _build_adjacency(n_joints)
        self.register_buffer("adj", adj)

        # X_i = BatchNorm(K_i) sull'input
        self.input_bn = nn.BatchNorm1d(in_channels * n_joints)

        dims = [in_channels] + [hidden_channels] * n_layers
        self.edge_importance = nn.ParameterList(
            [nn.Parameter(torch.ones_like(adj)) for _ in range(n_layers)]
        )
        self.weights = nn.ModuleList(
            [nn.Linear(dims[l], dims[l + 1], bias=False) for l in range(n_layers)]
        )
        # Residuo Conv2D 1x1 per i layer 1..n-1 (il layer 0 non ce l'ha)
        self.residuals = nn.ModuleList(
            [nn.Conv2d(dims[l], dims[l + 1], kernel_size=1)
             for l in range(1, n_layers)]
        )
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(n_joints * hidden_channels, out_dim)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """keypoints: [B, T, N, 3] -> [B, T, out_dim]"""
        B, T, N, C = keypoints.shape

        # BatchNorm sull'input
        x = keypoints.reshape(B, T, N * C).transpose(1, 2)   # [B, N*C, T]
        x = self.input_bn(x).transpose(1, 2).reshape(B, T, N, C)

        for l, (E, W) in enumerate(zip(self.edge_importance, self.weights)):
            adj = self.adj * E                                # E_l (*) A_hat
            h = torch.matmul(adj, x.reshape(B * T, N, -1).view(B, T, N, -1))
            # matmul broadcasting: [N,N] @ [B,T,N,C] -> [B,T,N,C]
            h = W(h)
            if l == 0:
                x = self.relu(h)                              # H_0: no residuo
            else:
                # Conv2D(H_{l-1}): [B, C, T, N]
                res = self.residuals[l - 1](
                    x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                x = self.relu(h + res)

        # Flatten sui giunti, FC -> [B, T, C_d]  (eq. 7)
        return self.fc_out(x.reshape(B, T, -1))




class AttentionPooling(nn.Module):
    """
    Attention pooling: media pesata dei token, con pesi appresi dal contenuto.

        score_t = w^T tanh(W h_t + b)
        alpha   = softmax(score)
        out     = sum_t alpha_t * h_t

    A differenza di "mean" (che pesa tutti i frame uguale) e di "last"/"cls"
    (che ne privilegiano uno fisso), qui il modello impara QUALI frame contano
    per la decisione — e i pesi sono ispezionabili, utile per capire su quale
    parte della finestra si concentra.
    """

    def __init__(self, d_model: int, hidden: int = None):
        super().__init__()
        hidden = hidden or d_model
        self.proj = nn.Linear(d_model, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """x: [B, T, D] -> [B, D]"""
        a = self.score(torch.tanh(self.proj(x)))      # [B, T, 1]
        w = torch.softmax(a, dim=1)                   # [B, T, 1]
        out = (w * x).sum(dim=1)                      # [B, D]
        return (out, w.squeeze(-1)) if return_weights else out


def _build_head(in_dim: int, out_dim: int, n_layers: int,
                hidden: int, dropout: float) -> nn.Module:
    """
    Testa di classificazione: singolo Linear (n_layers=1) o MLP.

    Con pooling="flatten" l'input e' T*d_model (es. 16*64=1024): un solo
    Linear 1024->1 e' una regressione lineare sui token concatenati, mentre
    un MLP puo' modellare interazioni tra istanti diversi. Con gli altri
    pooling l'input e' d_model e il singolo layer di solito basta.
    """
    if n_layers <= 1:
        return nn.Linear(in_dim, out_dim)
    layers = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)




# ─── Pose-STGAT (Dual-STGAT, Lian et al. 2025) ───────────────────────────────

def _build_partitioned_adjacency(n_joints: int,
                                 ref_joints=(11, 12)) -> torch.Tensor:
    """
    Partizionamento a configurazione spaziale (STGCN / Dual-STGAT eq. 19).

    L'adiacenza viene divisa in TRE matrici invece di una sola:
        A_0 : il nodo radice stesso (self-loop)
        A_1 : gruppo CENTRIPETO  — vicini piu' VICINI al baricentro
        A_2 : gruppo CENTRIFUGO  — vicini piu' LONTANI dal baricentro

    L'idea: un movimento verso il centro del corpo e uno verso l'esterno
    hanno significato diverso, quindi meritano pesi diversi. Con l'adiacenza
    unica di GTransPDM questa distinzione si perde.

    Il baricentro e' la media delle coordinate dei giunti; qui lo
    approssimiamo topologicamente con la distanza sul grafo dai giunti di
    riferimento (le anche), che e' stabile e non dipende dai dati.

    Returns: [3, N, N] gia' normalizzate D^-1/2 A D^-1/2
    """
    import collections

    # adiacenza binaria
    adj = torch.zeros(n_joints, n_joints)
    for i, j in _edges_for(n_joints):
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    # distanza sul grafo dal baricentro (BFS dai giunti di riferimento)
    INF = 10 ** 6
    dist = [INF] * n_joints
    q = collections.deque()
    for r in ref_joints:
        if r < n_joints:
            dist[r] = 0
            q.append(r)
    while q:
        u = q.popleft()
        for v in range(n_joints):
            if adj[u, v] > 0 and dist[v] == INF:
                dist[v] = dist[u] + 1
                q.append(v)

    A = torch.zeros(3, n_joints, n_joints)
    for i in range(n_joints):
        A[0, i, i] = 1.0                       # A_0: root
        for j in range(n_joints):
            if adj[i, j] > 0:
                if dist[j] < dist[i]:
                    A[1, i, j] = 1.0           # A_1: centripeto
                else:
                    A[2, i, j] = 1.0           # A_2: centrifugo

    # normalizzazione simmetrica di ciascuna partizione
    for k in range(3):
        deg = A[k].sum(-1)
        d = deg.pow(-0.5)
        d[torch.isinf(d)] = 0.0
        D = torch.diag(d)
        A[k] = D @ A[k] @ D
    return A


class VelocityAttention(nn.Module):
    """
    Velocity Attention di Dual-STGAT (sez. III-D.3), stile Squeeze-and-Excitation.

    La velocita' dell'ego-veicolo NON entra come ramo separato ma modula
    channel-wise le feature della posa:
        squeeze   : global average pooling sull'output del GCN -> [B, C]
        excite    : la velocita' viene proiettata a C canali
        sigmoid   : pesi in [0, 1]
        weighting : F <- F * w

    Concettualmente: quanto conta un certo canale della posa dipende da
    quanto va veloce il veicolo. Un movimento del busto significa una cosa
    diversa se il veicolo e' fermo o lanciato.
    """

    def __init__(self, channels: int, speed_dim: int = 1, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels + speed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor,
                speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, N, C], speed: [B, T, 1] -> [B, T, N, C]"""
        ctx = x.mean(dim=2)                                # squeeze: [B, T, C]
        if speed is not None:
            ctx = torch.cat([ctx, speed], dim=-1)
        else:
            ctx = torch.cat([ctx, torch.zeros_like(ctx[..., :1])], dim=-1)
        w = self.fc(ctx).unsqueeze(2)                      # [B, T, 1, C]
        return x * w


class STGATLayer(nn.Module):
    """
    Un layer STGAT: GCN partizionato -> Velocity Attention -> TCN.
    (Dual-STGAT, Fig. 4)
    """

    def __init__(self, in_ch: int, out_ch: int, adj: torch.Tensor,
                 speed_dim: int = 1, kt: int = 9, dropout: float = 0.0,
                 use_velocity: bool = True, residual: bool = True):
        super().__init__()
        self.register_buffer("adj", adj)                   # [3, N, N]
        self.n_part = adj.shape[0]

        # una matrice di pesi per ciascuna partizione (eq. 20)
        self.weights = nn.ModuleList(
            [nn.Linear(in_ch, out_ch, bias=False) for _ in range(self.n_part)])
        self.edge_importance = nn.Parameter(torch.ones_like(adj))
        self.bn_gcn = nn.BatchNorm2d(out_ch)

        self.vel_attn = (VelocityAttention(out_ch, speed_dim)
                         if use_velocity else None)

        # TCN: conv 1D lungo il tempo, kernel kt x 1
        pad = ((kt - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=(kt, 1), padding=pad),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(dropout),
        )

        if not residual:
            self.residual = None
        elif in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor,
                speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, N, C_in] -> [B, T, N, C_out]"""
        res = 0.0
        if self.residual is not None:
            res = (x if isinstance(self.residual, nn.Identity)
                   else self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        # GCN partizionato: somma sui 3 gruppi, pesi distinti (eq. 20)
        adj = self.adj * self.edge_importance
        out = 0.0
        for k in range(self.n_part):
            out = out + self.weights[k](torch.matmul(adj[k], x))
        out = self.bn_gcn(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = self.relu(out)

        # Velocity Attention (channel-wise)
        if self.vel_attn is not None:
            out = self.vel_attn(out, speed)

        # TCN lungo il tempo
        out = self.tcn(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.relu(out + res)


class PoseSTGAT(nn.Module):
    """
    Pose-STGAT di Dual-STGAT (Lian et al., TITS 2025).

    Differenze rispetto allo Skeleton Pose Encoder di GTransPDM:

      1. ADIACENZA PARTIZIONATA in 3 gruppi (root / centripeto / centrifugo)
         con pesi distinti, invece di un'unica adiacenza.
      2. VELOCITY ATTENTION: la velocita' dell'ego-veicolo modula channel-wise
         le feature della posa DENTRO l'encoder, invece di restare un ramo
         separato che si fonde dopo.
      3. TCN esplicito lungo il tempo (kernel kt x 1) invece del solo residuo
         Conv2D.

    Il paper usa 3 layer con canali 32, 64, 64 (Tabella IV: P=3 e' l'ottimo).
    """

    def __init__(
        self,
        n_joints: int = 20,
        in_channels: int = 3,
        out_dim: int = 64,
        channels=(32, 64, 64),
        kt: int = 9,
        speed_dim: int = 1,
        dropout: float = 0.0,
        use_velocity: bool = True,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.use_velocity = use_velocity
        adj = _build_partitioned_adjacency(n_joints)

        self.input_bn = nn.BatchNorm1d(in_channels * n_joints)
        dims = [in_channels] + list(channels)
        self.layers = nn.ModuleList([
            STGATLayer(dims[i], dims[i + 1], adj, speed_dim=speed_dim, kt=kt,
                       dropout=dropout, use_velocity=use_velocity,
                       residual=(i > 0))
            for i in range(len(channels))
        ])
        self.fc_out = nn.Linear(n_joints * channels[-1], out_dim)

    def forward(self, keypoints: torch.Tensor,
                speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """keypoints: [B,T,N,3], speed: [B,T,1] -> [B, T, out_dim]"""
        B, T, N, C = keypoints.shape
        x = keypoints.reshape(B, T, N * C).transpose(1, 2)
        x = self.input_bn(x).transpose(1, 2).reshape(B, T, N, C)

        for layer in self.layers:
            x = layer(x, speed)

        return self.fc_out(x.reshape(B, T, -1))


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
        "cls"       : token CLS appreso (stile BERT/ViT)
        "last"      : ultimo token = frame piu' vicino all'evento
        "mean"      : global average pooling su tutti i token
        "flatten"   : concatena tutti i T token -> testa (GTransPDM eq. 9)
        "attention" : media pesata dei token, pesi appresi dal contenuto

    TESTA — `head_layers=1` da' un singolo Linear (comportamento originale);
    >1 costruisce un MLP con ReLU e dropout. Utile soprattutto con "flatten",
    dove l'input e' T*d_model e un solo Linear e' una regressione lineare sui
    token concatenati.

    NOTA su "flatten": la testa dipende da T fisso, quindi va passato
    `obs_len` corretto (16, oppure 15 con drop_first_frame).

    POSIZIONE ASSOLUTA — due rappresentazioni ALTERNATIVE (al massimo una;
    nessuna delle due e' obbligatoria):
        `use_bbox`  : coordinate cartesiane della bbox [x1,y1,x2,y2]
        `use_polar` : coordinate quasi-polari rispetto a tre poli (PCIP) —
                      (cos theta, dist) per ciascun polo, senza selezione
                      dinamica (che introdurrebbe discontinuita' temporali)

    Sono alternative perche' codificano LA STESSA informazione — la posizione
    del pedone nel piano immagine — in due sistemi di coordinate diversi.
    Il tri-polare e' una riparametrizzazione biunivoca: da tre distanze si
    ricostruisce (x, y) per trilaterazione. Usarle insieme sarebbe ridondante,
    ma il modello puo' benissimo girare senza nessuna delle due (es. solo
    ego_speed, o solo displacement + delta): l'unico requisito e' che almeno
    una feature qualsiasi sia attiva.

    MOTO RELATIVO ALLA SCENA — feature aggiuntiva, indipendente:
        `use_pdm`   : Position Decoupling Module (GTransPDM eq. 2)

    DIMENSIONI VARIABILI: pdm_dim, polar_dim, displacement_dim e delta_dim
    dipendono da come il dataset ha costruito le feature. train.py li calcola
    dalla config, cosi' modello e dataset restano allineati.

    Il PDM NON e' alternativo ai due precedenti: e' complementare. Non codifica
    dove si trova il pedone (le differenze temporali eliminano la posizione
    assoluta), ma come si sta muovendo RISPETTO ALLE LINEE che delimitano la
    ROI di attraversamento — piu' l'area ratio come proxy di profondita', che
    nessun'altra feature cattura. Lo stesso spostamento in pixel ha significato
    opposto a seconda del lato in cui avviene, e bbox_delta non lo distingue.
    In GTransPDM il PDM convive infatti con displacement e velocita'.

    `use_bbox_displacement`, `use_bbox_delta` e `use_ego_speed` sono
    indipendenti e restano attivabili con qualunque combinazione.

    SKELETON — `use_skeleton` attiva il ramo posa; `skeleton_encoder` sceglie
    quale architettura usare:

      "gtranspdm" (GTransPDM eq. 6-7): 4 blocchi GCN con adiacenza anatomica
          unica, edge importance apprendibile e residui Conv2D.

      "stgat" (Dual-STGAT, Lian et al. 2025): adiacenza PARTIZIONATA in tre
          gruppi (root / centripeto / centrifugo) con pesi distinti, VELOCITY
          ATTENTION che fa modulare le feature della posa dalla velocita'
          dell'ego-veicolo channel-wise, e TCN esplicito lungo il tempo.
          Nota: con questo encoder la velocita' entra DUE volte — dentro la
          posa e come ramo suo — il che e' voluto: nel paper originale la
          velocita' non ha un ramo separato, qui puoi averli entrambi o
          disattivare use_ego_speed per restare fedele a Dual-STGAT.

    In entrambi i casi l'output e' X_ke [B,T,C_d] e con la fusione gerarchica
    e' il terzo ramo dell'eq. 8: X = FC(Concat[X_pe, X_ev, X_ke]).

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
        pooling:        str   = "cls",        # cls|last|mean|flatten|attention
        head_layers:    int   = 1,            # 1 = Linear singolo, >1 = MLP
        head_hidden:    int   = None,         # dim nascosta MLP (default: hidden_dim)
        head_dropout:   float = 0.1,
        max_len:        int   = 512,
        obs_len:        int   = 16,           # serve solo a pooling="flatten"
        separate_encoder_speed_kinematics: bool = False,
        norm_first:     bool  = False,        # Pre-LN dentro il Transformer
        use_input_layernorm: bool = False,    # LayerNorm prima del Transformer
        use_bbox:       bool  = True,
        use_pdm:        bool  = False,
        use_polar:      bool  = False,
        pdm_dim:        int   = 3,            # 2 base, +1 area_ratio, +2 keep_absolute
        polar_dim:      int   = 6,            # 6 base, 9 con sin, x2 con deltas
        displacement_dim: int = 4,            # 4 corners/center_size, 2 center
        delta_dim:      int   = 4,
        use_bbox_displacement: bool = False,
        use_bbox_delta: bool  = False,
        use_ego_speed:  bool  = True,
        use_skeleton:   bool  = False,
        skeleton_encoder: str = "gtranspdm",   # "gtranspdm" | "stgat"
        skeleton_n_joints: int = 20,       # 20 = 17 COCO + neck/hip/center (GTransPDM)
        skeleton_hidden: int  = 64,        # d_hid (solo gtranspdm)
        skeleton_layers: int  = 4,         # n. blocchi (solo gtranspdm)
        stgat_channels        = (32, 64, 64),  # canali dei layer STGAT (paper: 3 layer)
        stgat_kt:       int   = 9,         # kernel temporale del TCN
        stgat_velocity: bool  = True,      # velocity attention (SE con ego-speed)
    ):
        super().__init__()
        assert pooling in ("cls", "last", "mean", "flatten", "attention"), \
            f"pooling non valido: {pooling}"

        # use_bbox e use_polar sono ALTERNATIVI (al massimo uno dei due):
        # codificano la stessa informazione — la posizione assoluta nel piano
        # immagine — in due sistemi di coordinate diversi, quindi usarli
        # insieme sarebbe ridondante. Nessuno dei due e' pero' obbligatorio:
        # il modello puo' funzionare anche senza posizione assoluta (es. solo
        # ego_speed, o solo displacement + delta).
        # use_pdm e' INDIPENDENTE da entrambi.
        assert not (use_bbox and use_polar), (
            "use_bbox e use_polar sono alternativi: codificano la stessa "
            "informazione (posizione assoluta) in sistemi di coordinate "
            "diversi. Attivane al massimo uno."
        )

        self.d_model = hidden_dim
        self.pooling = pooling
        self.obs_len = obs_len
        self.separate_encoder_speed_kinematics = separate_encoder_speed_kinematics
        self.norm_first = norm_first
        self.use_input_layernorm = use_input_layernorm
        self.use_bbox = use_bbox
        self.use_pdm = use_pdm
        self.use_polar = use_polar
        self.use_bbox_displacement = use_bbox_displacement
        self.use_bbox_delta = use_bbox_delta
        self.use_ego_speed = use_ego_speed
        self.use_skeleton = use_skeleton

        # 1. Proiezioni lineari separate (una per feature, pesi distinti)
        self.projections = nn.ModuleDict()
        if use_bbox:
            self.projections["box"] = nn.Linear(4, hidden_dim)
        if use_pdm:
            self.projections["pdm"] = nn.Linear(pdm_dim, hidden_dim)
        if use_polar:
            self.projections["polar"] = nn.Linear(polar_dim, hidden_dim)
        if use_bbox_displacement:
            self.projections["box_displacement"] = nn.Linear(displacement_dim, hidden_dim)
        if use_bbox_delta:
            self.projections["box_delta"] = nn.Linear(delta_dim, hidden_dim)
        if use_ego_speed:
            self.projections["speed"] = nn.Linear(1, hidden_dim)

        # Skeleton Pose Encoder: ramo a parte, il suo output X_ke [B,T,C_d]
        # entra nella fusione come una modalita'.
        #   "gtranspdm" -> GCN residui + edge importance (GTransPDM eq. 6-7)
        #   "stgat"     -> Pose-STGAT (Dual-STGAT): adiacenza partizionata in
        #                  3 gruppi, velocity attention, TCN
        assert skeleton_encoder in ("gtranspdm", "stgat"), \
            f"skeleton_encoder non valido: {skeleton_encoder}"
        self.skeleton_encoder_type = skeleton_encoder
        self.skeleton_uses_speed = (use_skeleton and skeleton_encoder == "stgat"
                                    and stgat_velocity)

        if not use_skeleton:
            self.skeleton_encoder = None
        elif skeleton_encoder == "stgat":
            self.skeleton_encoder = PoseSTGAT(
                n_joints=skeleton_n_joints,
                out_dim=hidden_dim,
                channels=tuple(stgat_channels),
                kt=stgat_kt,
                dropout=dropout,
                use_velocity=stgat_velocity,
            )
        else:
            self.skeleton_encoder = SkeletonEncoder(
                n_joints=skeleton_n_joints,
                hidden_channels=skeleton_hidden,
                out_dim=hidden_dim,
                n_layers=skeleton_layers,
            )

        n_modalities = len(self.projections) + int(use_skeleton)
        assert n_modalities > 0, (
            "Almeno una feature di input deve essere attiva (use_bbox, "
            "use_polar, use_pdm, use_bbox_displacement, use_bbox_delta o "
            "use_ego_speed)."
        )

        # 2. Fusione
        if separate_encoder_speed_kinematics:
            # Livello 1: encoder separati
            n_kin = sum([use_bbox, use_pdm, use_polar,
                         use_bbox_displacement, use_bbox_delta])
            self.n_kin = n_kin

            # Ramo cinematico (GTransPDM eq. 3). Come sopra: con una sola
            # feature attiva la proiezione di fusione e' ridondante.
            self.proj_kinematics = (
                nn.Linear(n_kin * hidden_dim, hidden_dim) if n_kin > 1 else None
            )
            # Ramo ego-motion (GTransPDM eq. 5): X_ev = FC(Concat[FC(S), FC(Acc)]).
            # Con UNA SOLA feature (solo speed, senza accelerazione) la FC
            # esterna diventa una seconda lineare in fila sulla stessa
            # grandezza scalare: ridondante e peggiora il condizionamento.
            # In quel caso la saltiamo.
            self.n_ego = int(use_ego_speed)
            self.proj_ego = (nn.Linear(hidden_dim, hidden_dim)
                             if self.n_ego > 1 else None)

            # Livello 2: fusione tra encoder (GTransPDM eq. 8)
            # X = FC(Concat[X_pe, X_ev, X_ke])
            n_branches = int(n_kin > 0) + int(use_ego_speed) + int(use_skeleton)
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

        # 5b. Attention pooling (solo se pooling="attention")
        self.attn_pool = AttentionPooling(hidden_dim) if pooling == "attention" else None

        # 6. Classification Head (1 logit per BCEWithLogitsLoss)
        head_in = obs_len * hidden_dim if pooling == "flatten" else hidden_dim
        self.decoder = _build_head(
            head_in, 1, head_layers, head_hidden or hidden_dim, head_dropout)

    def _encode_skeleton(self, keypoints, ego_speed):
        """Applica l'encoder posa; STGAT riceve anche la velocita'."""
        if self.skeleton_encoder_type == "stgat":
            return self.skeleton_encoder(keypoints, ego_speed)
        return self.skeleton_encoder(keypoints)

    def forward(
        self,
        bbox:              torch.Tensor,
        bbox_displacement: torch.Tensor = None,
        bbox_delta:        torch.Tensor = None,
        ego_speed:         torch.Tensor = None,
        pdm:               torch.Tensor = None,
        polar:             torch.Tensor = None,
        keypoints:         torch.Tensor = None,   # [B, T, N, 3]
    ) -> torch.Tensor:
        B, T = bbox.shape[0], bbox.shape[1]

        if self.separate_encoder_speed_kinematics:
            # ── Livello 1a: ramo cinematico (bbox / displacement / delta) ──
            kin_feats = []
            if self.use_bbox:
                kin_feats.append(self.projections["box"](bbox))
            if self.use_pdm:
                kin_feats.append(self.projections["pdm"](pdm))
            if self.use_polar:
                kin_feats.append(self.projections["polar"](polar))
            if self.use_bbox_displacement:
                kin_feats.append(self.projections["box_displacement"](bbox_displacement))
            if self.use_bbox_delta:
                kin_feats.append(self.projections["box_delta"](bbox_delta))

            branches = []
            if kin_feats:
                x_pe = torch.cat(kin_feats, dim=-1)
                if self.proj_kinematics is not None:
                    x_pe = self.proj_kinematics(x_pe)
                branches.append(x_pe)                       # [B, T, d_model]

            # ── Livello 1b: ramo ego-motion (speed), percorso separato ──
            if self.use_ego_speed:
                x_ev = self.projections["speed"](ego_speed)
                if self.proj_ego is not None:
                    x_ev = self.proj_ego(x_ev)
                branches.append(x_ev)                       # [B, T, d_model]

            # ── Livello 1c: ramo skeleton (X_ke) ──
            # Con STGAT la velocita' entra DENTRO l'encoder (velocity
            # attention), non solo come ramo separato.
            if self.use_skeleton:
                branches.append(self._encode_skeleton(keypoints, ego_speed))

            # ── Livello 2: fusione tra encoder (eq. 8) ──
            x = self.proj_fusion(torch.cat(branches, dim=-1))  # [B, T, d_model]
        else:
            # ── Fusione piatta: tutto insieme ──
            proj_feats = []
            if self.use_bbox:
                proj_feats.append(self.projections["box"](bbox))
            if self.use_pdm:
                proj_feats.append(self.projections["pdm"](pdm))
            if self.use_polar:
                proj_feats.append(self.projections["polar"](polar))
            if self.use_bbox_displacement:
                proj_feats.append(self.projections["box_displacement"](bbox_displacement))
            if self.use_bbox_delta:
                proj_feats.append(self.projections["box_delta"](bbox_delta))
            if self.use_ego_speed:
                proj_feats.append(self.projections["speed"](ego_speed))

            if self.use_skeleton:
                proj_feats.append(self._encode_skeleton(keypoints, ego_speed))

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
        elif self.pooling == "attention":
            feat = self.attn_pool(x)        # [B, d_model], pesi appresi
        else:  # flatten (GTransPDM eq. 9)
            feat = x.reshape(B, -1)         # [B, T * d_model]

        # ── Decode logit ──
        logit = self.decoder(feat)          # [B, 1]
        return logit
