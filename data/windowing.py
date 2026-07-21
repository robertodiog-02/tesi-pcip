"""
Protocollo di finestratura PIE — UNICA FONTE DI VERITA'
=======================================================

La logica che decide QUALI finestre di 16 frame finiscono nel dataset vive
qui, cosi' build_samples, la visualizzazione degli scheletri e l'estrattore
di feature visive usano tutti lo stesso identico protocollo.

Protocollo (Kotseruba et al., WACV 2021):
  - osservazione: obs_len = 16 frame (~0.5 s)
  - time-to-event: TTE in [30, 60] frame (~1-2 s) prima del crossing point
  - finestre campionate con overlap 0.6 -> step = round(16 * 0.4) = 6
  - servono almeno obs_len + tte_min frame di traccia

Nota: nelle sequenze PIE generate con seq_type="crossing" la traccia e' gia'
troncata al crossing point, quindi l'evento e' l'ULTIMO frame e il TTE di una
finestra che finisce in w_end e' semplicemente len(track) - w_end.
"""

from typing import Iterator, List, Tuple

OBS_LEN = 16
TTE_MIN = 30
TTE_MAX = 60
OVERLAP = 0.6


def window_step(obs_len: int = OBS_LEN, overlap: float = OVERLAP) -> int:
    """Passo tra finestre consecutive."""
    return max(1, round(obs_len * (1.0 - overlap)))


def iter_windows(
    track_len: int,
    obs_len: int = OBS_LEN,
    overlap: float = OVERLAP,
    tte_min: int = TTE_MIN,
    tte_max: int = TTE_MAX,
) -> Iterator[Tuple[int, int, int]]:
    """
    Genera le finestre valide per una traccia di lunghezza track_len.

    Yields:
        (w_start, w_end, tte) — indici nella traccia, tte = track_len - w_end
    """
    if track_len < obs_len + tte_min:
        return

    start_idx = max(0, track_len - obs_len - tte_max)
    end_idx = track_len - obs_len - tte_min
    if end_idx < 0 or end_idx < start_idx:
        return

    step = window_step(obs_len, overlap)
    for w_start in range(start_idx, end_idx + 1, step):
        w_end = w_start + obs_len
        if w_end <= track_len:
            yield w_start, w_end, track_len - w_end


def window_starts(track_len: int, **kw) -> List[int]:
    """Solo gli indici di inizio delle finestre valide."""
    return [s for s, _, _ in iter_windows(track_len, **kw)]
