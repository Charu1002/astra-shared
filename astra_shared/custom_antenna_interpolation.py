"""Deterministic interpolation helpers for custom antenna gain lookup (Step 5.1)."""

from __future__ import annotations

import bisect
import math
from typing import Any

import numpy as np


def lookup_custom_gain_db(
    psi_deg: float,
    phi_deg: float,
    payload: dict[str, Any] | None,
    *,
    phi_policy: str = "wrap_360",
    out_of_range: str = "clamp",
) -> float:
    """Return interpolated custom gain for the requested (psi, phi) in dBi.

    Deterministic behavior:
    - Invalid payload -> 0.0
    - Exact grid hit -> exact value
    - Default interpolation -> bilinear
    - Degenerate grids / degenerate cells -> nearest-neighbor fallback
    - Psi handling -> clamp [0, 180]
    - Phi handling -> policy controlled (`wrap_360`, `wrap_180`, `clamp`)
    """
    if not _is_payload_valid(payload):
        return 0.0

    if payload is None:
        return 0.0
    psi_grid = [float(v) for v in payload.get("psi_deg", [])]
    phi_grid = [float(v) for v in payload.get("phi_deg", [])]
    gain = payload.get("gain_dbi", [])

    if not psi_grid or not phi_grid or not gain:
        return 0.0

    psi_q = _clamp_psi(psi_deg)
    phi_q = _normalize_phi(
        phi_deg, phi_grid, policy=phi_policy, out_of_range=out_of_range
    )

    i0, i1 = _bracket_indices(psi_grid, psi_q)
    j0, j1, phi_q = _bracket_phi_indices(phi_grid, phi_q, phi_policy)

    # Exact hit fast path
    if i0 == i1 and j0 == j1:
        return float(gain[i0][j0])

    # A genuinely one-dimensional grid has no bilinear cell, so retain the
    # documented nearest-neighbour fallback.  An exact hit on one axis of a
    # normal 2-D grid must still interpolate along the other axis.
    if len(psi_grid) == 1 or len(phi_grid) == 1:
        return _nearest_gain(psi_q, phi_q, psi_grid, phi_grid, gain)

    x0, x1 = psi_grid[i0], psi_grid[i1]
    y0, y1 = phi_grid[j0], phi_grid[j1]
    if j0 == len(phi_grid) - 1 and j1 == 0:
        # The final interpolation cell crosses the 360 -> 0 periodic seam.
        y1 += 360.0
    q11 = float(gain[i0][j0])
    q21 = float(gain[i1][j0])
    q12 = float(gain[i0][j1])
    q22 = float(gain[i1][j1])

    tx = 0.0 if x1 == x0 else (psi_q - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (phi_q - y0) / (y1 - y0)

    return (
        q11 * (1.0 - tx) * (1.0 - ty)
        + q21 * tx * (1.0 - ty)
        + q12 * (1.0 - tx) * ty
        + q22 * tx * ty
    )


def lookup_custom_gain_db_array(
    psi_deg: Any,
    phi_deg: Any,
    payload: dict[str, Any] | None,
    *,
    phi_policy: str = "wrap_360",
    out_of_range: str = "clamp",
) -> np.ndarray:
    """Vectorized bilinear custom gain lookup for NumPy-shaped inputs."""
    psi_arr, phi_arr = np.broadcast_arrays(
        np.asarray(psi_deg, dtype=np.float64),
        np.asarray(phi_deg, dtype=np.float64),
    )
    if not _is_payload_valid(payload) or payload is None:
        return np.zeros_like(psi_arr, dtype=np.float64)

    psi_grid = np.asarray(payload.get("psi_deg", []), dtype=np.float64)
    phi_grid = np.asarray(payload.get("phi_deg", []), dtype=np.float64)
    gain = np.asarray(payload.get("gain_dbi", []), dtype=np.float64)
    if (
        psi_grid.size == 0
        or phi_grid.size == 0
        or gain.ndim != 2
        or gain.shape != (psi_grid.size, phi_grid.size)
    ):
        return np.zeros_like(psi_arr, dtype=np.float64)

    psi_q = np.nan_to_num(psi_arr, nan=0.0, posinf=180.0, neginf=0.0)
    psi_q = np.clip(psi_q, float(psi_grid[0]), float(psi_grid[-1]))
    phi_q = _normalize_phi_array(
        phi_arr, phi_grid, policy=phi_policy, out_of_range=out_of_range
    )

    i1 = np.searchsorted(psi_grid, psi_q, side="left")
    j1 = np.searchsorted(phi_grid, phi_q, side="left")
    i1 = np.clip(i1, 0, psi_grid.size - 1)
    j1 = np.clip(j1, 0, phi_grid.size - 1)
    i0 = np.maximum(i1 - 1, 0)
    j0 = np.maximum(j1 - 1, 0)
    exact_i = psi_grid[i1] == psi_q
    exact_j = phi_grid[j1] == phi_q
    i0 = np.where(exact_i, i1, i0)
    j0 = np.where(exact_j, j1, j0)

    periodic_phi = _uses_periodic_phi_cell(phi_grid, phi_policy)
    if periodic_phi:
        raw_j1 = np.searchsorted(phi_grid, phi_q, side="left")
        seam = (~exact_j) & ((raw_j1 == 0) | (raw_j1 == phi_grid.size))
        j0 = np.where(seam, phi_grid.size - 1, j0)
        j1 = np.where(seam, 0, j1)
        phi_q = np.where((~exact_j) & (raw_j1 == 0), phi_q + 360.0, phi_q)

    x0 = psi_grid[i0]
    x1 = psi_grid[i1]
    y0 = phi_grid[j0]
    y1 = phi_grid[j1]
    if periodic_phi:
        y1 = np.where((j0 == phi_grid.size - 1) & (j1 == 0), y1 + 360.0, y1)
    dx = x1 - x0
    dy = y1 - y0
    tx = np.divide(psi_q - x0, dx, out=np.zeros_like(psi_q), where=dx != 0)
    ty = np.divide(phi_q - y0, dy, out=np.zeros_like(phi_q), where=dy != 0)

    q11 = gain[i0, j0]
    q21 = gain[i1, j0]
    q12 = gain[i0, j1]
    q22 = gain[i1, j1]
    interpolated = (
        q11 * (1.0 - tx) * (1.0 - ty)
        + q21 * tx * (1.0 - ty)
        + q12 * (1.0 - tx) * ty
        + q22 * tx * ty
    )

    degenerate = np.full(psi_q.shape, psi_grid.size == 1 or phi_grid.size == 1)
    if np.any(degenerate):
        nearest_i = np.where(np.abs(psi_q - x0) <= np.abs(psi_q - x1), i0, i1)
        nearest_j = np.where(np.abs(phi_q - y0) <= np.abs(phi_q - y1), j0, j1)
        interpolated = np.where(degenerate, gain[nearest_i, nearest_j], interpolated)
    return interpolated.astype(np.float64, copy=False)


def _is_payload_valid(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    validation = payload.get("validation")
    if not isinstance(validation, dict):
        return False
    return validation.get("is_valid") is True


def _clamp_psi(psi_deg: float) -> float:
    try:
        x = float(psi_deg)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(0.0, min(180.0, x))


def _normalize_phi(
    phi_deg: float, phi_grid: list[float], *, policy: str, out_of_range: str
) -> float:
    if not phi_grid:
        return 0.0
    try:
        x = float(phi_deg)
    except (TypeError, ValueError):
        x = phi_grid[0]

    if not math.isfinite(x):
        x = phi_grid[0]

    p = (policy or "wrap_360").strip().lower()
    if p == "wrap_180":
        x = ((x + 180.0) % 360.0) - 180.0
    elif p == "clamp":
        pass
    else:
        x = x % 360.0

    if out_of_range == "clamp" and not _uses_periodic_phi_cell(phi_grid, policy):
        if p == "wrap_360" and (x < phi_grid[0] or x > phi_grid[-1]):
            return _nearest_phi_edge(x, phi_grid)
        return max(phi_grid[0], min(phi_grid[-1], x))
    return x


def _normalize_phi_array(
    phi_deg: np.ndarray, phi_grid: np.ndarray, *, policy: str, out_of_range: str
) -> np.ndarray:
    x = np.asarray(phi_deg, dtype=np.float64)
    x = np.nan_to_num(
        x, nan=float(phi_grid[0]), posinf=float(phi_grid[-1]), neginf=float(phi_grid[0])
    )
    p = (policy or "wrap_360").strip().lower()
    if p == "wrap_180":
        x = ((x + 180.0) % 360.0) - 180.0
    elif p != "clamp":
        x = x % 360.0
    if out_of_range == "clamp" and not _uses_periodic_phi_cell(phi_grid, policy):
        if p == "wrap_360":
            outside = (x < phi_grid[0]) | (x > phi_grid[-1])
            x = np.where(outside, _nearest_phi_edge_array(x, phi_grid), x)
        else:
            x = np.clip(x, float(phi_grid[0]), float(phi_grid[-1]))
    return x


def _uses_periodic_phi_cell(phi_grid: Any, policy: str) -> bool:
    """Whether the missing circular gap is no larger than the largest phi step."""
    if (policy or "wrap_360").strip().lower() != "wrap_360" or len(phi_grid) < 2:
        return False
    steps = np.diff(np.asarray(phi_grid, dtype=np.float64))
    gap = 360.0 - float(phi_grid[-1]) + float(phi_grid[0])
    return gap <= float(np.max(steps)) + 1e-9


def _nearest_phi_edge(value: float, phi_grid: list[float]) -> float:
    first = phi_grid[0]
    last = phi_grid[-1]
    first_distance = min(abs(value - first), 360.0 - abs(value - first))
    last_distance = min(abs(value - last), 360.0 - abs(value - last))
    return first if first_distance <= last_distance else last


def _nearest_phi_edge_array(value: np.ndarray, phi_grid: np.ndarray) -> np.ndarray:
    first = float(phi_grid[0])
    last = float(phi_grid[-1])
    first_distance = np.minimum(np.abs(value - first), 360.0 - np.abs(value - first))
    last_distance = np.minimum(np.abs(value - last), 360.0 - np.abs(value - last))
    return np.where(first_distance <= last_distance, first, last)


def _bracket_phi_indices(
    grid: list[float], value: float, policy: str
) -> tuple[int, int, float]:
    """Bracket phi, including the final-to-first cell of a periodic grid."""
    if not _uses_periodic_phi_cell(grid, policy):
        i0, i1 = _bracket_indices(grid, value)
        return i0, i1, value

    right = bisect.bisect_left(grid, value)
    if right < len(grid) and grid[right] == value:
        return right, right, value
    if right == 0:
        return len(grid) - 1, 0, value + 360.0
    if right == len(grid):
        return len(grid) - 1, 0, value
    return right - 1, right, value


def _bracket_indices(grid: list[float], value: float) -> tuple[int, int]:
    if len(grid) == 1:
        return 0, 0
    if value <= grid[0]:
        return 0, 0
    if value >= grid[-1]:
        n = len(grid) - 1
        return n, n

    right = bisect.bisect_left(grid, value)
    if right < len(grid) and grid[right] == value:
        return right, right
    left = max(0, right - 1)
    return left, right


def _nearest_gain(
    psi_q: float,
    phi_q: float,
    psi_grid: list[float],
    phi_grid: list[float],
    gain: list[list[float]],
) -> float:
    i = min(range(len(psi_grid)), key=lambda idx: abs(psi_grid[idx] - psi_q))
    j = min(range(len(phi_grid)), key=lambda idx: abs(phi_grid[idx] - phi_q))
    return float(gain[i][j])
