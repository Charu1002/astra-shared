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
    """Vectorized form of ``lookup_custom_gain_db``: the same value for every query.

    It follows the scalar rules one for one, so a batched caller gets exactly the scalar
    result on every accepted grid, including repeated grid values and one-dimensional
    tables: a non-finite psi reads as 0 and a non-finite phi as the first phi sample; at
    or beyond an end of the psi grid (or a non-periodic phi grid) the end row is used,
    the last of any repeated end values; an exact hit is the first matching sample; the
    periodic phi cell crosses the 360 -> 0 seam; a one-dimensional grid takes the nearest
    sample (the first on a tie), measured as the scalar path measures it.
    """
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

    psi_q = np.where(np.isfinite(psi_arr), np.clip(psi_arr, 0.0, 180.0), 0.0)
    phi_q = _normalize_phi_array(
        phi_arr, phi_grid, policy=phi_policy, out_of_range=out_of_range
    )

    i0, i1 = _bracket_indices_array(psi_grid, psi_q)
    j0, j1, phi_q = _bracket_phi_indices_array(phi_grid, phi_q, phi_policy)
    exact = (i0 == i1) & (j0 == j1)

    if psi_grid.size == 1 or phi_grid.size == 1:
        nearest = gain[_nearest_index_array(psi_grid, psi_q), _nearest_index_array(phi_grid, phi_q)]
        return np.where(exact, gain[i0, j0], nearest).astype(np.float64, copy=False)

    x0 = psi_grid[i0]
    x1 = psi_grid[i1]
    y0 = phi_grid[j0]
    y1 = np.where((j0 == phi_grid.size - 1) & (j1 == 0), phi_grid[j1] + 360.0, phi_grid[j1])
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
    return np.where(exact, q11, interpolated).astype(np.float64, copy=False)


def _bracket_indices_array(grid: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``_bracket_indices`` for an array of values."""
    n = grid.size - 1
    if grid.size == 1:
        zeros = np.zeros(value.shape, dtype=np.intp)
        return zeros, zeros
    right = np.searchsorted(grid, value, side="left")
    right_c = np.clip(right, 0, n)
    exact = (right < grid.size) & (grid[right_c] == value)
    i0 = np.where(exact, right_c, np.maximum(right_c - 1, 0))
    i1 = right_c
    high = value >= grid[-1]
    i0 = np.where(high, n, i0)
    i1 = np.where(high, n, i1)
    low = value <= grid[0]  # checked first by the scalar rule
    return np.where(low, 0, i0), np.where(low, 0, i1)


def _bracket_phi_indices_array(
    grid: np.ndarray, value: np.ndarray, policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``_bracket_phi_indices`` for an array of values."""
    if not _uses_periodic_phi_cell(grid, policy):
        j0, j1 = _bracket_indices_array(grid, value)
        return j0, j1, value
    n = grid.size - 1
    right = np.searchsorted(grid, value, side="left")
    right_c = np.clip(right, 0, n)
    exact = (right < grid.size) & (grid[right_c] == value)
    before = (~exact) & (right == 0)
    after = (~exact) & (right == grid.size)
    j0 = np.where(exact, right_c, np.where(before | after, n, right_c - 1))
    j1 = np.where(exact, right_c, np.where(before | after, 0, right_c))
    return j0, j1, np.where(before, value + 360.0, value)


def _nearest_index_array(grid: np.ndarray, value: np.ndarray) -> np.ndarray:
    """``_nearest_gain``'s index on a sorted grid: the first index of the nearest sample."""
    if grid.size == 1:
        return np.zeros(value.shape, dtype=np.intp)
    k = np.searchsorted(grid, value, side="left")
    lo = np.clip(k - 1, 0, grid.size - 1)
    hi = np.clip(k, 0, grid.size - 1)
    pick = np.where(np.abs(grid[lo] - value) <= np.abs(grid[hi] - value), lo, hi)
    return np.searchsorted(grid, grid[pick], side="left")


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
    x = np.where(np.isfinite(x), x, float(phi_grid[0]))  # as the scalar path: the first sample
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
