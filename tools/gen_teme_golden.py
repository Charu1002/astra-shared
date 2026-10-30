"""Generate the golden TEME->ITRF vectors in tests/test_geo_math_teme.py (Issue #19).

Needs Skyfield (not a CI dependency):  python tools/gen_teme_golden.py
Paste the printed dict entries into GOLDEN in tests/test_geo_math_teme.py.
"""
import math

import numpy as np
import skyfield
from sgp4.api import WGS72, Satrec
from skyfield.sgp4lib import TEME_to_ITRF

# Synthetic TLE lines (same format as orbit_propagation.format_tle) at epoch 2026-10-30 06:00:00 UTC.
CASES = {
    "leo53": ("1 00001U 26001A   26303.25000000  .00000000  00000-0  00000-0 0  9990",
              "2 00001  53.0000  40.0000 0000000   0.0000  30.0000 15.05000000    00"),
    "sso97": ("1 00002U 26001A   26303.25000000  .00000000  00000-0  00000-0 0  9990",
              "2 00002  97.6000 120.0000 0000000   0.0000 200.0000 14.90000000    00"),
    "geo0": ("1 00003U 26001A   26303.25000000  .00000000  00000-0  00000-0 0  9990",
             "2 00003   0.0000  75.0000 0000000   0.0000  10.0000  1.00273791    00"),
}
OFFSET_MIN = {"leo53": 37.0, "sso97": 71.0, "geo0": 600.0}

print(f"# skyfield {skyfield.__version__}")
for name, (l1, l2) in CASES.items():
    sat = Satrec.twoline2rv(l1, l2, WGS72)
    jd = sat.jdsatepoch
    fr = sat.jdsatepochF + OFFSET_MIN[name] / 1440.0
    e, r, v = sat.sgp4(jd, fr)
    assert e == 0, (name, e)
    r_itrf, v_itrf = TEME_to_ITRF(jd, np.array(r), np.array(v) * 86400.0, 0.0, 0.0, fr)
    v_itrf = v_itrf / 86400.0
    print(f'    "{name}": {{')
    print(f"        \"jd\": {jd!r}, \"fr\": {fr!r},")
    print(f"        \"r_teme\": ({r[0]!r}, {r[1]!r}, {r[2]!r}),")
    print(f"        \"v_teme\": ({v[0]!r}, {v[1]!r}, {v[2]!r}),")
    print(f"        \"r_ecef\": ({float(r_itrf[0])!r}, {float(r_itrf[1])!r}, {float(r_itrf[2])!r}),")
    print(f"        \"v_ecef\": ({float(v_itrf[0])!r}, {float(v_itrf[1])!r}, {float(v_itrf[2])!r}),")
    print("    },")
