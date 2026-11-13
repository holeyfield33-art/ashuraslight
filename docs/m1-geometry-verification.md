# M1 Geometry Service — Verification Evidence

**Date:** 2026-11-13  
**Scope:** Tier-1 geometry contract + GeometryService adapter only.  
**Public submission repo:** `EnkiDaGoat/ashuraslight`  
**Working fork:** `holeyfield33-art/ashuraslight`

## Sampling convention

Window timestamps use a **fixed step from start, inclusive endpoint**:

- Samples are placed at `start + k * step` for `k = 0, 1, …` while the stamp is strictly before the window end.
- The **final sample is always the exact window end**.
- If `duration` is not an integer multiple of `step`, the last interval is shorter than `step`.

Example: 24-hour window, 7-hour step → stamps at 0, 7, 14, 21, **24** h (5 samples).

Maximum `duration_days` is **30**.

## Timezone policy

- Naive datetimes are rejected.
- Non-UTC offsets are rejected (offset must be exactly zero).
- Only timezone-aware UTC is accepted.

## Test suite (current)

| Suite | File | Requires kernels | Role |
|-------|------|------------------|------|
| Existing ephemeris | `tests/test_ephemeris.py` | Yes | CSPICE oracle + kernel integrity |
| Pure contract | `tests/test_geometry_contract.py` | No | Physics + status matrix |
| Service integration | `tests/test_geometry_service.py` | Yes | GeometryService instant/window |

**Do not treat historical counts from older reports as current evidence.** Re-run the suite and record the live result.

## Kernel hashes (manifest)

| File | SHA-256 |
|------|---------|
| `de421.bsp` | `08b20db2ae22488650641c5a9033e5bfda4b1c4b440cfeaf20f621cfa18ecdb3` |
| `moon_080317.tf` | `78732477b96f9863e7b0d65bcee3c22b8707ca5ed0db56d1173319cb2e8c7993` |
| `pck00011.tpc` | `3dff7b1dbeceaa01f25467767d3fa25816051c85d162d1edf04acb310ee28bb1` |
| `moon_pa_de421_1900-2050.bpc` | `656f90616403d75a75f0cd6c8830fc5b44f8cb4facb5ccb8915e752b397520cf` |

## Scientific limitations (unchanged)

- Approximate terrain mask only (no real LOLA DEM)
- Horizontal incidence is a geometric cosine × disk-fraction proxy, not array power
- Earth visibility is elevation-based, not a link budget
- Sub-solar / sub-Earth vectors are `None` (not computed)
- Status labels are first-order planning heuristics, not flight rules
