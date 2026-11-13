"""Kernel-backed integration tests for GeometryService (Milestone 1).

Requires provisioned kernels in data/kernels/.
Does not weaken existing CSPICE oracle tolerances.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from geometry.contract import InstantRequest, WindowRequest
from geometry.service import GeometryService
from sites import get_site_by_id
from ephemeris import GeometryInputError

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "data" / "kernels"

SPICE_REFERENCE = {
    "DM2-358": {
        "sun": (-3.012459542225528, 219.41923412141995),
        "earth": (1.7962960425281513, 305.94052773137696),
    },
}


@pytest.fixture(scope="module")
def service() -> GeometryService:
    if not (KERNEL_DIR / "de421.bsp").is_file():
        pytest.skip("Kernels not provisioned")
    svc = GeometryService(KERNEL_DIR)
    yield svc
    svc.close()


FIXED_EPOCH = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)


def test_instant_known_site_matches_cspice_oracle(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        horizon_mask_deg=0.0,
        site_id=site.site_id,
    )
    result = service.instant(req)
    expected = SPICE_REFERENCE["DM2-358"]
    assert result.sun_elevation_deg == pytest.approx(expected["sun"][0], abs=1e-6)
    assert result.sun_azimuth_deg == pytest.approx(expected["sun"][1], abs=1e-6)
    assert result.earth_elevation_deg == pytest.approx(expected["earth"][0], abs=1e-6)
    assert result.earth_azimuth_deg == pytest.approx(expected["earth"][1], abs=1e-6)


def test_instant_custom_coordinates_finite(service: GeometryService) -> None:
    req = InstantRequest(
        latitude_deg=-85.0,
        longitude_deg=30.0,
        epoch_utc=FIXED_EPOCH,
        horizon_mask_deg=0.0,
    )
    result = service.instant(req)
    assert -90.0 <= result.sun_elevation_deg <= 90.0
    assert 0.0 <= result.sun_azimuth_deg < 360.0
    assert -90.0 <= result.earth_elevation_deg <= 90.0
    assert 0.0 <= result.earth_azimuth_deg < 360.0


def test_instant_terrain_mask_affects_clearance(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    base = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        horizon_mask_deg=0.0,
        site_id=site.site_id,
    )
    masked = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        horizon_mask_deg=2.0,
        site_id=site.site_id,
    )
    r0 = service.instant(base)
    r2 = service.instant(masked)
    assert r2.sun_clearance_deg == pytest.approx(r0.sun_elevation_deg - 2.0)
    assert r2.earth_clearance_deg == pytest.approx(r0.earth_elevation_deg - 2.0)
    assert r2.sun_terrain_mask_deg == 2.0


def test_instant_horizontal_incidence_bounds(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        horizon_mask_deg=0.0,
        site_id=site.site_id,
    )
    result = service.instant(req)
    assert 0.0 <= result.horizontal_incidence_proxy <= 1.0
    assert 0.0 <= result.horizontal_incidence_pct <= 100.0
    if result.sun_elevation_deg <= 0.0:
        assert result.horizontal_incidence_proxy == pytest.approx(0.0)


def test_instant_provenance_and_limitations(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        site_id=site.site_id,
    )
    result = service.instant(req)
    assert result.provenance.method == "spice_de421_skyfield"
    assert result.provenance.frame == "MOON_ME_DE421"
    assert result.limitations.geometric_only is True
    assert result.limitations.no_power_system_model is True
    assert result.limitations.no_link_budget is True
    assert result.limitations.no_flight_rule_status is True
    assert result.limitations.no_terrain_dem is True


def test_instant_subsolar_fields_are_explicitly_unavailable(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        site_id=site.site_id,
    )
    result = service.instant(req)
    assert result.sub_solar_lat_deg is None
    assert result.sub_solar_lon_deg is None
    assert result.sub_earth_lat_deg is None
    assert result.sub_earth_lon_deg is None


def test_instant_earth_visibility_no_far_side(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = InstantRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        epoch_utc=FIXED_EPOCH,
        site_id=site.site_id,
    )
    result = service.instant(req)
    assert result.earth_visibility in {
        "VISIBLE",
        "GRAZING",
        "TERRAIN_BLOCKED",
        "BELOW_HORIZON",
    }


def test_window_sampling_and_aggregates(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    req = WindowRequest(
        latitude_deg=site.latitude_deg,
        longitude_deg=site.longitude_deg,
        start_utc=FIXED_EPOCH,
        duration_days=2.0,
        step_hours=1.0,
        horizon_mask_deg=0.0,
        site_id=site.site_id,
    )
    summary = service.window(req)
    assert len(summary.series) == 49
    assert summary.series[0].epoch_utc == FIXED_EPOCH
    assert summary.series[-1].offset_hours == pytest.approx(48.0)
    assert 0.0 <= summary.sunlight_pct <= 100.0
    assert 0.0 <= summary.earth_visible_pct <= 100.0
    assert 0.0 <= summary.simultaneous_sun_earth_visible_pct <= 100.0
    assert summary.sunlight_hours == pytest.approx(
        summary.sunlight_pct / 100.0 * 48.0, abs=1e-9
    )
    assert summary.status in {"FAVORABLE", "MARGINAL", "UNFAVORABLE", "DATA_INSUFFICIENT"}
    assert summary.provenance.method == "spice_de421_skyfield"
    assert summary.limitations.no_flight_rule_status is True


def test_window_identical_epochs_for_two_sites(service: GeometryService) -> None:
    a = get_site_by_id("DM2-358")
    b = get_site_by_id("DM1-1121")
    start = FIXED_EPOCH
    wa = service.window(
        WindowRequest(
            latitude_deg=a.latitude_deg,
            longitude_deg=a.longitude_deg,
            start_utc=start,
            duration_days=1.0,
            step_hours=2.0,
            site_id=a.site_id,
        )
    )
    wb = service.window(
        WindowRequest(
            latitude_deg=b.latitude_deg,
            longitude_deg=b.longitude_deg,
            start_utc=start,
            duration_days=1.0,
            step_hours=2.0,
            site_id=b.site_id,
        )
    )
    assert len(wa.series) == len(wb.series)
    for sa, sb in zip(wa.series, wb.series):
        assert sa.epoch_utc == sb.epoch_utc
        assert sa.offset_hours == sb.offset_hours


def test_window_first_state_change_type(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    summary = service.window(
        WindowRequest(
            latitude_deg=site.latitude_deg,
            longitude_deg=site.longitude_deg,
            start_utc=FIXED_EPOCH,
            duration_days=3.0,
            step_hours=1.0,
            site_id=site.site_id,
        )
    )
    if summary.hours_to_sun_state_change is not None:
        assert summary.hours_to_sun_state_change > 0
    if summary.hours_to_earth_state_change is not None:
        assert summary.hours_to_earth_state_change > 0


def test_window_rejects_duration_over_30_days(service: GeometryService) -> None:
    site = get_site_by_id("DM2-358")
    with pytest.raises(GeometryInputError, match="30"):
        service.window(
            WindowRequest(
                latitude_deg=site.latitude_deg,
                longitude_deg=site.longitude_deg,
                start_utc=FIXED_EPOCH,
                duration_days=31.0,
                step_hours=1.0,
                site_id=site.site_id,
            )
        )


def test_window_24h_with_7h_step_includes_endpoint(service: GeometryService) -> None:
    """Regression: 24 h window, 7 h step → 0,7,14,21,24 h (final interval shorter)."""
    site = get_site_by_id("DM2-358")
    summary = service.window(
        WindowRequest(
            latitude_deg=site.latitude_deg,
            longitude_deg=site.longitude_deg,
            start_utc=FIXED_EPOCH,
            duration_days=1.0,
            step_hours=7.0,
            site_id=site.site_id,
        )
    )
    offsets = [s.offset_hours for s in summary.series]
    assert offsets[0] == pytest.approx(0.0)
    assert offsets[-1] == pytest.approx(24.0)
    assert offsets == pytest.approx([0.0, 7.0, 14.0, 21.0, 24.0])
    assert len(summary.series) == 5


def test_instant_rejects_naive_timestamp(service: GeometryService) -> None:
    naive = datetime(2026, 12, 1, 0, 0)
    with pytest.raises(GeometryInputError, match="timezone-aware|UTC"):
        service.instant(
            InstantRequest(latitude_deg=-84.0, longitude_deg=60.0, epoch_utc=naive)
        )


def test_instant_rejects_non_utc_offset(service: GeometryService) -> None:
    non_utc = datetime(2026, 12, 1, 0, 0, tzinfo=timezone(timedelta(hours=-8)))
    with pytest.raises(GeometryInputError, match="UTC|offset"):
        service.instant(
            InstantRequest(latitude_deg=-84.0, longitude_deg=60.0, epoch_utc=non_utc)
        )


def test_window_rejects_non_utc_offset(service: GeometryService) -> None:
    non_utc = datetime(2026, 12, 1, 0, 0, tzinfo=timezone(timedelta(hours=5)))
    with pytest.raises(GeometryInputError, match="UTC|offset"):
        service.window(
            WindowRequest(
                latitude_deg=-84.0,
                longitude_deg=60.0,
                start_utc=non_utc,
                duration_days=1.0,
                step_hours=1.0,
            )
        )
