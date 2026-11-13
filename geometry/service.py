"""Geometry service adapter: LunarGeometry → corrected Tier-1 contract.

Does not modify the Skyfield calculation path. Adds mask, incidence,
visibility classification, window aggregates, and planning status.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ephemeris import LunarGeometry, GeometryInputError
from sites import Site

from geometry.contract import (
    InstantGeometry,
    InstantRequest,
    Limitations,
    Provenance,
    WindowRequest,
    WindowSample,
    WindowSummary,
)
from geometry.physics import (
    classify_earth_visibility,
    decide_planning_status,
    first_state_change_hours,
    horizontal_incidence_pct,
    horizontal_incidence_proxy,
    solar_disk_fraction,
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _default_provenance() -> Provenance:
    return Provenance(
        method="spice_de421_skyfield",
        frame="MOON_ME_DE421",
        kernels_or_source="de421.bsp + moon_080317.tf + pck00011.tpc + moon_pa_de421_1900-2050.bpc",
        notes=(
            "Geometric topocentric center directions. No light-time, "
            "stellar aberration, or refraction. Approximate mask is user-supplied."
        ),
        generated_at_utc=_utcnow_iso(),
    )


def _default_limitations(mask_deg: float) -> Limitations:
    # Tier 1 always uses an approximate (or zero) mask, never a real DEM.
    return Limitations(no_terrain_dem=True)


def _site_from_request(
    latitude_deg: float,
    longitude_deg: float,
    site_id: Optional[str] = None,
) -> Site:
    """Build a minimal Site for the existing calculate_site path."""
    from sites import Site as SiteCls

    return SiteCls(
        site_id=site_id or "custom",
        name=site_id or "Custom",
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        latitude_convention="planetocentric",
        longitude_convention="signed east-positive degrees",
        reference_frame="MOON_ME_DE421",
        elevation_m=0.0,
        coordinate_interpretation=(
            "User or curated coordinates interpreted as planetocentric MOON_ME_DE421."
        ),
        source_url="local://custom",
        source_coordinate_text=f"{latitude_deg:.6f}, {longitude_deg:.6f}",
    )


class GeometryService:
    """Thin adapter over LunarGeometry that emits the frozen contract."""

    def __init__(self, kernel_dir: str | Path):
        self._geom = LunarGeometry(kernel_dir)

    def close(self) -> None:
        self._geom.close()

    @staticmethod
    def _require_utc(dt: datetime, field_name: str) -> datetime:
        """Reject naive timestamps and non-UTC offsets. Accept only UTC-aware values."""
        if dt.tzinfo is None:
            raise GeometryInputError(f"{field_name} must be timezone-aware UTC")
        offset = dt.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise GeometryInputError(
                f"{field_name} must use UTC (offset 0); got offset {offset}"
            )
        return dt

    def instant(self, request: InstantRequest) -> InstantGeometry:
        epoch = self._require_utc(request.epoch_utc, "epoch_utc")
        if epoch is not request.epoch_utc:
            request = InstantRequest(
                latitude_deg=request.latitude_deg,
                longitude_deg=request.longitude_deg,
                epoch_utc=epoch,
                horizon_mask_deg=request.horizon_mask_deg,
                site_id=request.site_id,
            )
        site = _site_from_request(
            request.latitude_deg, request.longitude_deg, request.site_id
        )
        series = self._geom.calculate_site(site, (request.epoch_utc,))
        mask = max(0.0, float(request.horizon_mask_deg))

        sun_el = series.sun_elevation_deg[0]
        sun_az = series.sun_azimuth_deg[0]
        earth_el = series.earth_elevation_deg[0]
        earth_az = series.earth_azimuth_deg[0]

        sun_clear = sun_el - mask
        earth_clear = earth_el - mask

        earth_vis, earth_reason = classify_earth_visibility(earth_el, earth_clear)
        f_disk = solar_disk_fraction(sun_clear)
        i_proxy = horizontal_incidence_proxy(sun_el, sun_clear)
        i_pct = 100.0 * i_proxy

        # Sub-solar / sub-Earth are not computed by the current Skyfield path.
        # Explicit None prevents zero placeholders from being treated as geometry.
        return InstantGeometry(
            request=request,
            sun_elevation_deg=sun_el,
            sun_azimuth_deg=sun_az,
            earth_elevation_deg=earth_el,
            earth_azimuth_deg=earth_az,
            sun_terrain_mask_deg=mask,
            earth_terrain_mask_deg=mask,
            sun_clearance_deg=sun_clear,
            earth_clearance_deg=earth_clear,
            geometric_sun_visible=sun_el > 0.0,
            geometric_earth_visible=earth_el > 0.0,
            approx_sun_clear=sun_clear > 0.0,
            approx_earth_clear=earth_clear > 0.0,
            solar_disk_visible=sun_clear > -0.266,
            solar_disk_fully_clear=sun_clear > 0.266,
            solar_disk_fraction=f_disk,
            horizontal_incidence_proxy=i_proxy,
            horizontal_incidence_pct=i_pct,
            earth_visibility=earth_vis,
            earth_visibility_reason=earth_reason,
            sub_solar_lat_deg=None,
            sub_solar_lon_deg=None,
            sub_earth_lat_deg=None,
            sub_earth_lon_deg=None,
            provenance=_default_provenance(),
            limitations=_default_limitations(mask),
        )

    def window(self, request: WindowRequest) -> WindowSummary:
        start = self._require_utc(request.start_utc, "start_utc")
        if request.duration_days <= 0 or request.step_hours <= 0:
            raise GeometryInputError("duration_days and step_hours must be positive")
        if request.duration_days > 30.0:
            raise GeometryInputError(
                "duration_days must be <= 30 (Tier-1 maximum analysis window)"
            )

        # Sampling convention (documented):
        # Fixed step from start. The final sample is always the exact window end.
        # If duration is not an integer multiple of step, the last interval is shorter.
        # Example: 24 h window, 7 h step → stamps at 0, 7, 14, 21, 24 h (5 samples).
        total_hours = request.duration_days * 24.0
        step_h = float(request.step_hours)
        timestamps_list: list[datetime] = []
        t = 0.0
        while t < total_hours - 1e-12:
            timestamps_list.append(start + timedelta(hours=t))
            t += step_h
        end_stamp = start + timedelta(hours=total_hours)
        if not timestamps_list or timestamps_list[-1] != end_stamp:
            timestamps_list.append(end_stamp)
        timestamps = tuple(timestamps_list)

        site = _site_from_request(
            request.latitude_deg, request.longitude_deg, request.site_id
        )
        raw = self._geom.calculate_site(site, timestamps)
        mask = max(0.0, float(request.horizon_mask_deg))

        samples: list[WindowSample] = []
        sun_disk_flags: list[bool] = []
        earth_vis_flags: list[bool] = []
        simultaneous_flags: list[bool] = []

        for i, ts in enumerate(timestamps):
            sun_el = raw.sun_elevation_deg[i]
            sun_az = raw.sun_azimuth_deg[i]
            earth_el = raw.earth_elevation_deg[i]
            earth_az = raw.earth_azimuth_deg[i]
            sun_clear = sun_el - mask
            earth_clear = earth_el - mask
            disk_vis = sun_clear > -0.266
            earth_state, _ = classify_earth_visibility(earth_el, earth_clear)
            earth_agg = earth_state in ("VISIBLE", "GRAZING")

            offset_h = (ts - start).total_seconds() / 3600.0
            samples.append(
                WindowSample(
                    offset_hours=offset_h,
                    offset_days=offset_h / 24.0,
                    epoch_utc=ts,
                    sun_elevation_deg=sun_el,
                    sun_azimuth_deg=sun_az,
                    sun_clearance_deg=sun_clear,
                    geometric_sun_visible=sun_el > 0.0,
                    approx_sun_clear=sun_clear > 0.0,
                    solar_disk_visible=disk_vis,
                    horizontal_incidence_pct=horizontal_incidence_pct(sun_el, sun_clear),
                    earth_elevation_deg=earth_el,
                    earth_azimuth_deg=earth_az,
                    earth_clearance_deg=earth_clear,
                    geometric_earth_visible=earth_el > 0.0,
                    approx_earth_clear=earth_clear > 0.0,
                    earth_visible_for_aggregate=earth_agg,
                )
            )
            sun_disk_flags.append(disk_vis)
            earth_vis_flags.append(earth_agg)
            simultaneous_flags.append(disk_vis and earth_agg)

        n = len(samples)
        sunlight_count = sum(sun_disk_flags)
        earth_count = sum(earth_vis_flags)
        simultaneous_count = sum(simultaneous_flags)

        sunlight_pct = 100.0 * sunlight_count / n if n else 0.0
        earth_pct = 100.0 * earth_count / n if n else 0.0
        simultaneous_pct = 100.0 * simultaneous_count / n if n else 0.0

        hours_sun = first_state_change_hours(sun_disk_flags, request.step_hours)
        hours_earth = first_state_change_hours(earth_vis_flags, request.step_hours)

        status, label, reason = decide_planning_status(
            solar_disk_visible_at_t0=sun_disk_flags[0] if sun_disk_flags else False,
            earth_visible_at_t0=earth_vis_flags[0] if earth_vis_flags else False,
            sunlight_pct=sunlight_pct,
            earth_visible_pct=earth_pct,
            hours_to_sun_change=hours_sun,
            hours_to_earth_change=hours_earth,
            data_ok=n > 0,
        )

        sun_els = [s.sun_elevation_deg for s in samples]
        earth_els = [s.earth_elevation_deg for s in samples]

        return WindowSummary(
            request=request,
            series=tuple(samples),
            sunlight_hours=sunlight_pct / 100.0 * total_hours,
            sunlight_pct=sunlight_pct,
            earth_visible_hours=earth_pct / 100.0 * total_hours,
            earth_visible_pct=earth_pct,
            simultaneous_sun_earth_visible_hours=simultaneous_pct / 100.0 * total_hours,
            simultaneous_sun_earth_visible_pct=simultaneous_pct,
            peak_sun_elevation_deg=max(sun_els) if sun_els else 0.0,
            min_sun_elevation_deg=min(sun_els) if sun_els else 0.0,
            peak_earth_elevation_deg=max(earth_els) if earth_els else 0.0,
            min_earth_elevation_deg=min(earth_els) if earth_els else 0.0,
            hours_to_sun_state_change=hours_sun,
            hours_to_earth_state_change=hours_earth,
            status=status,
            status_label=label,
            status_reason=reason,
            provenance=_default_provenance(),
            limitations=_default_limitations(mask),
        )
