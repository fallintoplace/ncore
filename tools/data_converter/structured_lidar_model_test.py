# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from typing import Literal, cast

import numpy as np

from ncore.impl.data import util as data_util
from tools.data_converter.structured_lidar_model import (
    HDL32E_ELEVATIONS_RAD,
    HDL32E_FIRING_PAIR_INTERVAL_US,
    HDL32E_N_BEAMS,
    HDL32E_N_COLUMNS,
    HDL32E_SCAN_DURATION_US,
    AlignedFrameData,
    ColumnAlignment,
    _smooth_sparse_correction,
    assign_model_columns,
    compute_column_alignment,
    compute_frame_timestamps,
    compute_intra_column_firing_offsets,
    compute_model_consistency,
    derive_model_from_decompensated,
    derive_nominal_hdl32e,
    enforce_spinning_monotonic,
    extract_column_azimuths,
    optimize_model,
    upsample_model,
)


class TestStructuredLidarModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = derive_nominal_hdl32e()

    # --- compute_column_alignment tests ----------------------------------------

    def test_compute_column_alignment_exact_match(self) -> None:
        """Exact match: spin azimuths identical to model -> shift=0, near-zero error."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        spin_azimuths = model_azimuths.copy()

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.spin_column_range.start, alignment.static_column_range.start)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    def test_compute_column_alignment_with_shift(self) -> None:
        """Spin is a subset of model starting 5 columns in."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        spin_azimuths = model_azimuths[5:95].copy()

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.static_column_range.start, 5)
        self.assertEqual(alignment.spin_column_range.start, 0)
        self.assertEqual(len(alignment.spin_column_range), len(alignment.static_column_range))
        self.assertAlmostEqual(len(alignment.spin_column_range), 90, delta=2)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    def test_compute_column_alignment_fewer_spin_cols(self) -> None:
        """Model has 100 cols, spin has 95 cols offset by 3."""
        n_cols = 100
        step = -2 * np.pi / n_cols
        model_azimuths = np.arange(n_cols, dtype=np.float64) * step
        # Spin starts at column 3, has 95 columns
        spin_azimuths = (3 + np.arange(95, dtype=np.float64)) * step

        alignment = compute_column_alignment(spin_azimuths, model_azimuths)

        self.assertEqual(alignment.static_column_range.start, 3)
        self.assertEqual(alignment.spin_column_range.start, 0)
        self.assertLess(alignment.mean_alignment_error_rad, 1e-6)

    # --- extract_column_azimuths tests -----------------------------------------

    def test_extract_column_azimuths_synthetic(self) -> None:
        """Verify extraction from synthetic point cloud with known per-column azimuths."""
        n_cols = 20
        n_beams = 4
        n_points = n_cols * n_beams
        r = 30.0  # above default min_range_m=20

        expected_azimuths = np.linspace(0, np.pi, n_cols, endpoint=False)

        xyz = np.zeros((n_points, 3), dtype=np.float64)
        col_idx = np.zeros(n_points, dtype=np.int64)

        for c in range(n_cols):
            az = expected_azimuths[c]
            for b in range(n_beams):
                idx = c * n_beams + b
                xyz[idx, 0] = np.cos(az) * r
                xyz[idx, 1] = np.sin(az) * r
                xyz[idx, 2] = 0.0
                col_idx[idx] = c

        result = extract_column_azimuths(xyz, col_idx, n_cols, min_range_m=20.0, min_points_per_col=3)

        valid_mask = ~np.isnan(result)
        self.assertTrue(valid_mask.all())
        np.testing.assert_allclose(result, expected_azimuths, atol=0.001)

    # --- assign_model_columns tests --------------------------------------------

    def test_assign_model_columns_native(self) -> None:
        """Native resolution (1x): produces 1:1 mapping from alignment."""
        model = self.model
        n_spin = 1065
        spin_start = 5
        static_start = 3

        # Create spin azimuths matching model positions
        spin_col_azimuths = model.column_azimuths_rad[static_start : static_start + n_spin].astype(np.float64)

        alignment = ColumnAlignment(
            spin_column_range=range(spin_start, spin_start + n_spin),
            static_column_range=range(static_start, static_start + n_spin),
            mean_alignment_error_rad=0.0,
        )

        result = assign_model_columns(spin_col_azimuths, model, alignment, resolution_factor=1)

        # Within the overlap, should be static_start + (c - spin_start) for each c
        expected = np.zeros(len(spin_col_azimuths), dtype=np.int64)
        for c in range(len(spin_col_azimuths)):
            if alignment.spin_column_range.start <= c < alignment.spin_column_range.stop:
                expected[c] = static_start + (c - spin_start)
            elif c < alignment.spin_column_range.start:
                expected[c] = static_start
            else:
                expected[c] = min(
                    static_start + (alignment.spin_column_range.stop - 1 - spin_start),
                    model.n_columns - 1,
                )

        np.testing.assert_array_equal(result, expected)

    def test_assign_model_columns_4x_resolution(self) -> None:
        """4x resolution: picks nearest sub-column, not just coarse position."""
        model_4x = upsample_model(self.model, 4)
        n_spin = 100
        spin_start = 0
        static_start = 10

        alignment = ColumnAlignment(
            spin_column_range=range(spin_start, n_spin),
            static_column_range=range(static_start, static_start + n_spin),
            mean_alignment_error_rad=0.0,
        )

        # Create spin azimuths that are offset by 0.5 native columns from coarse positions
        # Each native column spans 4 model columns in the upsampled model
        model_az = model_4x.column_azimuths_rad.astype(np.float64)
        spin_col_azimuths = np.zeros(n_spin, dtype=np.float64)
        for c in range(n_spin):
            coarse_idx = (static_start + c) * 4
            # Offset by ~2 sub-columns (0.5 native column)
            target_idx = min(coarse_idx + 2, model_4x.n_columns - 1)
            spin_col_azimuths[c] = model_az[target_idx]

        result = assign_model_columns(spin_col_azimuths, model_4x, alignment, resolution_factor=4)

        # Each result should be close to coarse_idx + 2 (the offset sub-column)
        for c in range(n_spin):
            coarse_idx = (static_start + c) * 4
            target_idx = min(coarse_idx + 2, model_4x.n_columns - 1)
            self.assertAlmostEqual(result[c], target_idx, delta=1)

    # --- compute_frame_timestamps tests ----------------------------------------

    def test_compute_frame_timestamps_linearity(self) -> None:
        """Timestamps are linear with column index."""
        model_col = np.array([0, 500, 1000], dtype=np.int64)
        n_model_cols = 1000
        start = 0
        end = 50000

        result = compute_frame_timestamps(model_col, n_model_cols, start, end)

        expected = np.array([0, 25000, 50000], dtype=np.uint64)
        np.testing.assert_array_equal(result, expected)

    def test_compute_frame_timestamps_fencepost(self) -> None:
        """Last column (n-1) gets timestamp < frame_end (not equal)."""
        n = 1085
        model_col = np.array([0, n - 1], dtype=np.int64)
        start = 0
        end = 50000

        result = compute_frame_timestamps(model_col, n, start, end)

        self.assertEqual(result[0], 0)
        # Column n-1 out of n: fraction = (n-1)/n < 1, so timestamp < end
        self.assertLess(result[1], end)
        expected_last = int((n - 1) / n * end)
        self.assertEqual(result[1], expected_last)

    # --- upsample_model tests --------------------------------------------------

    def test_upsample_model_doubles_columns(self) -> None:
        """Upsampling by 2 doubles column count and preserves CW monotonicity."""
        model_2x = upsample_model(self.model, 2)

        self.assertEqual(model_2x.n_columns, HDL32E_N_COLUMNS * 2)
        self.assertEqual(len(model_2x.column_azimuths_rad), HDL32E_N_COLUMNS * 2)

        # CW: strictly decreasing
        diffs = np.diff(model_2x.column_azimuths_rad.astype(np.float64))
        self.assertTrue(np.all(diffs < 0))

        # First and last azimuths close to original
        orig_az = self.model.column_azimuths_rad.astype(np.float64)
        up_az = model_2x.column_azimuths_rad.astype(np.float64)
        self.assertAlmostEqual(up_az[0], orig_az[0], places=4)
        self.assertAlmostEqual(up_az[-1], orig_az[-1], places=2)

    def test_upsample_model_identity(self) -> None:
        """Upsampling by 1 returns unchanged model."""
        result = upsample_model(self.model, 1)
        self.assertIs(result, self.model)

    def test_upsample_model_preserves_monotonicity(self) -> None:
        """Upsampling by 4 maintains strictly decreasing azimuths (CW)."""
        model_4x = upsample_model(self.model, 4)

        diffs = np.diff(model_4x.column_azimuths_rad.astype(np.float64))
        self.assertTrue(np.all(diffs < 0))

    # --- optimize_model tests --------------------------------------------------

    def test_optimize_model_reduces_residual(self) -> None:
        """Optimization reduces residual when given a systematic offset."""
        model = self.model
        n_points = model.n_columns * model.n_rows

        # Create synthetic observations: model directions + systematic per-column offset
        model_cols = np.repeat(np.arange(model.n_columns, dtype=np.int64), model.n_rows)
        model_rows = np.tile(np.arange(model.n_rows, dtype=np.int64), model.n_columns)

        # "True" azimuths = model azimuths + 0.001 rad systematic offset
        offset = 0.001
        true_azimuths = (
            model.column_azimuths_rad[model_cols].astype(np.float64)
            + model.row_azimuth_offsets_rad[model_rows].astype(np.float64)
            + offset
        )

        distances = np.full(n_points, 30.0, dtype=np.float64)

        # Initial residual
        initial_predicted = model.column_azimuths_rad[model_cols].astype(np.float64) + model.row_azimuth_offsets_rad[
            model_rows
        ].astype(np.float64)
        initial_residual = np.abs(
            np.arctan2(
                np.sin(true_azimuths - initial_predicted),
                np.cos(true_azimuths - initial_predicted),
            )
        ).mean()

        # Optimize
        optimized = optimize_model(
            model,
            frame_azimuths=[true_azimuths],
            frame_model_cols=[model_cols],
            frame_model_rows=[model_rows],
            frame_distances=[distances],
            min_range_m=10.0,
            n_iterations=1,
        )

        # Compute residual after optimization
        opt_predicted = optimized.column_azimuths_rad[model_cols].astype(
            np.float64
        ) + optimized.row_azimuth_offsets_rad[model_rows].astype(np.float64)
        opt_residual = np.abs(
            np.arctan2(
                np.sin(true_azimuths - opt_predicted),
                np.cos(true_azimuths - opt_predicted),
            )
        ).mean()

        self.assertLess(opt_residual, initial_residual)
        # Should be near zero after 1 iteration with clean data
        self.assertLess(opt_residual, 1e-5)

    def test_optimize_model_sparse_columns_global_offset(self) -> None:
        """Sparse observations + a global phase offset must not tear the ramp.

        Regression for the scene-0103/0007 failure: a high-resolution model has
        far more columns than observed per frame, and the data carries a roughly
        constant phase offset (~pi for some scenes) relative to the model. The
        naive per-column update only shifted observed columns, leaving the
        unobserved majority behind -- exploding the azimuth span past 2*pi and
        producing a model that no longer reconstructs the point cloud. The
        global/local split must keep the ramp monotonic and consistent.
        """
        model = self.model
        n_obs_cols = model.n_columns // 4  # only a quarter of columns observed

        # Observe a sparse, evenly-spaced subset of columns (>=3 points each),
        # using only the reference row to keep the example small.
        observed = np.arange(0, model.n_columns, 4, dtype=np.int64)[:n_obs_cols]
        ref_row = model.n_rows // 2
        reps = 3
        model_cols = np.repeat(observed, reps)
        model_rows = np.full(model_cols.shape, ref_row, dtype=np.int64)

        # True azimuths = model azimuths + a large global phase offset.
        global_offset = 3.0  # ~pi, the scene-0103 case
        true_azimuths = np.arctan2(
            np.sin(model.column_azimuths_rad[model_cols].astype(np.float64) + global_offset),
            np.cos(model.column_azimuths_rad[model_cols].astype(np.float64) + global_offset),
        )
        distances = np.full(model_cols.shape, 30.0, dtype=np.float64)

        optimized = optimize_model(
            model,
            frame_azimuths=[true_azimuths],
            frame_model_cols=[model_cols],
            frame_model_rows=[model_rows],
            frame_distances=[distances],
            min_range_m=10.0,
            n_iterations=1,
        )

        # The result must still be a single, strictly-monotonic revolution.
        col_az = optimized.column_azimuths_rad
        rel = data_util.relative_angle(col_az[0], col_az, "cw")
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))
        span = float(col_az.astype(np.float64).max() - col_az.astype(np.float64).min())
        self.assertLess(span, 2 * np.pi)

        # The global offset must be absorbed: residual on observed columns small.
        pred = optimized.column_azimuths_rad[model_cols].astype(np.float64)
        resid = np.abs(np.arctan2(np.sin(true_azimuths - pred), np.cos(true_azimuths - pred)))
        self.assertLess(resid.mean(), 0.05)

    def test_optimize_model_noisy_correction_stays_monotonic(self) -> None:
        """Per-column correction noise must not reorder columns.

        Regression for the upsampled-model inversions: the per-column median
        residual carries high-frequency noise from uneven point-to-column
        assignment. Without smoothing, that noise exceeds the column spacing for
        a fraction of columns and swaps them, so the azimuth ramp is no longer
        monotonic and the model constructor rejects it. Smoothing the correction
        must keep the output strictly monotonic.
        """
        # Upsample so the column spacing is small relative to the per-column
        # noise (this is where the problem appears in real data).
        model = upsample_model(self.model, 4)
        n_cols = model.n_columns
        ref_row = model.n_rows // 2

        rng = np.random.default_rng(0)
        # Observe most columns, a few times each, with per-observation azimuth
        # noise an order of magnitude larger than the column spacing.
        observed = np.arange(0, n_cols, dtype=np.int64)
        reps = 3
        model_cols = np.repeat(observed, reps)
        model_rows = np.full(model_cols.shape, ref_row, dtype=np.int64)
        column_spacing = 2 * np.pi / n_cols
        noise = rng.normal(0.0, 5.0 * column_spacing, size=model_cols.shape)
        true_azimuths = model.column_azimuths_rad[model_cols].astype(np.float64) + noise
        distances = np.full(model_cols.shape, 30.0, dtype=np.float64)

        optimized = optimize_model(
            model,
            frame_azimuths=[true_azimuths],
            frame_model_cols=[model_cols],
            frame_model_rows=[model_rows],
            frame_distances=[distances],
            min_range_m=10.0,
            n_iterations=1,
        )

        col_az = optimized.column_azimuths_rad
        rel = data_util.relative_angle(col_az[0], col_az, "cw")
        self.assertTrue(
            np.all(np.diff(rel.relative_angle_rad) > 0),
            "optimized azimuths must be strictly monotonic despite correction noise",
        )

    # --- _smooth_sparse_correction tests ---------------------------------------

    def test_smooth_sparse_correction_interpolates_and_denoises(self) -> None:
        """Sparse, spiky correction becomes a smooth dense field."""
        n = 200
        cols = np.arange(n)
        # Smooth underlying trend.
        trend = 0.01 * np.sin(2 * np.pi * cols / n)
        # Only every 3rd column observed; observed values carry spiky noise.
        valid = (cols % 3) == 0
        rng = np.random.default_rng(1)
        values = np.zeros(n, dtype=np.float64)
        values[valid] = trend[valid] + rng.normal(0.0, 0.02, size=int(valid.sum()))

        smoothed = _smooth_sparse_correction(values, valid)

        # Dense (defined everywhere) and adjacent variation strongly reduced.
        self.assertEqual(smoothed.shape, (n,))
        raw_adj = np.abs(np.diff(values[valid]))
        smooth_adj = np.abs(np.diff(smoothed))
        self.assertLess(smooth_adj.max(), raw_adj.max())
        # Tracks the underlying trend within the noise level.
        self.assertLess(float(np.abs(smoothed - trend).mean()), 0.02)

    def test_smooth_sparse_correction_empty_valid_is_identity(self) -> None:
        """With no observed columns the correction is returned unchanged."""
        values = np.zeros(16, dtype=np.float64)
        valid = np.zeros(16, dtype=bool)
        out = _smooth_sparse_correction(values, valid)
        np.testing.assert_array_equal(out, values)

    # --- enforce_spinning_monotonic tests --------------------------------------

    def test_enforce_spinning_monotonic_repairs_equal_pair(self) -> None:
        """Adjacent equal azimuths are nudged apart to strictly decreasing (cw)."""
        n = 1085
        az = -np.arange(n, dtype=np.float64) * (2 * np.pi / n)
        # Force an exactly-degenerate adjacent pair.
        az[500] = az[499]

        repaired = enforce_spinning_monotonic(az, n, "cw")

        # The helper returns float32 (the dtype the model stores).
        self.assertEqual(repaired.dtype, np.float32)
        diffs = np.diff(repaired)
        self.assertTrue(np.all(diffs < 0), "must be strictly decreasing")

    def test_enforce_spinning_monotonic_repairs_local_inversion(self) -> None:
        """A small local inversion is repaired without flipping global order."""
        n = 1085
        az = -np.arange(n, dtype=np.float64) * (2 * np.pi / n)
        # Swap two neighbours to create a tiny inversion.
        az[300], az[301] = az[301], az[300]

        repaired = enforce_spinning_monotonic(az, n, "cw")

        diffs = np.diff(repaired)
        self.assertTrue(np.all(diffs < 0))

    def test_enforce_spinning_monotonic_survives_float32_cast(self) -> None:
        """The float32 result passes the ncore strict-monotonicity check (cw)."""
        n = 1085
        az = -np.arange(n, dtype=np.float64) * (2 * np.pi / n)
        az[500] = az[499]

        repaired32 = enforce_spinning_monotonic(az, n, "cw")
        self.assertEqual(repaired32.dtype, np.float32)

        rel = data_util.relative_angle(repaired32[0], repaired32, "cw")
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))

    def test_enforce_spinning_monotonic_reference_near_pi_boundary(self) -> None:
        """A spin whose reference column sits near -pi still passes the check.

        Regression for the empirical scene-1077 crash. The underlying defect was
        in util.relative_angle (mixed float32/float64 `% 2pi` reduction); this
        test guards the end-to-end behaviour: a strictly-decreasing CW sweep
        starting just above -pi must yield strictly-increasing relative angles
        and not wrap.
        """
        n = 4340
        # Decreasing sweep whose first element is just above -pi.
        az = -np.pi + 1e-3 - np.arange(n, dtype=np.float64) * ((2 * np.pi - 2e-3) / n)

        repaired = enforce_spinning_monotonic(az, n, "cw")

        rel = data_util.relative_angle(repaired[0], repaired, "cw")
        self.assertEqual(float(rel.relative_angle_rad[0]), 0.0)
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))
        self.assertTrue(np.all(~rel.wrap_around_flag))

    def test_enforce_spinning_monotonic_keeps_span_below_2pi(self) -> None:
        """Repairing the real-world handful of inversions keeps span below 2*pi.

        A near-full-revolution sweep (as the optimizer produces for a 4x model)
        with a scattering of small adjacent inversions must still come out
        strictly monotonic and strictly under one revolution.
        """
        n = 4340
        # Uniform near-full-revolution sweep, then inject a handful of small
        # local inversions like the per-column median corrections produce.
        az = -np.arange(n, dtype=np.float64) * ((2 * np.pi - 1e-3) / n)
        for i in (37, 311, 1024, 2570, 3999):
            az[i], az[i + 1] = az[i + 1], az[i]

        repaired32 = enforce_spinning_monotonic(az, n, "cw")

        span = float(repaired32[0]) - float(repaired32[-1])
        self.assertLess(span, 2 * np.pi, "span must stay below one revolution")
        self.assertTrue(np.all(np.diff(repaired32) < 0), "strictly decreasing in float32")
        rel = data_util.relative_angle(repaired32[0], repaired32, "cw")
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))

    def test_enforce_spinning_monotonic_rejects_more_than_one_revolution(self) -> None:
        """Input covering more than a full revolution is rejected, not reshaped.

        A genuine spinning sweep covers slightly less than 2*pi. An estimate that
        sweeps past a full revolution indicates an upstream estimation bug, so the
        helper raises rather than silently compressing it into a plausible-looking
        but meaningless model.
        """
        n = 1085
        # A monotonic sweep decreasing by 2.5*pi total -- more than one revolution.
        az = -np.linspace(0.0, 2.5 * np.pi, n)

        with self.assertRaises(ValueError):
            enforce_spinning_monotonic(az, n, "cw")

    def test_enforce_spinning_monotonic_ccw_repairs_to_increasing(self) -> None:
        """For CCW rotation the repaired azimuths are strictly increasing."""
        n = 1085
        # Increasing CCW sweep with a degenerate pair and a local inversion.
        az = np.arange(n, dtype=np.float64) * (2 * np.pi / n)
        az[500] = az[499]
        az[300], az[301] = az[301], az[300]

        repaired = enforce_spinning_monotonic(az, n, "ccw")

        self.assertEqual(repaired.dtype, np.float32)
        self.assertTrue(np.all(np.diff(repaired) > 0), "must be strictly increasing")
        # CCW relative angles must be strictly increasing per the model contract.
        rel = data_util.relative_angle(repaired[0], repaired, "ccw")
        self.assertEqual(float(rel.relative_angle_rad[0]), 0.0)
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))
        self.assertTrue(np.all(~rel.wrap_around_flag))

    def test_enforce_spinning_monotonic_rejects_invalid_direction(self) -> None:
        """An unknown spinning direction raises ValueError."""
        az = -np.arange(8, dtype=np.float64) * (2 * np.pi / 8)
        with self.assertRaises(ValueError):
            enforce_spinning_monotonic(az, 8, cast(Literal["cw", "ccw"], "sideways"))

    # --- derive_model_from_decompensated tests ---------------------------------

    def _make_decompensated_grid(self, column_azimuths: np.ndarray, n_beams: int) -> np.ndarray:
        """Build a synthetic decompensated point cloud [n_cols*n_beams, 3].

        All beams in a column share the column azimuth; rows get distinct
        elevations. Reshaped as [n_cols, n_beams, 3] by the estimator.
        """
        n_cols = len(column_azimuths)
        elevations = np.linspace(np.radians(-30.0), np.radians(10.0), n_beams)
        distance = 30.0  # far-range valid returns
        xyz = np.zeros((n_cols, n_beams, 3), dtype=np.float64)
        for c in range(n_cols):
            az = column_azimuths[c]
            for r in range(n_beams):
                el = elevations[r]
                cos_el = np.cos(el)
                xyz[c, r, 0] = distance * cos_el * np.cos(az)
                xyz[c, r, 1] = distance * cos_el * np.sin(az)
                xyz[c, r, 2] = distance * np.sin(el)
        return xyz.reshape(n_cols * n_beams, 3)

    def test_derive_model_from_decompensated_near_degenerate(self) -> None:
        """Near-degenerate adjacent azimuths must not break model construction.

        Reproduces the scene-0007 failure: real per-column azimuths are not
        perfectly uniform, so adjacent columns can become equal (or invert)
        after the float32 cast, tripping the ncore strict-monotonicity assert.
        The fix repairs these so the model constructs successfully.
        """
        n_cols = HDL32E_N_COLUMNS
        n_beams = HDL32E_N_BEAMS

        # Uniform CW decreasing azimuths with a couple of near-degenerate pairs:
        # one sub-float32-eps step and one tiny local inversion.
        az = -np.arange(n_cols, dtype=np.float64) * (2 * np.pi / n_cols)
        az[400] = az[399] - 1e-9  # collapses to equal after float32 cast
        az[800], az[801] = az[801], az[800]  # local inversion

        xyz = self._make_decompensated_grid(az, n_beams)

        model = derive_model_from_decompensated(
            xyz_decompensated=xyz,
            n_beams_per_column=n_beams,
            n_target_cols=n_cols,
            spinning_direction="cw",
            spinning_frequency_hz=20.0,
        )

        assert model is not None
        # The model constructor already enforces strict monotonicity; verify it
        # explicitly so this test documents the invariant.
        col_az = model.column_azimuths_rad
        rel = data_util.relative_angle(col_az[0], col_az, "cw")
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))

    def test_derive_model_from_decompensated_rejects_per_beam_outliers(self) -> None:
        """Per-beam azimuth outliers must not corrupt the column azimuths.

        The per-column azimuth is the circular median across all valid rows, so a
        few rows carrying spurious returns (wrong object, multipath) in each
        column are rejected. A single reference row would instead let those
        outliers reorder columns. The recovered azimuths must match the true
        per-column sweep and be strictly monotonic.
        """
        n_cols = HDL32E_N_COLUMNS
        n_beams = HDL32E_N_BEAMS

        true_az = -np.arange(n_cols, dtype=np.float64) * (2 * np.pi / n_cols)
        xyz = self._make_decompensated_grid(true_az, n_beams).reshape(n_cols, n_beams, 3)

        # Corrupt a few beams per column with a large azimuth offset (gross
        # outliers), but keep the majority of rows clean so the median survives.
        rng = np.random.default_rng(0)
        distance = 30.0
        for c in range(n_cols):
            bad_rows = rng.choice(n_beams, size=3, replace=False)
            for r in bad_rows:
                bad_az = true_az[c] + rng.uniform(1.0, 3.0)  # ~60-170 deg off
                xyz[c, r, 0] = distance * np.cos(bad_az)
                xyz[c, r, 1] = distance * np.sin(bad_az)
                xyz[c, r, 2] = 0.0
        xyz = xyz.reshape(n_cols * n_beams, 3)

        model = derive_model_from_decompensated(
            xyz_decompensated=xyz,
            n_beams_per_column=n_beams,
            n_target_cols=n_cols,
            spinning_direction="cw",
            spinning_frequency_hz=20.0,
        )

        assert model is not None
        col_az = model.column_azimuths_rad
        # Strictly monotonic.
        rel = data_util.relative_angle(col_az[0], col_az, "cw")
        self.assertTrue(np.all(np.diff(rel.relative_angle_rad) > 0))
        # And close to the true sweep (outliers rejected), allowing a constant
        # global offset (the model frame is defined up to a rotation).
        recovered = np.unwrap(col_az.astype(np.float64))
        truth = np.unwrap(true_az)
        residual = (recovered - truth) - np.median(recovered - truth)
        self.assertLess(float(np.abs(residual).max()), np.radians(1.0))

    # --- derive_nominal_hdl32e tests -------------------------------------------

    def test_derive_nominal_hdl32e_dimensions(self) -> None:
        """Nominal HDL-32E model has correct dimensions and direction."""
        model = self.model

        self.assertEqual(model.n_rows, 32)
        self.assertEqual(model.n_columns, 1085)
        self.assertEqual(model.spinning_direction, "cw")
        np.testing.assert_array_equal(model.row_elevations_rad, HDL32E_ELEVATIONS_RAD)
        self.assertEqual(len(model.column_azimuths_rad), 1085)
        self.assertEqual(len(model.row_azimuth_offsets_rad), 32)

    def test_derive_nominal_hdl32e_uniform_azimuths(self) -> None:
        """Column azimuths are uniformly spaced (after unwrap)."""
        model = self.model
        az_unwrapped = np.unwrap(model.column_azimuths_rad.astype(np.float64))
        diffs = np.diff(az_unwrapped)

        expected_step = -2 * np.pi / HDL32E_N_COLUMNS
        np.testing.assert_allclose(diffs, expected_step, atol=1e-5)

    # --- compute_intra_column_firing_offsets tests -----------------------------

    def test_compute_intra_column_firing_offsets_range(self) -> None:
        """Offsets have expected total angular range and correct dtype/shape."""
        offsets = compute_intra_column_firing_offsets(
            n_beams=32,
            beam_pair_interval_us=1.152,
            scan_duration_us=50000,
            spinning_direction="cw",
        )

        self.assertEqual(offsets.dtype, np.float32)
        self.assertEqual(len(offsets), 32)

        # Total range: 2 banks of 16, max beam_in_bank=15
        # time span per bank = 15 * 1.152 * 2 = 34.56 us
        # angular range = time_span * (2*pi / 50000)
        angular_rate = 2.0 * np.pi / 50000
        max_time_us = 15 * 1.152 * 2
        expected_range_rad = max_time_us * angular_rate
        actual_range = float(offsets.max() - offsets.min())

        # The range should be close to expected (mean-subtraction doesn't change range)
        self.assertAlmostEqual(actual_range, expected_range_rad, places=5)

    def test_compute_intra_column_firing_offsets_symmetry(self) -> None:
        """Offsets are mean-subtracted (sum ~0). Two banks have similar patterns."""
        offsets = compute_intra_column_firing_offsets(
            n_beams=32,
            beam_pair_interval_us=1.152,
            scan_duration_us=50000,
            spinning_direction="cw",
        )

        # Mean-subtracted: mean should be ~0
        self.assertAlmostEqual(float(offsets.mean()), 0.0, places=6)

        # Two banks (in ring order before reversal): even rings and odd rings
        # In model order (reversed ring), first 16 and last 16
        bank1 = offsets[:16].astype(np.float64)
        bank2 = offsets[16:].astype(np.float64)

        # Both banks should span similar ranges (same timing pattern)
        range1 = bank1.max() - bank1.min()
        range2 = bank2.max() - bank2.min()
        self.assertAlmostEqual(range1, range2, places=5)

    # --- compute_model_consistency tests ---------------------------------------

    def test_compute_model_consistency_perfect(self) -> None:
        """Perfect consistency: stored directions match model predictions exactly."""
        model = self.model
        n_points = 1000

        # Create random valid model element indices
        rng = np.random.default_rng(42)
        model_rows = rng.integers(0, model.n_rows, size=n_points).astype(np.uint16)
        model_cols = rng.integers(0, model.n_columns, size=n_points).astype(np.uint16)
        model_element = np.stack([model_rows, model_cols], axis=1)

        # Compute model-predicted directions
        model_az = model.column_azimuths_rad[model_cols].astype(np.float64) + model.row_azimuth_offsets_rad[
            model_rows
        ].astype(np.float64)
        model_el = model.row_elevations_rad[model_rows].astype(np.float64)
        cos_el = np.cos(model_el)
        directions = np.stack(
            [cos_el * np.cos(model_az), cos_el * np.sin(model_az), np.sin(model_el)],
            axis=1,
        ).astype(np.float32)

        distances = np.full(n_points, 30.0, dtype=np.float32)

        mean_err_all, mean_err_far, mean_az_shift = compute_model_consistency(
            directions, model_element, distances, model
        )

        self.assertAlmostEqual(mean_err_all, 0.0, places=1)  # float32 precision ~0.005 deg
        self.assertAlmostEqual(mean_err_far, 0.0, places=1)
        self.assertAlmostEqual(mean_az_shift, 0.0, places=2)

    def test_compute_model_consistency_with_offset(self) -> None:
        """Systematic azimuth offset produces expected error magnitude."""
        model = self.model
        n_points = 2000
        az_offset_rad = 0.01

        rng = np.random.default_rng(123)
        model_rows = rng.integers(0, model.n_rows, size=n_points).astype(np.uint16)
        model_cols = rng.integers(0, model.n_columns, size=n_points).astype(np.uint16)
        model_element = np.stack([model_rows, model_cols], axis=1)

        # Compute directions WITH systematic azimuth offset
        model_az = (
            model.column_azimuths_rad[model_cols].astype(np.float64)
            + model.row_azimuth_offsets_rad[model_rows].astype(np.float64)
            + az_offset_rad
        )
        model_el = model.row_elevations_rad[model_rows].astype(np.float64)
        cos_el = np.cos(model_el)
        directions = np.stack(
            [cos_el * np.cos(model_az), cos_el * np.sin(model_az), np.sin(model_el)],
            axis=1,
        ).astype(np.float32)

        distances = np.full(n_points, 30.0, dtype=np.float32)

        mean_err_all, mean_err_far, mean_az_shift = compute_model_consistency(
            directions, model_element, distances, model
        )

        expected_deg = np.degrees(az_offset_rad)  # ~0.573 deg
        # The angular error won't be exactly equal to the az offset due to elevation,
        # but the azimuth shift metric should be close
        self.assertAlmostEqual(mean_az_shift, expected_deg, places=1)
        # Total angular error should also be in the right ballpark
        self.assertAlmostEqual(mean_err_all, expected_deg, delta=0.1)


if __name__ == "__main__":
    unittest.main()
