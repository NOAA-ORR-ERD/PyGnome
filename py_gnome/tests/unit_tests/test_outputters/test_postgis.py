"""
Tests for the PostGIS outputter.

No real database is used; persist/row_factory are plain callables.
"""

import pytest
from datetime import datetime, timedelta

import numpy as np

from gnome.model import Model
from gnome.maps import MapFromBNA
from gnome.movers import SimpleMover
from gnome.spills.spill import point_line_spill
from gnome.spills.substance import NonWeatheringSubstance
from gnome.outputters.postgis import PostGISOutput

import os

_MAPFILE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "sample_data",
    "MapBounds_Island.bna",
)


@pytest.fixture
def simple_model():
    """
    A minimal model: SimpleMover, one spill, cache enabled, no uncertainty.
    Suitable for testing outputters without a real DB.
    """
    release_time = datetime(2012, 9, 15, 12, 0)
    map_ = MapFromBNA(_MAPFILE, refloat_halflife=6)
    model = Model(
        time_step=timedelta(minutes=15),
        start_time=release_time,
        duration=timedelta(hours=1),
        map=map_,
        uncertain=False,
    )
    model.movers += SimpleMover(velocity=(1.0, -1.0, 0.0))
    spill = point_line_spill(
        5,
        start_position=(-127.1, 47.93, 0.0),
        release_time=release_time,
        substance=NonWeatheringSubstance(),
        amount=100,
        units='kg',
    )
    model.spills += spill
    return model


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_valid(self):
        out = PostGISOutput(persist=lambda rows: None)
        assert callable(out._persist)
        assert callable(out._row_factory)
        assert out.run_id is None
        assert out.metadata == {}
        assert out.round_data is True
        assert out.round_to == 4

    def test_persist_not_callable_raises(self):
        with pytest.raises(TypeError, match="persist must be a callable"):
            PostGISOutput(persist="not_a_function")

    def test_row_factory_not_callable_raises(self):
        with pytest.raises(TypeError, match="row_factory must be a callable"):
            PostGISOutput(persist=lambda rows: None, row_factory="bad")

    def test_default_row_factory_is_identity(self):
        out = PostGISOutput(persist=lambda rows: None)
        sentinel = {"a": 1}
        assert out._row_factory(sentinel) is sentinel

    def test_run_id_stored(self):
        out = PostGISOutput(persist=lambda rows: None, run_id="abc-123")
        assert out.run_id == "abc-123"

    def test_metadata_stored(self):
        meta = {"scenario": "test"}
        out = PostGISOutput(persist=lambda rows: None, metadata=meta)
        assert out.metadata == meta

    def test_metadata_defaults_to_empty_dict(self):
        out = PostGISOutput(persist=lambda rows: None, metadata=None)
        assert out.metadata == {}


# ---------------------------------------------------------------------------
# Integration: model run
# ---------------------------------------------------------------------------

class TestModelRun:
    def test_persist_called_each_timestep(self, simple_model):
        calls = []
        out = PostGISOutput(persist=lambda rows: calls.append(rows))
        simple_model.outputters += out
        simple_model.full_run()
        assert len(calls) == simple_model.num_time_steps

    def test_rows_have_required_keys(self, simple_model):
        received = []
        out = PostGISOutput(persist=lambda rows: received.extend(rows))
        simple_model.outputters += out
        simple_model.full_run()

        required = {"run_id", "step", "element_index", "time", "lon", "lat",
                    "depth", "status_code", "mass"}
        for row in received:
            assert required.issubset(row.keys()), (
                f"Missing keys: {required - row.keys()}"
            )

    def test_run_id_present_in_every_row(self, simple_model):
        received = []
        out = PostGISOutput(
            persist=lambda rows: received.extend(rows),
            run_id="run-xyz",
        )
        simple_model.outputters += out
        simple_model.full_run()

        assert len(received) > 0
        assert all(r["run_id"] == "run-xyz" for r in received)

    def test_metadata_merged_into_rows(self, simple_model):
        received = []
        out = PostGISOutput(
            persist=lambda rows: received.extend(rows),
            metadata={"scenario": "unit-test", "version": 2},
        )
        simple_model.outputters += out
        simple_model.full_run()

        assert len(received) > 0
        for row in received:
            assert row["scenario"] == "unit-test"
            assert row["version"] == 2

    def test_custom_row_factory_applied(self, simple_model):
        received = []

        def factory(elem):
            return {"x": elem["lon"], "y": elem["lat"]}

        out = PostGISOutput(
            persist=lambda rows: received.extend(rows),
            row_factory=factory,
        )
        simple_model.outputters += out
        simple_model.full_run()

        assert len(received) > 0
        for row in received:
            assert set(row.keys()) == {"x", "y"}

    def test_step_index_increments(self, simple_model):
        calls = []
        out = PostGISOutput(persist=lambda rows: calls.append(rows))
        simple_model.outputters += out
        simple_model.full_run()

        steps_seen = [rows[0]["step"] for rows in calls if rows]
        assert steps_seen == sorted(steps_seen)


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    def test_row_factory_exception_does_not_crash(self, simple_model):
        """A failing row_factory should be logged but not raise."""
        persist_calls = []

        def bad_factory(elem):
            raise RuntimeError("factory boom")

        out = PostGISOutput(
            persist=lambda rows: persist_calls.append(rows),
            row_factory=bad_factory,
        )
        simple_model.outputters += out
        # Should not raise
        simple_model.full_run()
        # persist is called with empty lists (all rows dropped)
        assert all(len(r) == 0 for r in persist_calls)

    def test_persist_exception_does_not_crash(self, simple_model):
        """A failing persist should be logged but not raise."""
        def bad_persist(rows):
            raise RuntimeError("db boom")

        out = PostGISOutput(persist=bad_persist)
        simple_model.outputters += out
        # Should not raise
        simple_model.full_run()
