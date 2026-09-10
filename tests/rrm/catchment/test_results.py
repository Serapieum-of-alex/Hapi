"""Tests for `SimulationResults`' own surface: what it refuses, and what it writes.

The presentation methods moved here off `Catchment`, so this is where the arrays are read,
dated and written. The happy paths live beside the fixtures that produce them --
`test_save_results_distributed.py` for the rasters, `test_plot_animation.py` for the
animation, `test_rrm_catchment.py` for the lumped CSV. What is left, and what this file
covers, is the behaviour at the edges of those methods: the run they need and might not have,
the fields a routing step has not filled yet, the dates that are not steps of the run, and the
CSV options nothing else exercises.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from hapi.catchment import Catchment
from hapi.inputs import FlowNetwork, MeteoInputs
from hapi.results import STATE_VARIABLES, RoutingKind, SimulationResults
from hapi.routing import Routing
from hapi.rrm.distrrm import DistributedRRM
from hapi.rrm.hbv_bergestrom92 import HBVBergestrom92 as HBVLumped
from hapi.run import Run
from hapi.runs import DistributedRun

DATE_REGEX = r"\d{4}.\d{2}.\d{2}"


@pytest.fixture(scope="module")
def distributed_run(
    coello_start_date: str,
    coello_end_date: str,
    coello_prec_path: str,
    coello_temp_path: str,
    coello_evap_path: str,
    coello_acc_path: str,
    coello_fd_path: str,
    coello_dist_parameters_muskingum: str,
    coello_cat_area: int,
    coello_initial_cond: list,
) -> DistributedRun:
    """A validated distributed Coello run, with no engine having touched it yet.

    Returns:
        DistributedRun: The narrowed run.
    """
    model = Catchment(
        "coello",
        coello_start_date,
        coello_end_date,
        spatial_resolution="Distributed",
        temporal_resolution="Daily",
    )
    model.meteo = MeteoInputs.from_rasters(
        coello_prec_path,
        coello_temp_path,
        coello_evap_path,
        start=coello_start_date,
        end=coello_end_date,
        regex_string=DATE_REGEX,
        date=True,
        file_name_data_fmt="%Y.%m.%d",
    )
    model.flow_network = FlowNetwork.from_rasters(coello_acc_path, coello_fd_path)
    model.read_parameters(coello_dist_parameters_muskingum, False)
    model.read_lumped_model(HBVLumped, coello_cat_area, coello_initial_cond)
    return DistributedRun.from_model(model)


@pytest.fixture(scope="module")
def unrouted(distributed_run: DistributedRun) -> SimulationResults:
    """Results straight out of the per-cell model, before any routing step.

    Args:
        distributed_run: The validated run.

    Returns:
        SimulationResults: `quz`, `qlz` and the states filled; every routed field `None`.
    """
    return DistributedRRM.run_lumped_model(distributed_run)


@pytest.fixture(scope="module")
def lumped_results(
    coello_rrm_date: list,
    lumped_meteo_data_path: str,
    coello_AreaCoeff: float,
    coello_InitialCond: list,
    lumped_parameters_path: str,
    coello_Snow: int,
) -> SimulationResults:
    """A finished lumped run's results.

    Returns:
        SimulationResults: Routed `RoutingKind.LUMPED`, carrying its `LumpedRun`.
    """
    model = Catchment("rrm", coello_rrm_date[0], coello_rrm_date[1])
    model.read_lumped_inputs(lumped_meteo_data_path)
    model.read_lumped_model(HBVLumped, coello_AreaCoeff, coello_InitialCond)
    model.read_parameters(lumped_parameters_path, coello_Snow)
    return Run.run_lumped(model, 1, Routing.muskingum_v)


class TestTheRunTravelsWithTheArrays:
    """`run` is what makes the arrays interpretable, so its absence is reported by name."""

    def test_a_finished_run_is_carried_on_its_results(
        self, distributed_run: DistributedRun, unrouted: SimulationResults
    ):
        """Test that the engine attaches the run it was given to what it returns.

        Args:
            distributed_run: The validated run.
            unrouted: What the per-cell model produced from it.

        Test scenario:
            Without this the arrays have no calendar to be indexed by and no grid to be
            masked with, and every presentation method would need them passed back in.
        """
        assert unrouted.run is distributed_run, (
            "the results must carry the run that produced them, not a copy"
        )

    def test_a_lumped_run_is_carried_too(self, lumped_results: SimulationResults):
        """Test that the lumped engine attaches its run as well.

        Args:
            lumped_results: A finished lumped run's results.

        Test scenario:
            `save` dates the CSV from `run.period`, so the lumped path needs the run just as
            much as the distributed one -- it only needs less of it.
        """
        assert lumped_results.run is not None, "a lumped run is provenance too"
        assert lumped_results.routing is RoutingKind.LUMPED, (
            f"expected LUMPED, got {lumped_results.routing}"
        )

    def test_a_lumped_run_cannot_be_animated(self, lumped_results: SimulationResults):
        """Test that asking a lumped run for an animation says why it cannot.

        Args:
            lumped_results: A finished lumped run's results.

        Test scenario:
            A lumped run has no grid and no spatial drivers, so there is nothing to animate.
            `LumpedRun` carries neither `flow_network` nor `meteo`, so without this guard the
            failure would be an `AttributeError` naming a field of the run rather than the
            reason.
        """
        with pytest.raises(ValueError, match="lumped run"):
            lumped_results.animate("2012-06-14", "2012-06-20", option=1)


class TestUnfilledFieldsAreNamed:
    """A routed field is `None` until its routing step runs, and that is a real state."""

    @pytest.mark.parametrize("option, field", [(1, "q_total"), (2, "quz_routed")])
    def test_an_unrouted_field_names_itself_and_the_routing(
        self, unrouted: SimulationResults, option: int, field: str
    ):
        """Test that reading a routed field before routing names the field and the state.

        Args:
            unrouted: Results from the per-cell model, with no routing applied.
            option: The animation option that reads the field.
            field: The field it reads.

        Test scenario:
            `run_lumped_model` leaves every routed field `None` and the routing `UNROUTED`.
            Slicing one used to raise `TypeError: 'NoneType' object is not subscriptable`
            from inside the option dispatch, naming neither the field nor the missing step.
        """
        with pytest.raises(ValueError, match=field) as exc:
            unrouted.animate("2009-01-01", "2009-01-05", option=option)

        assert "unrouted" in str(exc.value), (
            f"the error should say which state the results are in: {exc.value}"
        )

    def test_an_unread_state_option_names_the_switch(
        self, distributed_run: DistributedRun
    ):
        """Test that a state option on a run that dropped the states names the switch.

        Args:
            distributed_run: The validated run, rebuilt here without the states.

        Test scenario:
            `keep_state_variables=False` is the memory opt-out; the guard has to name it
            rather than fail on `None` inside a slice several frames away.
        """
        model = distributed_run
        dropped = DistributedRun(
            period=model.period,
            meteo=model.meteo,
            flow_network=model.flow_network,
            parameters=model.parameters,
            model_setup=model.model_setup,
            keep_state_variables=False,
        )
        results = DistributedRRM.run_lumped_model(dropped)

        with pytest.raises(ValueError, match="keep_state_variables"):
            results.animate("2009-01-01", "2009-01-05", option=4)


class TestDatesAreResolvedAgainstTheRun:
    """The arrays are positional; the run's calendar is what turns a date into a step."""

    def test_datetime_arguments_are_accepted_as_they_are(
        self, unrouted: SimulationResults, tmp_path
    ):
        """Test that `datetime` bounds are used directly rather than re-parsed.

        Args:
            unrouted: Results carrying the run whose calendar dates them.
            tmp_path: Unused destination; the call is expected to fail before writing.

        Test scenario:
            `start` and `end` are documented as `str | datetime`. A string is read with
            `fmt`; a datetime must skip that, since `strptime` on one raises `TypeError`.
            Reaching the *result*-option error proves both bounds resolved.
        """
        with pytest.raises(ValueError, match="between 1 and 8"):
            unrouted.save(
                path=str(tmp_path),
                result=99,
                start=dt.datetime(2009, 1, 1),
                end=dt.datetime(2009, 1, 5),
                flow_acc_path="unused",
            )

    @pytest.mark.parametrize(
        "start, end, offender",
        [
            ("2008-12-31", "2009-01-05", "start"),
            ("2009-01-01", "2030-01-01", "end"),
        ],
    )
    def test_a_date_outside_the_run_is_refused(
        self, unrouted: SimulationResults, start: str, end: str, offender: str
    ):
        """Test that a date the run never covered is named rather than silently missed.

        Args:
            unrouted: Results carrying the run whose calendar dates them.
            start: Start date under test.
            end: End date under test.
            offender: Which of the two is out of range.

        Test scenario:
            The lookup is `np.nonzero(index == value)[0][0]`, which raises
            `IndexError: index 0 is out of bounds` on a miss -- naming neither the date nor
            the span the run actually covers.
        """
        with pytest.raises(ValueError, match=f"{offender} date") as exc:
            unrouted.animate(start, end, option=1)

        assert "which covers" in str(exc.value), (
            f"the error should show the span the run does cover: {exc.value}"
        )

    def test_empty_bounds_mean_the_whole_run(
        self, lumped_results: SimulationResults, tmp_path
    ):
        """Test that omitting both dates writes the run's whole span.

        Args:
            lumped_results: A finished lumped run's results.
            tmp_path: Destination directory.

        Test scenario:
            `""` is the documented default for both bounds and means "the first step" and
            "the last step". The row count is what proves it, since a wrong default would
            still write a valid file.
        """
        out = tmp_path / "whole.csv"

        lumped_results.save(path=str(out), result=1)

        written = pd.read_csv(out)
        assert len(written) == len(lumped_results.run.period), (
            f"expected one row per step ({len(lumped_results.run.period)}), "
            f"got {len(written)}"
        )


class TestTheCsvBranch:
    """A lumped run has no grid, so `save` writes a CSV -- read off `routing`, not passed."""

    @pytest.mark.parametrize(
        "result, columns",
        [
            (1, ["Qsim"]),
            (2, ["Quz"]),
            (3, ["Qlz"]),
            (4, STATE_VARIABLES),
            (5, ["Qsim", "Quz", "Qlz", *STATE_VARIABLES]),
        ],
    )
    def test_each_option_writes_its_own_columns(
        self, lumped_results: SimulationResults, tmp_path, result: int, columns: list
    ):
        """Test that every lumped option writes exactly the series it names.

        Args:
            lumped_results: A finished lumped run's results.
            tmp_path: Destination directory.
            result: The option under test.
            columns: The columns it must produce, beside `date`.

        Test scenario:
            The five options were a chain of `elif` branches each re-writing the file; they
            are now additive, so option 5 is the union of the others rather than a separate
            copy of them. Only 1 and 5 were exercised anywhere before this.
        """
        out = tmp_path / f"result-{result}.csv"

        lumped_results.save(path=str(out), result=result)

        written = pd.read_csv(out)
        assert written.columns.to_list() == ["date", *columns], (
            f"option {result} should write {['date', *columns]}, "
            f"got {written.columns.to_list()}"
        )
        assert len(written) == len(lumped_results.run.period), (
            f"expected one row per step, got {len(written)}"
        )

    def test_the_index_follows_the_run_not_a_daily_default(
        self, lumped_results: SimulationResults, tmp_path
    ):
        """Test that the written dates come from the run's own calendar.

        Args:
            lumped_results: A finished lumped run's results.
            tmp_path: Destination directory.

        Test scenario:
            This branch used to build its index with a hard-coded `freq="D"`, so an hourly
            lumped run wrote a daily index against hourly values -- the dates and the numbers
            described different steps. Comparing against `period.date_index` is what pins
            that the calendar is the run's.
        """
        out = tmp_path / "dates.csv"

        lumped_results.save(path=str(out), result=1)

        written = pd.read_csv(out)
        expected = [
            f"'{str(step)[:10]}'" for step in lumped_results.run.period.date_index
        ]
        assert written["date"].to_list() == expected, (
            "the dates must be the run's own steps, not a fresh daily range"
        )

    @pytest.mark.parametrize("result", [0, 6])
    def test_an_option_outside_the_range_is_refused(
        self, lumped_results: SimulationResults, tmp_path, result: int
    ):
        """Test that a lumped option outside 1-5 raises rather than writing a bare file.

        Args:
            lumped_results: A finished lumped run's results.
            tmp_path: Destination directory.
            result: An out-of-range option.

        Test scenario:
            Without the guard the frame would be built, no series added, and a file with
            only a `date` column written -- a silent wrong answer rather than an error.
        """
        out = tmp_path / "never.csv"

        with pytest.raises(ValueError, match="between 1 and 5"):
            lumped_results.save(path=str(out), result=result)

        assert not out.exists(), "nothing should be written when the option is refused"


class TestResultsBuiltByHand:
    """Not every `SimulationResults` came from a run, and the difference has to be visible."""

    @pytest.fixture
    def orphan(self) -> SimulationResults:
        """Result arrays with no run behind them.

        Returns:
            SimulationResults: Routed-looking arrays, built directly.
        """
        cube = np.zeros((2, 3, 4), dtype="float32")
        return SimulationResults(RoutingKind.MUSKINGUM, cube, cube, None, q_total=cube)

    def test_saving_without_a_run_says_what_is_missing(self, orphan, tmp_path):
        """Test that `save` on hand-built results names the absent run.

        Args:
            orphan: Results built directly rather than by a run.
            tmp_path: Destination directory.

        Test scenario:
            There is no calendar to date the files by, so the failure has to name the run
            rather than surface as an `AttributeError` on `None.period`.
        """
        with pytest.raises(ValueError, match="carry no run"):
            orphan.save(path=str(tmp_path), flow_acc_path="unused")

    def test_a_non_string_path_is_refused_before_anything_else(self, orphan):
        """Test that `path` is type-checked ahead of the run lookup.

        Args:
            orphan: Results built directly rather than by a run.

        Test scenario:
            `outputs.results_dir` is optional in a run configuration, so a caller forwarding
            it can hold `None`. The check has to come first, or the message would name the
            missing run instead of the argument the caller actually got wrong.
        """
        with pytest.raises(TypeError, match="path must be a string") as exc:
            orphan.save(path=None)

        assert "NoneType" in str(exc.value), (
            f"the error should name what it got: {exc.value}"
        )

    def test_the_animation_state_is_not_a_constructor_argument(self, orphan):
        """Test that `anim` and the glyph start empty and are not settable at construction.

        Args:
            orphan: Results built directly rather than by a run.

        Test scenario:
            They are `field(init=False)` because they are set by `animate`, not by whoever
            builds the results. Passing one would otherwise silently create a results object
            claiming an animation it does not have.
        """
        assert orphan.anim is None, "no animation until `animate` runs"

        with pytest.raises(TypeError):
            SimulationResults(
                RoutingKind.MUSKINGUM,
                orphan.quz,
                orphan.qlz,
                None,
                anim="not a constructor argument",
            )
