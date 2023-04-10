import os
from typing import List
from matplotlib.figure import Figure
from pandas import DataFrame
from Hapi.hm.event import Event, Catalog


def test_create_event_instance(event_instance_attrs: List[str]):
    event_obg = Event(f"test")
    assert event_obg.depth_prefix == "DepthMax"
    assert isinstance(event_obg.reference_index, DataFrame)
    assert all([i in event_obg.__dict__.keys() for i in event_instance_attrs])


def test_create_from_overtopping(overtopping_file: str):
    event_obg = Event("test")
    event_obg.create_from_overtopping(overtopping_file)
    assert hasattr(event_obg, "event_index")
    assert isinstance(event_obg.event_index, DataFrame)


def test_calculateVolumeError(
    overtopping_file: str, volume_error_file: str, event_index_volume_attrs: List[str]
):
    event_obg = Event("test")
    event_obg.create_from_overtopping(overtopping_file)
    event_obg.calculate_volume_error(volume_error_file)
    assert all([i in event_obg.event_index.columns for i in event_index_volume_attrs])


def test_read_event_index(
    event_index_file: str, volume_error_file: str, event_index_volume_attrs2: List[str]
):
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    assert isinstance(event.event_index, DataFrame)
    assert all([i in event.event_index.columns for i in event_index_volume_attrs2])


def test_get_event_start(event_index_file: str, event_index_volume_attrs2: List[str]):
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    max_overtopping_ind = event.event_index["cells"].idxmax()
    start_ind, start_day = event.get_event_start(max_overtopping_ind)
    end_ind, end_day = event.get_event_end(max_overtopping_ind)
    assert start_day == 8195
    assert end_day == 8200


def test_get_event_by_index(
    event_index_file: str, volume_error_file: str, event_index_volume_attrs2: List[str]
):
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    event_details = event.get_event_by_index(1)
    assert event_details == {"start": 35, "end": 38, "cells": 1023}


def test_detailed_overtopping(
    event_index_file: str,
):
    event = Event.read_event_index("test", event_index_file, start="1955-01-01")
    event.two_d_result_path = "tests/hm/data"
    fig, ax, opts = event.histogram(8200, 0, 1)
    # {"n": n, "bins": bins, "patches": patches}
    assert isinstance(fig, Figure)
    assert all(elem in opts.keys() for elem in ["n", "bins", "patches"])


class TestCatalog:
    def test_read_file(self):
        path = "tests/hm/data/catalog.yaml"
        catalog = Catalog.read_file(path)
        assert isinstance(catalog.catalog, dict)
        assert catalog.events == [19, 50]

    def test_to_file(self):
        path = "tests/hm/data/catalog.yaml"
        save_to = "tests/hm/data/save_catalog.yaml"
        if os.path.exists(save_to):
            os.remove(save_to)

        catalog = Catalog.read_file(path)
        catalog.to_file(save_to)
        assert os.path.exists(path)
        os.remove(save_to)

    def test_dunder_methods(self):
        path = "tests/hm/data/catalog.yaml"
        catalog = Catalog.read_file(path)
        # test __len__
        assert len(catalog) == 2
        # test __iter__
        assert len(list(catalog)) == 2
        # test __getitem__
        assert catalog[19] == {
            "day": 19,
            "depth": {
                5: [0.0199, 0.0099, 0.14, 0.09, 0.0199, 0.0399],
                29: [0.4199, 0.31, 0.31, 0.3499, 0.34, 0.4099],
            },
            "reaches": [29.0, 5.0],
        }
        # test __setitem__
        # catalog[20] == {}
        # assert 20 in catalog.events
