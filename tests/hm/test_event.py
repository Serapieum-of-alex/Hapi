from typing import List
import pytest
from pandas import DataFrame
from Hapi.hm.event import Event


def test_create_event_instance(event_instance_attrs: List[str]) -> Event:
    event_obg = Event(f"test")
    assert event_obg.depth_prefix == "DepthMax"
    assert isinstance(event_obg.reference_index, DataFrame)
    assert all([i in event_obg.__dict__.keys() for i in event_instance_attrs])


def test_create_from_overtopping(overtopping_file: str) -> Event:
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
