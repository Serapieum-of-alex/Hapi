from typing import List
import pytest
from pandas import DataFrame
from Hapi.hm.event import Event


def test_create_event_instance(event_instance_attrs: List[str]) -> Event:
    event_obg = Event(f"test")
    assert event_obg.depth_prefix == "DepthMax"
    assert isinstance(event_obg.reference_index, DataFrame)
    assert all([i in event_obg.__dict__.keys() for i in event_instance_attrs])
    return event_obg


def test_readOvertopping(overtopping_file: str) -> Event:
    event_obg = Event("test")
    event_obg.readOvertopping(overtopping_file)
    assert hasattr(event_obg, "event_index")
    assert isinstance(event_obg.event_index, DataFrame)


def test_calculateVolumeError(
    overtopping_file: str, volume_error_file: str, event_index_volume_attrs: List[str]
):
    event_obg = Event("test")
    event_obg.readOvertopping(overtopping_file)
    event_obg.calculateVolumeError(volume_error_file)
    assert all([i in event_obg.event_index.columns for i in event_index_volume_attrs])
