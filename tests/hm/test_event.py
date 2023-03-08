from typing import List
from pandas import DataFrame
import Hapi.hm.event as E


def test_create_event_instance(event_instance_attrs: List[str]):
    Event = E.Event(f"test")
    assert Event.depth_prefix == "DepthMax"
    assert isinstance(Event.reference_index, DataFrame)
    assert all([i in Event.__dict__.keys() for i in event_instance_attrs])
