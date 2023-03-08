from pandas import DataFrame
import Hapi.hm.event as E

event_instance_attrs = [
    "left_overtopping_suffix",
    "right_overtopping_suffix",
    "duration_prefix",
    "return_period_prefix",
    "two_d_result_path",
    "compressed",
    "extracted_values",
    "event_index",
]


def test_create_event_instance():
    Event = E.Event(f"test")
    assert Event.depth_prefix == "DepthMax"
    assert isinstance(Event.reference_index, DataFrame)
    assert all([i in Event.__dict__.keys() for i in event_instance_attrs])
