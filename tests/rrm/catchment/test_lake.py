import datetime as dt
from Hapi.catchment import Lake

def test_lake():
    start_date = "2012.06.14 19:00:00"
    end_date = "2012.6.15 14:00:00"
    fmt = "%Y.%m.%d %H:%M:%S"

    lake = Lake(
        start=start_date,
        end=end_date,
        fmt="%Y.%m.%d %H:%M:%S",
        temporal_resolution="Hourly",
        split=True,
    )


    assert len(lake.Index) == 20
    assert lake.start == dt.datetime.strptime(start_date, fmt)
    assert lake.end == dt.datetime.strptime(end_date, fmt)
