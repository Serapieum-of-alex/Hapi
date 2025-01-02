from pathlib import Path
from Hapi.parameters.parameters import Parameter


def test_constructor():
    parameters = Parameter()
    assert isinstance(parameters, Parameter)
    assert len(parameters.article_id) == 13
    assert len(parameters.parameter_set_id) == 13
    assert len(parameters.parameter_set_path) == 13
    assert len(parameters.param_list) == 18
    assert parameters.baseurl == "https://api.figshare.com/v2"
