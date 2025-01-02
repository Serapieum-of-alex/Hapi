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
    assert parameters.headers == {"Content-Type": "application/json"}
    assert parameters.version == 1
    parameters.version = 2
    assert parameters.version == 2


def test_get_url():
    parameters = Parameter()
    set_id = 1
    url = parameters._get_url(set_id)
    assert url == "https://api.figshare.com/v2/articles/19999901"
    version = 1
    url = parameters._get_url(set_id, version)
    assert url == "https://api.figshare.com/v2/articles/19999901/versions/1"

