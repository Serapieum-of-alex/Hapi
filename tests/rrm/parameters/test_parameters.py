import shutil
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlretrieve

import pytest

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


def test_send_request():
    # raise for status
    url = "https://ndownloader.figshare.com/files/35589995"
    parameters = Parameter()
    response = parameters._send_request("GET", url, headers=parameters.headers)
    assert isinstance(response, bytes)
    url = "https://api.figshare.com/v2/articles/19999997"
    response = parameters._send_request("GET", url, headers=parameters.headers)
    assert len(response["files"]) == 19


def test_get_set_details():
    parameters = Parameter()
    set_id = 3
    response = parameters.get_set_details(set_id)
    assert len(response) == 19
    response = parameters.get_set_details(set_id, version=1)
    assert len(response) == 19


def test_list_parameters():
    parameters = Parameter()
    response = parameters.list_parameters(1)
    assert len(response) == 19
    response = parameters.list_parameters(1, version=1)
    assert len(response) == 19


def test_retrieve_parameter_set_e2e():
    parameters = Parameter()
    set_id = 3
    path = "tests/rrm/data/parameters/test_download_set_3_v1"
    parameters._retrieve_parameter_set(set_id, directory=path)

    assert Path(path).exists()
    assert len(list(Path(path).iterdir())) == 19
    try:
        shutil.rmtree(path)
    except PermissionError:
        pass


@pytest.mark.e2e
def test_list_set_versions():
    parameters = Parameter()
    set_id = 3
    response = parameters.list_set_versions(set_id)
    assert len(response) == 1
    assert response[0]["version"] == 1
    assert (
        response[0]["url"] == "https://api.figshare.com/v2/articles/19999997/versions/1"
    )


@pytest.mark.e2e
def test_get_parameter_set():
    parameters = Parameter()
    set_id = 3
    path = "tests/rrm/data/parameters/test_get_parameter_set_3_v1"
    parameters.get_parameter_set(set_id, directory=path)
    assert Path(path).exists()
    assert len(list(Path(path).iterdir())) == 19
    shutil.rmtree(path)


@pytest.mark.mock
def test_get_parameters():
    parameters = Parameter()
    with patch("Hapi.parameters.parameters.Parameter.get_parameter_set") as mock:
        parameters.get_parameters()
        assert mock.call_count == 13
