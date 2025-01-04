import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from Hapi.parameters.parameters import FigshareAPIClient, FileManager, Parameter


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


class TestFigshareAPIClient:

    def test_figshare_api_client_get_article(self):
        """
        Test an actual API call to Figshare's API to retrieve an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        response = client.send_request("GET", "articles/19999901")

        # Check basic response structure
        assert isinstance(response, dict), "Response should be a dictionary."
        assert "id" in response, "Response should include an 'id' key."
        assert (
            response["id"] == 19999901
        ), "The article ID should match the requested ID."

    def test_figshare_api_client_get_article_version(self):
        """
        Test an actual API call to retrieve a specific version of an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        response = client.get_article_version(article_id=19999901, version=1)

        # Check basic response structure
        assert isinstance(response, dict), "Response should be a dictionary."
        assert "id" in response, "Response should include an 'id' key."
        assert "version" in response, "Response should include a 'version' key."
        assert (
            response["version"] == 1
        ), "The version should match the requested version."

    def test_figshare_api_client_list_article_versions(self):
        """
        Test an actual API call to retrieve all available versions of an article.

        This test requires internet access and valid article IDs.
        """
        client = FigshareAPIClient()
        versions = client.list_article_versions(19999901)

        # Check basic response structure
        assert isinstance(versions, list), "Response should be a list."
        assert len(versions) > 0, "There should be at least one version."
        for version in versions:
            assert (
                "version" in version
            ), "Each version entry should include a 'version' key."

    def test_figshare_api_client_invalid_article(self):
        """
        Test how the API client handles an invalid article ID.
        """
        client = FigshareAPIClient()

        with pytest.raises(Exception):
            client.send_request("GET", "articles/0")

    def test_figshare_api_client_no_content(self):
        """
        Test the API client for an endpoint with no content.
        """
        client = FigshareAPIClient()
        response = client.send_request(
            "GET", "articles"
        )  # Assuming this endpoint exists but has no content

        assert response is not None, "Response should not be None."
        assert isinstance(response, list), "Response should be a list."


class TestFileManager:

    @pytest.fixture
    def temp_directory(self, tmp_path):
        """Fixture to create a temporary directory for testing."""
        return tmp_path

    def test_directory_creation(self, temp_directory):
        """Test that a directory is created if it doesn't exist."""
        new_dir = temp_directory / "new_subdir"
        file_path = new_dir / "example.txt"

        FileManager.download_file("https://www.example.com", file_path)

        assert new_dir.exists(), "The directory should be created."
        assert file_path.exists(), "The file should be created in the new directory."

    def test_clear_directory(self, temp_directory):
        """Test clearing all files in a directory."""
        # Create sample files
        for i in range(3):
            (temp_directory / f"file_{i}.txt").write_text(f"Content {i}")

        FileManager.clear_directory(temp_directory)

        assert not any(
            temp_directory.iterdir()
        ), "The directory should be empty after clearing."

    def test_clear_empty_directory(self, temp_directory):
        """Test clearing an already empty directory."""
        FileManager.clear_directory(temp_directory)

        assert not any(temp_directory.iterdir()), "The directory should remain empty."

    def test_clear_nonexistent_directory(self):
        """Test clearing a directory that doesn't exist."""
        non_existent_dir = Path("./non_existent_directory")

        FileManager.clear_directory(non_existent_dir)

        assert (
            not non_existent_dir.exists()
        ), "The directory should not exist and should not cause errors."
