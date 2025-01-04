import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from Hapi.parameters.parameters import (
    FigshareAPIClient,
    FileManager,
    Parameter,
    ParameterManager,
)


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
    try:
        shutil.rmtree(path)
    except PermissionError:
        pass


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


@pytest.mark.mock
class TestParameterManagerMock:

    @pytest.fixture
    def mock_api_client(self):
        """Fixture to provide a mock API client."""
        return MagicMock(spec=FigshareAPIClient)

    @pytest.fixture
    def parameter_manager(self, mock_api_client):
        """Fixture to provide a parameter manager with a mocked API client."""
        return ParameterManager(api_client=mock_api_client)

    def test_get_parameter_set_details(self, parameter_manager, mock_api_client):
        """Test retrieving details of an article."""
        mock_api_client.send_request.return_value = {"id": 1, "title": "Sample Article"}
        result = parameter_manager.get_parameter_set_details(set_id=1)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert result["id"] == 1, "The article ID should match."
        assert result["title"] == "Sample Article", "The article title should match."

    def test_list_files(self, parameter_manager, mock_api_client):
        """Test listing all files in an article."""
        mock_api_client.send_request.return_value = {
            "files": [{"name": "file1.txt"}, {"name": "file2.txt"}]
        }
        result = parameter_manager.list_files(set_id=1)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert len(result) == 2, "There should be two files."
        assert result[0]["name"] == "file1.txt", "The first file name should match."

    def test_download_files(self, parameter_manager, mock_api_client, tmp_path):
        """Test downloading files from an article."""
        mock_api_client.send_request.return_value = {
            "files": [
                {"name": "file1.txt", "download_url": "http://example.com/file1"},
                {"name": "file2.txt", "download_url": "http://example.com/file2"},
            ]
        }
        mock_download = MagicMock()
        FileManager.download_file = mock_download

        parameter_manager.download_files(set_id=1, dest_directory=tmp_path)

        mock_api_client.send_request.assert_called_once_with("GET", "articles/19999901")
        assert mock_download.call_count == 2, "Two files should be downloaded."
        mock_download.assert_any_call(
            "http://example.com/file1", tmp_path / "file1.txt"
        )
        mock_download.assert_any_call(
            "http://example.com/file2", tmp_path / "file2.txt"
        )

    def test_get_article_id_from_friendly_id(self, parameter_manager):
        """Test mapping a friendly ID to an article ID."""
        result = parameter_manager._get_article_id(1)
        assert (
            result == ParameterManager.ARTICLE_IDS[0]
        ), "The article ID should match the corresponding friendly ID."

        result = parameter_manager._get_article_id("max")
        assert (
            result == ParameterManager.ARTICLE_IDS[-2]
        ), "The article ID should match the corresponding friendly ID."

        with pytest.raises(ValueError):
            parameter_manager._get_article_id("invalid_id")

    def test_get_article_details_with_version(self, parameter_manager, mock_api_client):
        """Test retrieving details of an article with a specific version."""
        mock_api_client.send_request.return_value = {
            "id": 1,
            "version": 2,
            "title": "Sample Article",
        }
        result = parameter_manager.get_parameter_set_details(set_id=1, version=2)

        mock_api_client.send_request.assert_called_once_with(
            "GET", "articles/19999901/versions/2"
        )
        assert result["version"] == 2, "The version should match the requested version."
        assert result["title"] == "Sample Article", "The article title should match."


@pytest.mark.integration
class TestParameterManagerIntegration:

    @pytest.fixture
    def real_api_client(self):
        """Provide an actual API client for integration testing."""
        return FigshareAPIClient()

    @pytest.fixture
    def int_test_dir(self, tmp_path):
        """Provide a temporary directory for testing file downloads."""
        return tmp_path / "integration_test_files"

    @pytest.fixture
    def parameter_manager(self, real_api_client):
        """Provide a ParameterManager with a real API client."""
        return ParameterManager(api_client=real_api_client)

    def test_integration_get_parameter_set_details(self, parameter_manager):
        """Integration test for retrieving article details."""
        set_id = 1
        article_id = 19999901
        details = parameter_manager.get_parameter_set_details(set_id)

        assert isinstance(details, dict), "Details should be a dictionary."
        assert "id" in details, "Article details should include an 'id' key."
        assert (
            details["id"] == article_id
        ), "The article ID should match the requested ID."

    def test_integration_list_files(self, parameter_manager):
        """Integration test for listing files in an article."""
        set_id = 1
        files = parameter_manager.list_files(set_id)

        assert isinstance(files, list), "Files should be returned as a list."
        assert len(files) > 0, "The article should have at least one file."
        assert "name" in files[0], "Each file should have a 'name' key."

    # def test_integration_download_files(self, parameter_manager):
    #     """Integration test for downloading files from an article."""
    #     set_id = 1
    #     int_test_dir = Path("tests/rrm/data/parameters/download_files")
    #     int_test_dir.mkdir(parents=True, exist_ok=True)
    #
    #     parameter_manager.download_files(set_id, int_test_dir)
    #
    #     downloaded_files = list(int_test_dir.iterdir())
    #     assert (
    #         len(downloaded_files) == 19
    #     ), "Files should be downloaded to the specified directory."

    def test_integration_get_article_id(self, parameter_manager):
        """Integration test for mapping a friendly ID to an article ID."""
        article_id = parameter_manager._get_article_id(1)
        assert (
            article_id == ParameterManager.ARTICLE_IDS[0]
        ), "The friendly ID should map to the correct article ID."

        article_id = parameter_manager._get_article_id("avg")
        assert (
            article_id == ParameterManager.ARTICLE_IDS[-3]
        ), "The friendly ID 'avg' should map to the correct article ID."

    def test_integration_get_article_details_with_version(self, parameter_manager):
        """Integration test for retrieving article details with a specific version."""
        set_id = 1
        version = 1
        details = parameter_manager.get_parameter_set_details(set_id, version=version)

        assert isinstance(details, dict), "Details should be a dictionary."
        assert "version" in details, "Article details should include a 'version' key."
        assert (
            details["version"] == version
        ), "The version should match the requested version."
