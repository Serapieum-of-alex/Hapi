"""hydrological-model parameter."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.request import urlretrieve

import requests
from loguru import logger

from Hapi import __file__ as hapi_root

BASE_URL = "https://api.figshare.com/v2"


class FigshareAPIClient:
    """
    A client for interacting with the Figshare API.

    Parameters
    ----------
    headers : dict, optional
        Headers to include in the API requests, by default None.

    Examples
    --------
    >>> client = FigshareAPIClient()
    """

    def __init__(self, headers: Optional[dict] = None):
        """initialize."""
        self.base_url = BASE_URL
        self.headers = headers or {"Content-Type": "application/json"}

    def send_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        binary: bool = False,
    ) -> Dict[str, int]:
        """
        Send an HTTP request to the Figshare API.

        Parameters
        ----------
        method : str
            HTTP method (e.g., 'GET', 'POST').
        endpoint : str
            API endpoint to interact with.
        data : dict, optional
            Payload to include in the request, by default None.
        binary : bool, optional
            Whether the data payload is binary, by default False.

        Returns
        -------
        dict
            The parsed JSON response from the API.

        Raises
        ------
        requests.exceptions.HTTPError
            If the API request fails.

        Examples
        --------
        >>> client = FigshareAPIClient()
        >>> response = client.send_request("GET", "articles/19999901") #doctest: +SKIP
        >>> print(response) #doctest: +SKIP
        {'files': [{'id': 35589521,
           'name': '01_TT.tif',
           'size': 1048736,
           'is_link_only': False,
           'download_url': 'https://ndownloader.figshare.com/files/35589521',
           'supplied_md5': '1ddb354132c2f7f54dec6e72bdb62422',
           'computed_md5': '1ddb354132c2f7f54dec6e72bdb62422',
           'mimetype': 'image/tiff'},
         'authors': [{'id': 11888465,
           'full_name': 'Mostafa Farrag',
           'first_name': 'Mostafa',
           'last_name': 'Farrag',
           'is_active': True,
           'url_name': 'Mostafa_Farrag',
           'orcid_id': '0000-0002-1673-0126'}],
         'figshare_url': 'https://figshare.com/articles/dataset/parameter_set-1/19999901',
         'download_disabled': False,
         ...
         'version': 2,
         'status': 'public',
         'size': 19878928,
         'created_date': '2022-06-04T14:15:43Z',
         'modified_date': '2022-06-04T14:15:44Z',
         'is_public': True,
         'is_confidential': False,
         'is_metadata_record': False,
         'confidential_reason': '',
         'metadata_reason': '',
         'license': {'value': 1,
          'name': 'CC BY 4.0',
         'id': 19999901,
         'title': 'Parameter set-1',
         'doi': '10.6084/m9.figshare.19999901.v2',
         'url': 'https://api.figshare.com/v2/articles/19999901',
         'published_date': '2022-06-04T14:15:43Z',
         'url_private_api': 'https://api.figshare.com/v2/account/articles/19999901',
         'url_public_api': 'https://api.figshare.com/v2/articles/19999901',
         'url_private_html': 'https://figshare.com/account/articles/19999901',
         'url_public_html': 'https://figshare.com/articles/dataset/parameter_set-1/19999901',
         'timeline': {'posted': '2022-06-04T14:15:43',
          'firstOnline': '2022-06-04T13:52:54'},
         }
        """
        url = f"{self.base_url}/{endpoint}"
        payload = json.dumps(data) if data and not binary else data

        try:
            response = requests.request(method, url, headers=self.headers, data=payload)
            response.raise_for_status()
            return response.json() if response.text else None
        except requests.exceptions.HTTPError as error:
            logger.error(f"HTTPError: {error}, Response: {response.text}")
            raise

    def get_article_version(self, article_id: int, version: int) -> Dict[str, int]:
        """
        Retrieve a specific version of an article from the Figshare API.

        Parameters
        ----------
        article_id : int
            The ID of the article to retrieve.
        version : int
            The version number of the article to retrieve.

        Returns
        -------
        dict
            Details of the specific version of the article.

        Raises
        ------
        requests.exceptions.HTTPError
            If the API request fails.

        Examples
        --------
        >>> client = FigshareAPIClient()
        >>> response = client.get_article_version(19999901, 1) #doctest: +SKIP
        >>> print(response) #doctest: +SKIP
        """
        endpoint = f"articles/{article_id}/versions/{version}"
        return self.send_request("GET", endpoint)

    def list_article_versions(self, article_id: int) -> List[Dict[str, int]]:
        """
        Retrieve all available versions of a specific article from the Figshare API.

        Parameters
        ----------
        article_id : int
            The ID of the article to retrieve versions for.

        Returns
        -------
        List[Dict[str, int]]:
            A list of available versions for the specified article.

        Raises
        ------
        requests.exceptions.HTTPError
            If the API request fails.

        Examples
        --------
        >>> client = FigshareAPIClient()
        >>> versions = client.list_article_versions(19999901) #doctest: +SKIP
        >>> print(versions) #doctest: +SKIP
        [{'version': 1,
          'url': 'https://api.figshare.com/v2/articles/19999901/versions/1'},
         {'version': 2,
          'url': 'https://api.figshare.com/v2/articles/19999901/versions/2'}]
        """
        endpoint = f"articles/{article_id}/versions"
        return self.send_request("GET", endpoint)


class FileManager:
    """
    Handle file operations such as downloading and saving files.

    Methods
    -------
    download_file(url: str, dest_path: Path):
        Download a file from the specified URL to the destination path.
    clear_directory(directory: Path):
        Clears all files in the specified directory.

    Examples
    --------
    >>> FileManager.download_file("http://example.com/file", Path("./downloads/file.txt"))
    >>> FileManager.clear_directory(Path("./downloads"))
    """

    @staticmethod
    def download_file(url: str, download_path: Path):
        """
        Download a file from the specified URL to the destination path.

        Parameters
        ----------
        url : str
            The URL of the file to download.
        download_path : Path
            The local file path where the file will be saved.

        Examples
        --------
        >>> FileManager.download_file("http://example.com/file", Path("./downloads/file.txt"))
        """
        download_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, download_path)
        logger.debug(f"File downloaded: {download_path}")

    @staticmethod
    def clear_directory(directory: Path):
        """
        Clear all files in the specified directory.

        Parameters
        ----------
        directory : Path
            The directory to clear.

        Examples
        --------
        >>> FileManager.clear_directory(Path("./downloads"))
        """
        if directory.exists():
            for file in directory.iterdir():
                if file.is_file():
                    file.unlink()
            logger.debug(f"Cleared directory: {directory}")


class ParameterManager:
    """
    Manages hydrological parameters and integrates with Figshare for data retrieval.

    Attributes
    ----------
    ARTICLE_IDS : list
        List of article IDs corresponding to parameter sets.
    PARAMETER_NAMES : list
        List of parameter names.
    PARAMETER_SET_ID : list
        User-friendly IDs for parameter sets (e.g., 1-10, avg, max, min).

    Methods
    -------
    get_article_details(article_id: int, version: Optional[int] = None):
        Retrieves details of an article from the Figshare API.
    list_files(article_id: int, version: Optional[int] = None):
        Lists all files in an article.
    download_files(article_id: int, dest_directory: Path, version: Optional[int] = None):
        Downloads all files in an article to the specified directory.
    get_article_id_from_friendly_id(friendly_id: Union[int, str]) -> int:
        Maps a user-friendly ID to the corresponding article ID.

    Examples
    --------
    >>> manager = ParameterManager(api_client)
    >>> files = manager.list_files(12345)
    >>> manager.download_files(12345, Path("./downloads"))
    """

    ARTICLE_IDS = [
        19999901,
        19999988,
        19999997,
        20000006,
        20000012,
        20000018,
        20000015,
        20000024,
        20000027,
        20000030,
        20153402,
        20153405,
        20362374,
    ]

    PARAMETER_NAMES = [
        "01_tt",
        "02_rfcf",
        "03_sfcf",
        "04_cfmax",
        "05_cwh",
        "06_cfr",
        "07_fc",
        "08_beta",
        "09_etf",
        "10_lp",
        "11_k0",
        "12_k1",
        "13_k2",
        "14_uzl",
        "15_perc",
        "16_maxbas",
        "17_K_muskingum",
        "18_x_muskingum",
    ]

    PARAMETER_SET_ID = list(range(1, 11)) + ["avg", "max", "min"]

    def __init__(self, api_client: FigshareAPIClient):
        """initialize."""
        self.api_client = api_client

    def get_parameter_set_details(
        self, set_id: int, version: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Retrieve details of a parameter set from the Figshare API.

        Parameters
        ----------
        set_id : int
            The ID of the parameter set to retrieve.
        version : int, optional, default is None
            The version of the parameter set.

        Returns
        -------
        Dict[str, int]:
            Details of the parameter set.

        Raises
        ------
        requests.exceptions.HTTPError
            If the API request fails.
        """
        article_id = self.get_article_id(set_id)
        endpoint = f"articles/{article_id}"
        if version:
            endpoint += f"/versions/{version}"
        return self.api_client.send_request("GET", endpoint)

    def list_files(self, set_id: int, version: Optional[int] = None):
        """
        List all files in an article.

        Parameters
        ----------
        set_id : int
            The ID of the article to list files for.
        version : int, optional
            The version of the article, by default None.

        Returns
        -------
        list
            A list of files in the article.
        """
        details = self.get_parameter_set_details(set_id, version)
        return details.get("files", [])

    def download_files(
        self, set_id: int, download_dir: Path, version: Optional[int] = None
    ):
        """
        Download all files in an article to the specified directory.

        Parameters
        ----------
        set_id : int
            The ID of the article to download files from.
        download_dir : Path
            The local directory to save the files.
        version : int, optional, by default None.
            The version of the article.

        Examples
        --------
        >>> manager.download_files(1, Path("./downloads"))
        """
        files = self.list_files(set_id, version)
        for file in files:
            dest_path = download_dir / file["name"]
            FileManager.download_file(file["download_url"], dest_path)

    def get_article_id(self, set_id: Union[int, str]) -> int:
        """
        Map a user-friendly ID (1-10, avg, max, min) to the corresponding article ID.

        Parameters
        ----------
        set_id : int or str
            The parameter set id (1-10, avg, max, min).

        Returns
        -------
        int
            The corresponding article ID.

        Raises
        ------
        ValueError
            If the friendly ID is invalid.

        Examples
        --------
        >>> manager.get_article_id(1)
        19999901
        """
        try:
            index = self.PARAMETER_SET_ID.index(set_id)
            return self.ARTICLE_IDS[index]
        except ValueError:
            raise ValueError(
                f"Invalid Parameter Set ID: {set_id}, valid IDs: {self.PARAMETER_SET_ID}"
            )


class Parameter:
    """
    A simplified interface for handling hydrological parameters.

    Attributes
    ----------
    version : int
        The version of the parameter sets to retrieve.

    Methods
    -------
    get_parameters(dest_directory: Path):
        Downloads all parameter sets to the specified directory.
    get_parameter_by_friendly_id(friendly_id: Union[int, str], dest_directory: Path):
        Downloads a specific parameter set based on a user-friendly ID.
    list_parameter_names() -> List[str]:
        Lists all parameter names.

    Examples
    --------
    >>> parameter = Parameter(version=1)
    >>> parameter.get_parameters(Path("./parameters"))
    >>> parameter.get_parameter_by_friendly_id(1, Path("./parameters"))
    """

    def __init__(self, version: int = 1):
        """initialize."""
        self.version = version
        self.api_client = FigshareAPIClient()
        self.manager = ParameterManager(self.api_client)

    def get_parameters(self, download_dir: Path):
        """
        Download all parameter sets to the specified directory.

        Parameters
        ----------
        download_dir : Path
            The directory where parameter sets will be saved.

        Examples
        --------
        >>> parameter = Parameter(version=1)
        >>> parameter.get_parameters(Path("./parameters"))
        """
        for set_id in ParameterManager.PARAMETER_SET_ID:
            self.get_parameter_set(set_id, download_dir)
            logger.debug(f"Downloaded parameter set: {set_id} to {download_dir}")

    def get_parameter_set(self, set_id: int, download_dir: Optional[Path] = None):
        """
        Download all parameter sets to the specified directory.

        Parameters
        ----------
        set_id: int
            The ID of the parameter set to download.
        download_dir : Path, optional, default is None
            The directory where parameter sets will be saved.

        Examples
        --------
        >>> parameter = Parameter(version=1)
        >>> parameter.get_parameter_set(Path("./parameters"))
        """
        if set_id not in ParameterManager.PARAMETER_SET_ID:
            raise ValueError(
                f"Invalid friendly ID: {set_id}, valid IDs: {ParameterManager.PARAMETER_SET_ID}"
            )

        if download_dir is None:
            download_dir = Path(f"{os.path.dirname(hapi_root)}/parameters/{set_id}")

        self.manager.download_files(set_id, download_dir, self.version)
        logger.debug(f"Downloaded parameter set: {set_id} to {download_dir}")

    @staticmethod
    def list_parameter_names() -> List[str]:
        """
        List all parameter names.

        Returns
        -------
        list
            A list of parameter names.

        Examples
        --------
        >>> parameter = Parameter(version=1)
        >>> names = parameter.list_parameter_names()
        >>> print(names)
        """
        return ParameterManager.PARAMETER_NAMES
