"""hydrological-model parameter."""

import json
import os
from typing import Dict, List, Optional, Union
from urllib.request import urlretrieve

import requests
from loguru import logger
from requests.exceptions import HTTPError

import Hapi

BASE_URL = "https://api.figshare.com/v2"

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
PARAMSTER_NAMES = [
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
URL = "https://api.figshare.com/v2"
HEADERS = {"Content-Type": "application/json"}


class Parameter:
    """Parameter class."""

    def __init__(self, version: int = 1):
        """__init__.

        Parameters
        ----------
        version : int, optional
            Figshare article version. If None, selects the most recent version. default is 1
        """
        self._version = version

    @property
    def param_list(self):
        """param_list."""
        return PARAMSTER_NAMES

    @property
    def baseurl(self):
        """baseurl."""
        return URL

    @property
    def headers(self):
        """headers."""
        return HEADERS

    @property
    def article_id(self):
        """article_id."""
        return ARTICLE_IDS

    @property
    def version(self):
        """version."""
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def parameter_set_id(self) -> List[str]:
        """parameter_set_id."""
        name_list = list(range(1, 11))
        return name_list + ["avg", "max", "min"]

    @property
    def parameter_set_path(self) -> List[str]:
        """parameter_set_path."""
        name_list = list(range(1, 11))
        return [str(name) for name in name_list] + ["avg", "max", "min"]

    def _get_url(self, set_id: int, version: int = None):
        """
        Return the URL for a given parameter set and version.

        Parameters
        ----------
        set_id: int
            parameter set index (from 1 to 10, avg, max, and min)
        version: int, default is None
            Figshare article version. If None, selects the most recent version

        Returns
        -------
        url: str
            URL for the request
        """
        article_id = self._get_set_article_id(set_id)
        if version is None:
            url = f"{self.baseurl}/articles/{article_id}"
        else:
            url = f"{self.baseurl}/articles/{article_id}/versions/{version}"

        return url

    @staticmethod
    def _send_request(method, url, headers, data=None, binary=False):
        """issue_request.

            Wrapper for HTTP request.

        Parameters
        ----------
        method : str
            HTTP method. One of GET, PUT, POST or DELETE
        url : str
            URL for the request
        headers: dict
            HTTP header information
        data: dict
            Figshare article data
        binary: bool
            Whether data is binary or not

        Returns
        -------
        response_data: dict
            JSON response for the request returned as python dict
        """
        if data is not None and not binary:
            data = json.dumps(data)

        response = requests.request(method, url, headers=headers, data=data)

        try:
            response.raise_for_status()
            try:
                response_data = json.loads(response.text)
            except ValueError:
                response_data = response.content
        except HTTPError as error:
            print(f"Caught an HTTPError: {error}")
            print("Body:\n", response.text)
            raise

        return response_data

    def get_set_details(self, set_id: Union[int, str], version=None):
        """get_set_details.

            Return the details of an article with a given article ID.

        Parameters
        ----------
        set_id : [str/int]
            parameter set id [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, avg, max, min]
        version: [str/int]
            Figshare article version. If None, selects the most recent version. default is None

        Returns
        -------
        response : dict
            HTTP request response as a python dict

        Examples
        --------
        >>> par = Parameter()
        >>> set_id = 2
        >>> par.get_set_details(set_id)
        """
        url = self._get_url(set_id, version)
        response = self._send_request("GET", url, headers=self.headers)
        return response["files"]

    def list_parameters(self, set_id, version=None):
        """list_parameters.

            List all the files associated with a given article.

        Parameters
        ----------
        set_id : [str/int]
            parameter set id [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, avg, max, min]
        version : str or id, default is None
            Figshare article version. If None, the function selects the most recent version.

        Returns
        -------
        response : dict
            HTTP request response as a python dict
        """
        url = self._get_url(set_id, version)
        response = self._send_request("GET", url, headers=self.headers)
        return response["files"]

    def _retrieve_parameter_set(self, set_id, directory=None):
        """retrieveParameterSet.

            Retrieve files and save them locally.

        By default, files will be stored in the current working directory
        under a folder called figshare_<article_id> by default.
        Specify <out-path> for: <out-path>/figshare_<article_id>

        Parameters
        ----------
        set_id : [str/int]
            parameter set id [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, avg, max, min]
        directory: [str]
            path
        """
        if directory is None:
            directory = os.getcwd()

        file_list = self.list_parameters(set_id)

        os.makedirs(directory, exist_ok=True)
        logger.info(
            f"The download of the parameter set starts to the following directory: {directory}"
        )

        for file_dict in file_list:
            urlretrieve(
                file_dict["download_url"], os.path.join(directory, file_dict["name"])
            )
            logger.info(f"{file_dict['name']} has been downloaded")

    def get_parameter_set(self, set_id: Union[int, str], directory: str = None):
        """get_parameter_set.

            get_parameter_set retrieves a parameter set

        Parameters
        ----------
        set_id: [int]
            parameter set index (from 1 to 10, avg, max, and min)
        directory: [str]
            directory where the downloaded parameters are going to be saved

        Returns
        -------
        None
        """
        ind = self.parameter_set_id.index(set_id)
        if directory is not None:
            rpath = directory
        else:
            par_path = self.parameter_set_path[ind]
            rpath = f"{os.path.dirname(Hapi.__file__)}/parameters/{par_path}"
        self._retrieve_parameter_set(set_id, directory=rpath)

    def get_parameters(self):
        """get_parameters.

            get_parameters retrieves all the parameters in the default directory
            Hapi/Hapi/Parameters/...

        Returns
        -------
        None
        """
        for set_id in self.parameter_set_id:
            logger.info(
                f"Download the Hydrological parameters for the dataset-{set_id}"
            )
            self.get_parameter_set(set_id)

    def _get_set_article_id(self, set_id: int):
        """get_set_article_id.

            get_set_article_id retrieves the article id for a given parameter set

        Parameters
        ----------
        set_id: [int]
            parameter set index (from 1 to 10, avg, max, and min)

        Returns
        -------
        article_id: [int]
            article id
        """
        ind = self.parameter_set_id.index(set_id)
        return self.article_id[ind]

    def list_set_versions(self, set_id: int):
        """Return the details of an article with a given article ID.

        Parameters
        ----------
        set_id : str or int
            Figshare article ID

        Returns
        -------
        response : dict
            HTTP request response as a python dict
        """
        article_id = self._get_set_article_id(set_id)
        url = f"{self.baseurl}/articles/{article_id}/versions"
        headers = self._get_headers()
        response = self._send_request("GET", url, headers=headers)
        return response

    @staticmethod
    def _get_headers(token=None):
        """HTTP header information."""
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "token {0}".format(token)

        return headers


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
