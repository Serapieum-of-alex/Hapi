"""hydrological-model parameter."""
import json
import os
from typing import Union, List
from urllib.request import urlretrieve

import requests
from loguru import logger
from requests.exceptions import HTTPError

import Hapi

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
        return PARAMSTER_NAMES

    @property
    def baseurl(self):
        return URL

    @property
    def headers(self):
        return HEADERS

    @property
    def article_id(self):
        return ARTICLE_IDS

    @property
    def version(self):
        return self._version

    @version.setter
    def version(self, value):
        self._version = value

    @property
    def parameter_set_id(self) -> List[str]:
        name_list = list(range(1, 11))
        return name_list + ["avg", "max", "min"]

    @property
    def parameter_set_path(self) -> List[str]:
        name_list = list(range(1, 11))
        return [str(name) for name in name_list] + ["avg", "max","min"]

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
            parameter set id
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

    def list_parameters(self, article_id, version=None):
        """listFiles.

            List all the files associated with a given article.

        Parameters
        ----------
        article_id : str or int
            Figshare article ID

        version : str or id, default is None
            Figshare article version. If None, the function selects the most recent version.

        Returns
        -------
        response : dict
            HTTP request response as a python dict
        """
        if version is None:
            url = f"{self.baseurl}/articles/{article_id}/files"
            response = self.issue_request("GET", url, headers=self.headers)
            return response
        else:
            request = self.get_set_details(article_id, version)
            return request["files"]

    def retrieve_parameter_set(self, article_id, directory=None):
        """retrieveParameterSet.

            Retrieve files and save them locally.

        By default, files will be stored in the current working directory
        under a folder called figshare_<article_id> by default.
        Specify <out-path> for: <out-path>/figshare_<article_id>

        Parameters
        ----------
        article_id : str or int
            Figshare article ID
        directory: [str]
            path
        """

        if directory is None:
            directory = os.getcwd()

        # Get a list of files
        file_list = self.list_parameters(article_id)

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
        article_id = self.article_id[ind]
        if directory is not None:
            rpath = directory
        else:
            par_path = self.parameter_set_path[ind]
            rpath = f"{os.path.dirname(Hapi.__file__)}/parameters/{par_path}"
        self.retrieve_parameter_set(article_id, directory=rpath)

    def get_parameters(self):
        """get_parameters.

            get_parameters retrieves all the parameters in the default directory
            Hapi/Hapi/Parameters/...

        Returns
        -------
        None
        """
        for i in range(len(self.parameter_set_id)):
            set_id = self.parameter_set_id[i]
            logger.info(
                f"Download the Hydrological parameters for the dataset-{set_id}"
            )
            self.get_parameter_set(set_id)
