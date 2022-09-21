"""Hydrological model parameter."""
import json
import os
from typing import Union
from urllib.request import urlretrieve

import requests
from loguru import logger
from requests.exceptions import HTTPError

import Hapi


class Parameter:
    """Parameter class."""

    def __init__(self):
        self.parameterset_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "avg", "max", "min"]
        self.parameterser_path = [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "avg",
            "max",
            "min",
        ]
        self.article_id = [
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
        self.baseurl = "https://api.figshare.com/v2"
        self.headers = {"Content-Type": "application/json"}
        self.param_list = [
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
        pass

    # def get_parameter_set(self, parameter_set_id, directory=None):

    def issueRequest(self, method, url, headers, data=None, binary=False):
        """Wrapper for HTTP request.

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
            print("Caught an HTTPError: {}".format(error))
            print("Body:\n", response.text)
            raise

        return response_data

    def getSetDetails(self, set_id, version=None):
        """getArticleDetails.

            Return the details of an article with a given article ID.

        Parameters
        ----------
        set_id : str or id
            parameter set id
        version : str or id, default is None
            Figshare article version. If None, selects the most recent version.

        Returns
        -------
        response : dict
            HTTP request response as a python dict

        Examples
        --------
        >>> par = Parameter()
        >>> set_id = 2
        >>> par.getSetDetails(set_id)
        """
        ind = self.parameterset_id.index(set_id)
        article_id = self.article_id[ind]

        if version is None:
            url = f"{self.baseurl}/articles/{article_id}"
        else:
            url = f"{self.baseurl}/articles/{article_id}/versions/{version}"

        response = self.issueRequest("GET", url, headers=self.headers)
        return response

    def listParameters(self, article_id, version=None):
        """listFiles.

            List all the files associated with a given article.

        Parameters
        ----------
        article_id : str or int
            Figshare article ID

        version : str or id, default is None
            Figshare article version. If None, selects the most recent version.

        Returns
        -------
        response : dict
            HTTP request response as a python dict
        """
        if version is None:
            url = f"{self.baseurl}/articles/{article_id}/files"
            response = self.issueRequest("GET", url, headers=self.headers)
            return response
        else:
            request = self.getSetDetails(article_id, version)
            return request["files"]

    def retrieveParameterSet(self, article_id, directory=None):
        """retrieveParameterSet.

            Retrieve files and save them locally.

        By default, files will be stored in the current working directory
        under a folder called figshare_<article_id> by default.
        Specify <outpath> for: <outpath>/figshare_<article_id>

        Parameters
        ----------
        article_id : str or int
            Figshare article ID
        """

        if directory is None:
            directory = os.getcwd()

        # Get list of files
        file_list = self.listParameters(article_id)

        # dir0 = os.path.join(directory, f"figshare_{article_id}/")
        os.makedirs(directory, exist_ok=True)  # This might require Python >=3.2
        logger.info(
            f"The download of the parameter set starts to the following directory: {directory}"
        )

        for file_dict in file_list:
            urlretrieve(
                file_dict["download_url"], os.path.join(directory, file_dict["name"])
            )
            logger.info(f"{file_dict['name']} has been downloaded")

    def getParameterSet(self, set_id: Union[int, str], directory: str = None):
        """getParameterSet.

            getParameterSet retrieve parameter set

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
        ind = self.parameterset_id.index(set_id)
        article_id = self.article_id[ind]
        if directory is not None:
            rpath = directory
        else:
            par_path = self.parameterser_path[ind]
            rpath = f"{os.path.dirname(Hapi.__file__)}/parameters/{par_path}"
        self.retrieveParameterSet(article_id, directory=rpath)

    def getParameters(self):
        """getParameters.

            getParameters retrieves all the parameters in the default directory
            Hapi/Hapi/Parameters/...

        Returns
        -------
        None
        """
        for i in range(len(self.parameterset_id)):
            set_id = self.parameterset_id[i]
            logger.info(f"Dowload the Hydrological parameters for the dataset-{set_id}")
            self.getParameterSet(set_id)
