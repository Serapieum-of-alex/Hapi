"""Download Rainfall runoff model parameter."""
from Hapi.parameters.parameters import Parameter


def download():
    """Download Parameters."""
    print("downloading parameters")
    par = Parameter()
    par.get_parameters()


if __name__ == "__main__":
    download()
