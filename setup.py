from setuptools import find_packages, setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# requirements = [line.strip() for line in open("requirement.txt").readlines()]
# requirements = requirements[1:]

test_requirements = ['pytest>=3', ]

setup(
    name="HAPI-Nile",
    version="1.0.5",
    description="Distributed Hydrological model",
    author="Mostafa Farrag",
    author_email="moah.farag@gmail.come",
    url="https://github.com/MAfarrag/Hapi",
    keywords=["Hydrology", "Distributed hydrological model"],
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    license="GNU General Public License v3",
    zip_safe=False,
    packages=find_packages(include=['Hapi', 'Hapi.*']),
    test_suite="tests",
    tests_require=test_requirements,
    # install_requires=requirements,
    entry_points={
        'console_scripts': [
            'Hapi=Hapi.cli:main',
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Natural Language :: English",
        'Programming Language :: Python :: 3',
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    include_package_data=True,
    package_data={
        "": [
            "Parameters/01/*.tif",
            "Parameters/02/*.tif",
            "Parameters/03/*.tif",
            "Parameters/04/*.tif",
            "Parameters/05/*.tif",
            "Parameters/06/*.tif",
            "Parameters/07/*.tif",
            "Parameters/08/*.tif",
            "Parameters/09/*.tif",
            "Parameters/10/*.tif",
            "Parameters/avg/*.tif",
            "Parameters/min/*.tif",
            "Parameters/max/*.tif",
        ]
    },
)
