from setuptools import setup , find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='HAPI-Nile',
    version='1.0.3',
    description='Distributed Hydrological model',
    author='Mostafa Farrag',
    author_email='moah.farag@gmail.come',
    url='https://github.com/MAfarrag/HAPI',
    keywords=['Hydrology', 'Distributed hydrological model'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    License="MIT" ,
    zip_safe=False, 
    packages=find_packages(),
    classifiers=[	
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Software Development',
    ],
	include_package_data=True,
	package_data={'': [
		'Parameters/01/*.tif',
		'Parameters/02/*.tif',
		'Parameters/03/*.tif',
		'Parameters/04/*.tif',
		'Parameters/05/*.tif',
		'Parameters/06/*.tif',
		'Parameters/07/*.tif',
		'Parameters/08/*.tif',
		'Parameters/09/*.tif',
		'Parameters/10/*.tif'
		]
		},
	 )