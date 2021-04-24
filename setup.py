from setuptools import setup , find_packages

#try:    
#    import gdal
#except: 
#    print("Could not import gdal, install using conda install gdal")    
#    try:
#        import numpy
#    except:
#        print("Could not import numpy, install using conda install numpy")    

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='HAPI-Nile',
    version='1.0.4',
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
    test_suite="tests",
    classifiers=[	
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Hydrology',
    'Topic :: Scientific/Engineering :: GIS',
    "Intended Audience :: Science/Research",
    'Intended Audience :: Developers',
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
		'Parameters/10/*.tif',
        'Parameters/avg/*.tif',
        'Parameters/min/*.tif',
        'Parameters/max/*.tif'
		]
		},
	 )