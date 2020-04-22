from setuptools import setup , find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name='HAPI-Nile',
    #packages=['Hapi'],
    version='0.1.0',
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
    'Programming Language :: Python ',
    	#'Programming Language :: Python :: Implementation :: CPython',
    'Topic :: Software Development',
    ]
	 )