from setuptools import setup, find_packages

setup(
    name='RUN_pchem_fitting',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
		'scipy',
		'matplotlib',
		'ipywidgets'
    ],
    url='https://github.com//RUN-pchem/fitting',
    author='Colin Kinz-Thompson',
    author_email='colin.kinzthompson@rutgers.edu',
    description='Interactive Data Fitting Library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)