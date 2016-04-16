from setuptools import setup, find_packages

setup(name='cogstat',
    version='1.4.0',
    description='Simple statistics for researchers.',
    url='https://github.com/cogstat/cogstat',
    author='Attila Krajcsi',
    author_email='krajcsi@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    license='GNU GPL 3',
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'IPython']  # TODO how to set Qt and R dependecies?
    )
