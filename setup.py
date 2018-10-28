from setuptools import setup, find_packages

setup(name='cogstat',
    version='1.8.0dev',
    description='Simple statistics for researchers.',
    url='https://www.cogstat.org',
    author='Attila Krajcsi',
    author_email='krajcsi@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    license='GNU GPL 3',
    install_requires=['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'IPython', 'Jupyter', 'savReaderWriter',
                      'configobj', 'python-bidi','rpy2']
    # You cannot set the pyqt and R dependency here, because it is not available with pip installation
    )
