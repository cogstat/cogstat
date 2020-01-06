from setuptools import setup, find_packages

setup(name='cogstat',
    version='1.9.0dev',
    description='Simple statistics for researchers.',
    url='https://www.cogstat.org',
    author='Attila Krajcsi',
    author_email='krajcsi@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    license='GNU GPL 3',
    install_requires=['numpy', 'pandas >= 0.14.1', 'scipy >= 0.10', 'statsmodels >= 0.9', 'scikit-posthocs',
                      'matplotlib >= 1.2.0 !=2.0.1 != 2.0.2', 'IPython', 'Jupyter', 'savReaderWriter', 'configobj',
                      'python-bidi']
    # You cannot set the pyqt and R dependency here, because it is not available with pip installation
    )
