from setuptools import setup, find_packages

setup(name='cogstat',
      version='2.4.1',
      description='Simple statistics for researchers.',
      url='https://www.cogstat.org',
      author='Attila Krajcsi',
      author_email='krajcsi@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      license='GNU GPL 3',
      install_requires=['numpy', 'scipy >=1.9, !=1.10', 'matplotlib >=1.5.0, !=2.0.1, !=2.0.2',
                        'pandas >=1.5.0', 'statsmodels >=0.13.0', 'pingouin >=0.3.12', 'scikit-posthocs >= 0.7.0',
                        'pyreadstat >=1.1.5', 'odfpy', 'xlrd', 'openpyxl >=3.0.7', 'pyreadr >=0.4.5',
                        'appdirs', 'python-bidi', 'IPython', 'Jupyter', 'scikit-learn', 'svgutils'],
      python_requires='>=3.6',
      entry_points={'console_scripts': ['cogstat=cogstat.cogstat_gui:main']},
      extras_requires={'GUI': ['PyQt5']}
)
