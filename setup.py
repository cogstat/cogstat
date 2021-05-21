from setuptools import setup, find_packages

setup(name='cogstat',
      version='2.1.0',
      description='Simple statistics for researchers.',
      url='https://www.cogstat.org',
      author='Attila Krajcsi',
      author_email='krajcsi@gmail.com',
      packages=find_packages(),
      include_package_data=True,
      license='GNU GPL 3',
      install_requires=['numpy', 'scipy >=0.10', 'matplotlib >=1.2.0, !=2.0.1, !=2.0.2',
                        'pandas >=1.1.0', 'statsmodels >=0.9', 'pingouin >=0.3.5', 'scikit-posthocs',
                        'pyreadstat', 'odfpy', 'xlrd', 'openpyxl', 'pyreadr',
                        'configobj', 'python-bidi', 'IPython', 'Jupyter'],
      python_requires='>=3.6',
      extras_requires={'GUI': ['PyQt5 ==5.9.2; platform_system == "Windows"', 'PyQt5; platform_system != "Windows"']}  # On Win 8.1 versions PyQt5 >5.9.2 do not work
      )
