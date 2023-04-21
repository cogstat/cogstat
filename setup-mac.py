# Mac-specific setup.py file
from setuptools import setup, find_packages


APP = ['run_cogstat_gui.py']
APP_NAME = "CogStat"
DATA_FILES = ['cogstat/resources/cogstat.icns', 'README.md', 'LICENSE.txt', 'changelog.md']
OPTIONS = {
    'optimize': 2,
    'semi_standalone': 'False',
    'argv_emulation': 'False',
    'site_packages': 'False',
    'qt_plugins': ['sqldrivers'],
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleGetInfoString': "Simple statistics for researchers.",
        'CFBundleIdentifier': "com.cogstat.org.cogstat",
        'CFBundleVersion': "2.4beta",
        'CFBundleShortVersionString': "2.4beta",
        'CFBundleIconFile': "cogstat.icns",
        'NSHumanReadableCopyright': "GNU GPL 3",
        'NSRequiresAquaSystemAppearance': 'YES',
        "LSApplicationCategoryType": 'public.app-category.education',
        "NSPrincipalClass": 'NSApplication',
        "NSHighResolutionCapable": True,
        'NSAppleScriptEnabled': False,
    },
    'packages': ['pyreadstat', 'pandas_flavor', 
                'appdirs', 'xlrd', 'pyreadr', 
                'configobj', 'ipython_genutils', 
                'odf', 'bidi', 'seaborn'],
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    version='2.4beta',
    description='Simple statistics for researchers.',
    url='https://www.cogstat.org',
    author='Attila Krajcsi',
    author_email='krajcsi@gmail.com',
    include_package_data=True,
    setup_requires=['py2app'],
    python_requires='>=3.6',
    packages = find_packages(),
)
