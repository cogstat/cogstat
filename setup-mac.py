# Mac-specific setup.py file
from setuptools import setup, find_packages


APP = ['__boot__.py']
APP_NAME = "CogStat"
DATA_FILES = ['cogstat', 'README.md', 'cogstat.icns']
OPTIONS = {
    'plist': {
        'CFBundleName': APP_NAME,
        'CFBundleDisplayName': APP_NAME,
        'CFBundleGetInfoString': "Simple statistics for researchers.",
        'CFBundleIdentifier': "com.krajcsi.osx.cogstat",
        'CFBundleVersion': "2.0.0",
        'CFBundleShortVersionString': "2.0.0",
        'CFBundleIconFile': "cogstat.icns",
        'NSHumanReadableCopyright': "GNU GPL 3",
        'NSRequiresAquaSystemAppearance': 'YES'
    }
}

setup(
    name=APP_NAME,
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    version='2.0.0',
    description='Simple statistics for researchers.',
    url='https://www.cogstat.org',
    author='Attila Krajcsi',
    author_email='krajcsi@gmail.com',
    include_package_data=True,
    setup_requires=['py2app'],
    python_requires='>=3.6'
)