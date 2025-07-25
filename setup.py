from setuptools import setup, find_packages
from pathlib import Path

from vulguard.cli import __version__

VERSION = __version__
DESCRIPTION = 'VulGuard, An Unified Tool for Evaluating Just-In-Time Vulnerability Prediction Models'

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Setting up
setup(
    name="vulguard",
    version=VERSION,
    author="duongnd",
    author_email="duong.nd215336@sis.hust.edu.vn",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "vulguard.utils": ["*.jsonl", "*.csv"],
        "vulguard.models": ["*.pkl", "*.pth"],
    },
    install_requires=Path("requirements.txt").read_text(encoding="utf-8").splitlines(),
    keywords=['python', 'vulnerability', 'prediction', 'just-in-time', 'vulnerability prediction'],
    entry_points={
        'console_scripts': ['vulguard=vulguard.cli:main'], 
    },
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
