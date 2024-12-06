from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="Orcar",
    version="0.0.2",
    author="Zhongming Yu",
    author_email="zhy025@ucsd.edu",
    # url="https://github.com/ShishirPatil/gorilla/",
    include_package_data=True,
    description="AI copilot with efficient runtime planning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "": ["templates/*"],
    },
    py_modules=["cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "orcar=cli:main",
        ],
    },
)
