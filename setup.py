from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name="Orcar",
    version="0.0.2",
    author="Zhongming Yu",
    author_email="zhy025@ucsd.edu",
    url="https://github.com/ShishirPatil/gorilla/",
    description="AI copilot with efficient runtime planning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    # py_modules=["cli"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    # entry_points={
    #     'console_scripts': [
    #         'orcar=cli:main',
    #     ],
    # },
)