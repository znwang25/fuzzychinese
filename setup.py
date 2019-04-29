import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fuzzychinese",
    version="0.1.5",
    author="znwang25",
    author_email="znwang25@gmail.com",
    description="A small package to fuzzy match chinese words 中文模糊匹配",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/znwang25/fuzzychinese",
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: Chinese (Traditional)",
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Text Processing',
    ],
    keywords='NLP,fuzzy matching,Chinese word',
    package_data={'fuzzychinese': ['*.*']})
