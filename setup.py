from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='python_parallelism',
      version="0.0.1",
      description="Image preprocessing and model training using parallelism in python",
      license="MIT",
      author="G-Gress",
      author_email="gabrielgress.ds@gmail.com",
      #url="https://github.com/G-Gress/python_parallelism",
      install_requires=requirements,
      packages=find_packages(),
      include_package_data=True)
