from setuptools import find_packages , setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns a list of installed packages.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        # Remove the local package installation trigger from the list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='movie_recommendation_system',
    version='0.0.1',
    author='Dhruvit',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)