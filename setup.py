from setuptools import setup, find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """
    Returns a list of required packages.
    """
    requirement_list: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != "-e .":
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("Requirements file not found.")
    return requirement_list


setup(
    name="Vehicle Insurance Claim Prediction",
    version="0.0.1",
    author="Bhavesh Patil",
    author_email="imbhavesh7@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
