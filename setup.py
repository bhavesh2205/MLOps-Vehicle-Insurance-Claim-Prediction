from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    """
    This function will return the list of requirements
    """
    requirements:List[str]=[]
    try:
        with open("requirements.txt", "r") as file:
            lines=file.readlines()
            
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements.append(requirement)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    return requirements


setup(
    name="Vehicle_Insurance_Claim_Prediction",
    version="0.0.1",
    author="Bhavesh Patil",
    author_email="imbhavesh7@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)