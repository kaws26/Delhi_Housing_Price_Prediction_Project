from setuptools import find_packages,setup
from typing import List

HYPHEN_DOT_E='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return a list of strings from requirements.txt
    '''
    requirements=[]
    with open(file_path) as F:
        requirements=F.readlines()
        requirements=[req.replace('/n','')  for req in requirements]
        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
        
    return requirements






setup(
    name='Delhi Housing project',
    version='1.0',
    author='Kawaljeet Singh',
    author_email='kawaljeetsingh0026@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)