from os.path import join,dirname
from setuptools import setup, find_packages

NAME='FaceBlur'

def read(fname):
    return open(join(dirname(__file__), fname)).read(encoding='utf-8')

def get_requirements(fname):
    '''Get list of requirements to install from fname.'''
    iter=(l.strip() for l in open(fname))
    les_requirements=[l for l in iter if l and not l.startswith("---extra-index-url")]

# Get the long description from the README file
long_description=read('README.md')

setup(
    name=NAME,
    author='Malek Youssef',
    author_email='youssefmalek29@gmail.com',
    description=("Faces detectection thanks to RetinaFace."),
    long_description=long_description
    version='1.0.1',
    packages=find_packages(where='faceblur')

    install_requires=get_requirements('requirements.txt'),
    extras_require={
        "gpu" :["torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html"] 
    },
    entry_points={
        'console_scripts': ['run-faceDetection=faecDetection.face.main:main']
    }
)