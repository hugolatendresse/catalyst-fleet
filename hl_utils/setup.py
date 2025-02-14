from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent.resolve()

# Read the README file for a long description (if you have one)
README = (HERE / "README.md").read_text() if (HERE / "README.md").exists() else ""

setup(
    name="hl_utils",          
    version="0.1.0",                
    description="A set of utils developed by Hugo to work on TVM",
    author="Hugo Latendresse",            
    author_email="hlatendr@cmu.edu",  
    packages=find_packages(),         
    install_requires=[                # List your package dependencies here
        # 'dependency_one>=1.0',
        # 'dependency_two>=2.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change license if needed
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',          # Specify your required Python versions
)
