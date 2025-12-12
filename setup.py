from setuptools import setup, find_packages

# Function to read requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as req_file:
        return [line.strip() for line in req_file if line.strip() and not line.startswith("#")]

setup(
    name="ood_nuclei-segmentation",
    version="1.1.0",
    description="OOD-Nuclei-Segmentation-via-LNNs-with-MHN Project",
    author="Bishal Swain",
    author_email="blue.bishal@rocketmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
)
