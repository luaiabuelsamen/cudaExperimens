import os
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import subprocess

class BuildCUDA(build_ext):
    def run(self):
        # Build the CUDA shared library inside the package folder
        src = os.path.join("vector_add", "VecAdd.cu")
        out = os.path.join("vector_add", "libvector_add.so")
        subprocess.check_call([
            "nvcc", "-Xcompiler", "-fPIC", "-shared",
            src,
            "-o", out
        ])

setup(
    name="vector_add",
    version="0.1",
    packages=find_packages(),
    cmdclass={"build_ext": BuildCUDA},
)
