import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class BuildCUDA(build_ext):
    def build_extension(self, ext):
        src = os.path.join(*ext.sources)
        output_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        os.makedirs(output_dir, exist_ok=True)
        output_file = self.get_ext_fullpath(ext.name)
        subprocess.check_call([
            "nvcc", "-Xcompiler", "-fPIC", "-shared",
            src,
            "-o", output_file,
        ])
        # Copy and rename the .so to vector_add/libvector_add.so
        import shutil
        package_dir = os.path.join(os.path.dirname(__file__), "vector_add")
        dest_file = os.path.join(package_dir, "libvector_add.so")
        shutil.copy2(output_file, dest_file)

ext_modules = [
    Extension(
        "vector_add.libvector_add",  # module name inside package vector_add
        sources=["vector_add/VecAdd.cu"],
    )
]

setup(
    name="vector_add",
    version="0.1",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCUDA},
    zip_safe=False,
)
