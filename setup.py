import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess

class BuildCUDA(build_ext):
    def build_extension(self, ext):
        sources = ext.sources
        output_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        os.makedirs(output_dir, exist_ok=True)
        output_file = self.get_ext_fullpath(ext.name)

        cu_file = None
        cpp_file = None
        for src in sources:
            if src.endswith('.cu'):
                cu_file = src
            elif src.endswith('.cpp'):
                cpp_file = src

        cu_obj = os.path.join(output_dir, 'VecAdd.cu.o')
        cpp_obj = os.path.join(output_dir, 'VecAdd.cpp.o')

        # Compile CUDA file
        if cu_file:
            subprocess.check_call([
                "nvcc", "-c", cu_file, "-o", cu_obj, "-Xcompiler", "-fPIC"
            ])

        # Compile C++ file
        if cpp_file:
            subprocess.check_call([
                "g++", "-c", cpp_file, "-o", cpp_obj, "-fPIC"
            ])

        # Link both object files into a shared library
        objs = []
        if cu_file:
            objs.append(cu_obj)
        if cpp_file:
            objs.append(cpp_obj)
        subprocess.check_call([
            "g++", "-shared", "-o", output_file
        ] + objs + ["-lcudart", "-L/usr/local/cuda/lib64"])

        # Copy and rename the .so to vector_add/libvector_add.so
        import shutil
        package_dir = os.path.join(os.path.dirname(__file__), "vector_add")
        dest_file = os.path.join(package_dir, "libvector_add.so")
        shutil.copy2(output_file, dest_file)
        # Copy and rename the .so to vector_add/libvector_add.so
        import shutil
        package_dir = os.path.join(os.path.dirname(__file__), "vector_add")
        dest_file = os.path.join(package_dir, "libvector_add.so")
        shutil.copy2(output_file, dest_file)

ext_modules = [
    Extension(
        "vector_add.libvector_add",  # module name inside package vector_add
        sources=["vector_add/VecAdd.cu", "vector_add/VecAdd.cpp"],
        language="c++"
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
