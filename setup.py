from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os

eigen_include_dir = '/usr/include/eigen3'

setup(
    name='stacking_module',
    ext_modules=[
        CppExtension(
            name='stacking_module',
            sources=['stacking.cpp'],
            include_dirs=[eigen_include_dir],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
