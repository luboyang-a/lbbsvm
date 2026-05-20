import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import numpy

ext_modules = [
    Extension(
        'lbbsvm',
        sources=[
            'src/list.cpp',
            'src/lrucache.cpp',
            'src/qmatrix.cpp',
            'src/solver.cpp',
            'src/model.cpp',
            'src_python/bind.cpp'
        ],
        include_dirs=[
            'include',
            pybind11.get_include(),
            numpy.get_include()
        ],
        language='c++'
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        opts = []
        link_opts = []  # 🚨 必须新增链接参数列表！

        if sys.platform == 'win32':
            opts.append('/O2')
            opts.append('/std:c++14')
            opts.append('/arch:AVX2')
            opts.append('/openmp')
        else:
            opts.append('-O3')
            opts.append('-std=c++14')
            opts.append('-fPIC')
            opts.append('-march=native')
            opts.append('-mavx2')
            opts.append('-mfma')
            opts.append('-fopenmp')
            link_opts.append('-fopenmp')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
            
        super().build_extensions()

setup(
    name='lbbsvm',
    version='0.1.0',
    author='YourName',
    description='lbbsvm',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.6',
)