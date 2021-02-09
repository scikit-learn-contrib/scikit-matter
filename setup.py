#!/usr/bin/env python3
from setuptools import Extension, setup
from distutils.command.build import build
import re
import os


class build_cython(build):
    # Allow to install this package without requiring previous install of
    # numpy/cython. Inspired by https://stackoverflow.com/a/60730258/4692076
    def finalize_options(self):
        super().finalize_options()
        import numpy

        use_cython = True
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())

            for path in extension.sources:
                if not os.path.exists(path):
                    use_cython = False

        if use_cython:
            from Cython.Build import cythonize

            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules, language_level=3
            )
        else:
            # we are in a source build, only .c files are available
            for extension in self.distribution.ext_modules:
                sources = []
                for path in extension.sources:
                    if path.endswith(".pyx"):
                        sources.append(path.replace(".pyx", ".c"))
                    else:
                        sources.append(path)

                extension.sources = sources


extensions = [
    Extension(
        name="skcosmo.feature_selection.fps",
        sources=[
            "skcosmo/feature_selection/fps.pyx",
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open("skcosmo/__init__.py").read()
).group(1)

setup(
    version=__version__,
    ext_modules=extensions,
    cmdclass={"build": build_cython},
)
