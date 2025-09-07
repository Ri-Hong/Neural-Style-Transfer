from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gram_cuda',
    ext_modules=[
        CUDAExtension('gram_cuda', [
            'gram_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
