from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
	name='stochround',
	ext_modules=[
		CUDAExtension('stochround',[
			'stochastic_rounding.cpp',
			'stochastic_rounder.cu',
		])

	],
	cmdclass={
		'build_ext':BuildExtension
	}
	)