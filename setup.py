import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resampler",
    version="1.0.1",
    description="tensorflow bilinear resampler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kor01/resampler",
    packages=['resampler', 'resampler.python', 'resampler.python.ops'],
    package_data={'resampler': ['python/ops/_resampler_ops.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Linux",
    ],
)
