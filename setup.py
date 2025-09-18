import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="raystrack",
        version="1.0.0",
        description="Lightweight Monte-Carlo view-factor solver with CPU, CUDA and BVH paths",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        author="Philip Balizki",
        author_email="philip.balizki@web.de",
        license="GPL-3.0-only",
        python_requires=">=3.9",
        packages=setuptools.find_packages(where="src") or setuptools.find_packages(),
        package_dir={"": "src"} if (setuptools.find_packages(where="src")) else {},
        install_requires=[
            "numpy>=1.24,<3.0",
            "numba>=0.59,<0.60",
        ],
        project_urls={
            "Homepage": "https://github.com/Philip-rptu/raystrack",
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "Operating System :: OS Independent",
        ]
    )
