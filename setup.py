import setuptools
import deepmet


def main():

    setuptools.setup(
        name="deepmet",
        version=deepmet.__version__,
        description="",
        long_description=open('README.md').read(),
        url="https://github.com/computational-metabolomics/deepmet",
        license="GPLv3",
        platforms=['Windows, UNIX'],
        keywords=['Metabolomics', 'Lipidomics', 'Mass spectrometry', 'Metabolite Identification'],
        packages=setuptools.find_packages(),
        python_requires='>=3.6',
        include_package_data=True,
        classifiers=[
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering :: Bio-Informatics",
          "Topic :: Scientific/Engineering :: Chemistry",
          "Topic :: Utilities",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
        ],
        entry_points={
         'console_scripts': [
             'deepmet = deepmet.__main__:main'
         ]
        }
    )


if __name__ == "__main__":
    main()
