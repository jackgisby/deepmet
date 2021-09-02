Installation
============

Conda (recommended)
-------------------

Install Miniconda, follow the steps described `here <https://docs.conda.io/projects/conda/en/latest/user-guide/install>`_

Start the ``conda prompt``

* Windows: Open the ``Anaconda Prompt`` via the Start menu
* macOS or Linux: Open a ``Terminal``

Create a DeepMet specific ``conda`` environment.
This will install a the dependencies required to run ``deepmet``::

    $ conda create --yes --name deepmet -c conda-forge -c bioconda -c computational-metabolomics

.. note::

    * The installation process will take a few minutes.
    * Feel free to use a different name for the Conda environment

    You can use the following command to remove a conda environment::

        $ conda env remove -y --name deepmet

    This is only required if something has gone wrong in the previous step.

Activate the ``deepmet`` environment::

    $ conda activate deepmet

To test your ``deepmet`` installation, in your Conda Prompt, run the command::

    $ deepmet --help

or::

    $ python
    import deepmet

Close and deactivate the ``deepmet`` environment when you’re done::

    $ conda deactivate


PyPi
----

Install the current release of ``deepmet`` with ``pip``::

    $ pip install deepmet

.. note::

    * The installation process will take a few minutes.

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade deepmet

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user deepmet

Alternatively, you can manually download ``deepmet`` from
`GitHub <https://github.com/computational-metabolomics/deepmet/releases>`_  or
`PyPI <https://pypi.python.org/pypi/deepmet>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Testing
-------
*DeepMet* uses the Python ``unittest`` testing package.