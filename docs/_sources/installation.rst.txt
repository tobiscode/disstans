Installation & Updates
======================

.. note::

    A Fortran 90 compiler is required to install DISSTANS and some of its
    external dependencies, e.g., the `GNU Compiler Collection (gcc)
    <https://gcc.gnu.org/>`_.

DISSTANS depends on many packages, which in turn depend on even more, so I highly
recommend using a package manager and virtual Python environments.
Some of the requirements aren't found in the Python Package Index (PyPI) that is
used by pip, but the conda channel ``conda-forge`` does. So, the easiest is to
`install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
and then go from there depending on your use case.

Minimal installation
--------------------

.. note::

    Use this installation method if you only want to use the code, without adding or
    modifying functionality, if you don't need a local copy of the documentation
    (including sample scripts), and if you're not intending to try out the newest,
    potentially buggy, code changes. If you want to do any of these things, consider a
    full development installation as presented further down.

The easiest way to install DISSTANS is to first create a conda environment using
the specification file provided in the repository, and then using pip to download
and install DISSTANS directly from its PyPI
`project homepage <https://pypi.org/project/disstans/>`_.

.. code-block:: bash

    # download the environment file
    wget https://raw.githubusercontent.com/tobiscode/disstans/main/environment.yml
    # create the environment, including all prerequisites
    conda env create -f environment.yml
    # activate the environment
    conda activate disstans
    # install DISSTANS from the Python Package Index
    pip install disstans

Done!

Full development installation
-----------------------------

This installation method will recreate the environment that I use to write, develop
and debug DISSTANS, and then having a local copy of the entire repository (from where
the package can then be installed). This is the best way if you think you might want
to extend some of DISSTANS's functionalities, and possibly feed them back into the
main DISSTANS repository. Another benefit of this version is that you have a local
copy of the HTML documentation, and that you'll be able to track the ``development``
branch to try out new features and fixes.

.. code-block:: bash

    # clone repository to disstans/
    git clone https://github.com/tobiscode/disstans.git
    # change into folder
    cd disstans/
    # create the environment, including all development prerequisites
    conda env create -f environment-dev.yml
    # activate the environment
    conda activate disstans-dev
    # install the package into the environment
    pip install .

.. note::

    If you want to try out modifications to the code, but still be able to import
    the package as it were installed fully, use pip's `editable installs
    <https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs>`_ feature
    in the last step:

    .. code-block:: bash

        pip install -e .

    This only links the folder to the Python package installation location, and
    any changes you do will be available next time you load the package.

    This works especially well in conjunction with tracking the ``development``
    branch of the repository, which you can switch to using

    .. code-block:: bash

        git checkout development

    as the editable install will always match the current state of the local
    repository. (If you don't use the editable install feature, this command
    needs to be run before ``pip install``.)


Updates
-------

.. note::

    To be notified of new releases, consider "watching" the project on GitHub!

Depending on your installation method, the update process will look slightly different:

1. If you're on the minimal installation, just run ``pip install --upgrade disstans``
   to download and install the newest version from PyPI.
2. If you've chosen the full development installation, a simple ``git pull`` will be
   enough to match your local version with the remote one. If you're on an editable
   install, you're already done; if you're not, simply run ``pip install --upgrade .``
   to upgrade the previously-installed package.

Keep in mind that these update mechanisms might not always end up with the same version.
The PyPI repository (used in the minimal installation) is only updated when I publish a
new release with a new version number.
The ``main`` branch of the GitHub repository (which is the default repository when
cloning a repository, as done in the full installation) is supposed to track the PyPI
version, but discrepancies may arise.
