Installation
============

.. note::

    A Fortran 90 compiler is required to install DISSTANS and some of its
    external dependencies, e.g., the `GNU Compiler Collection (gcc)
    <https://gcc.gnu.org/>`_.

.. note::

    There aren't any prebuilt packages (yet?); you'll have to install DISSTANS
    from the Github repository.

DISSTANS depends on many packages, which in turn depend on even more, so I highly
recommend using a package manager and virtual Python environments.
Some of the requirements aren't found in the Python Package Index (PyPI) that is
used by pip, but the conda channel ``conda-forge`` does. So, the easiest is to
`install conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_
and then go from there depending on your use case.

Full development installation
-----------------------------

The easiest installation option is just to recreate the environment that I use to
write, develop and debug DISSTANS, and having a local version of the repository
(from where the package can then be installed). This is also the best way if you
think you might want to extend some of DISSTANS's functionalities, and possibly
feed them back into the main DISSTANS repository. Another benefit of this version
is that you have a local copy of the HTML documentation.

.. code-block:: bash

    # clone repository to disstans/
    git clone https://github.com/tobiscode/disstans.git
    # change into folder
    cd disstans/
    # create the conda environment using the specification file,
    # installing all dependencies along the way
    conda env create -f environment.yml
    # activate the environment
    conda activate disstans
    # install the package into the environment
    pip install .

Done! You can now have a look at the tutorials to make sure the installation worked
and to get started with DISSTANS.

.. note::

    If you want to try out modifications to the code, but still be able to import
    the package as it were installed fully, use pip's `editable installs
    <https://pip.pypa.io/en/stable/cli/pip_install/#editable-installs>`_ feature:

    .. code-block:: bash

        pip install -e .

    This only links the folder to the Python package installation location, and
    any changes you do will be available next time you load the package.

Minimal environment
-------------------

If you don't want a full development environment, but still want to use conda,
you can manually create a conda environment. The packages you need are the ones in
`environment.yml <https://raw.githubusercontent.com/tobiscode/disstans/main/environment.yml>`_
without the ``# optional`` comment at the end. Be sure to use the ``conda-forge``
channel for everything, since there are dependencies to be installed as well:

.. code-block:: bash

    # create the environment with all packages defined at once
    conda create -n my_env -c conda-forge --override-channels "python>=3.9" "numpy>=1.20" ...

(Alternatively, you can download the environment file, remove the unnecessary rows,
and then follow the steps in the previous section.)

Finally, use pip to install ``okada_wrapper`` (which isn't on conda) and then the
DISSTANS package after activating your new environment:

.. code-block:: bash

    # activate the environment, replacing the environment name you used
    conda activate my_env
    # install non-conda prerequisite
    pip install git+https://github.com/tbenthompson/okada_wrapper.git
    # install the DISSTANS from the remote repository
    pip install git+https://github.com/tobiscode/disstans.git

This will still temporarily download the entire DISSTANS repository, but automatically
delete it afterwards.
