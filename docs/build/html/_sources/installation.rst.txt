Installation
============

Before installing the package, make sure you have **Python 3.12 or higher** installed on your system.

To avoid any conflicts with your system's Python packages, it is recommended to create and activate a dedicated virtual environment:

.. code-block:: bash

    python -m venv /path/to/beeoptimal_env
    source /path/to/beeoptimal_env/bin/activate

Installing via PIP
------------------

You can install the package from the *Python Package Index (PyPI)* via pip:

.. code-block:: bash

   pip install beeoptimal


Installing from source
----------------------

1. Clone the repository:
   
   .. code-block:: bash

      git clone https://github.com/giuliofantuzzi/BeeOptimal.git

2. Move into the repository directory and install the package with:
   
   .. code-block:: bash

      cd BeeOptimal/
      pip install .

Optional dependencies
---------------------

In addition to the core functionalities, this package offers optional dependencies for specific use cases.

To build and work with the documentation, you can install the package with the `docs` extra:

.. code-block:: bash

    pip install beeoptimal[docs]

To use the tutorials and their required dependencies, install the package with the `tutorials` extra:

.. code-block:: bash

    pip install beeoptimal[tutorials]

To install both the documentation and the tutorials, you can use:

.. code-block:: bash

    pip install beeoptimal[docs,tutorials]

.. note::

    The same syntax can be followed when installing from source. Moreover, if you're using the `zsh` shell, you will need to wrap the extras in quotes to prevent conflicts with shell globbing, as unquoted square brackets (`[ ]`) are used for pattern matching in `zsh`.
