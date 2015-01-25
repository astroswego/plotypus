Installation
============

To install plotypus, you will first need to install some dependencies. Read
the documentation for installing each package, and ensure that they are all
built for Python **3**. plotypus is not compatible with Python 2.

* `Python 3 <https://www.python.org/>`_
* `setuptools <http://pythonhosted.org/setuptools/>`_
* `numpy <http://www.numpy.org/>`_
* `scipy <http://scipy.org/>`_
* `scikit-learn <http://scikit-learn.org/stable/>`_
* `matplotlib <http://matplotlib.org/>`_


Once you have installed all dependencies, download the source from github::

    git clone https://github.com/astroswego/plotypus.git

and install::

  > cd plotypus
  > python3 setup.py install

If you would like to stay up-to-date with development, replace the last command with::

  > python3 setup.py develop

Then, when you want to update, simply run::

  > git pull

.. note::

   Depending on your setup, you may need to run *setup.py* as root, or
   install locally by appending the ``--user`` flag.
