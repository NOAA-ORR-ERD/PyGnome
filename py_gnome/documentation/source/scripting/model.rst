.. include:: ../links.rst

.. _scripting_model:

.. note:: This section is currently incomplete -- see the API reference for details.

The PyGNOME Model
=================

The :class:`gnome.model.Model` class in the overall interface to the gnome system -- it manages all the elements, movers, outputters, etc used to drive the model. When writting scripts, a ``Model`` instance is used to manage how the overall model is run.

Initialization
--------------

Most of the model parameters can be set after creating a ``Model`` object. But there are common ones to set for the usual cases.

The common parameters set when creating a model instance are::

    name='Model',
    time_step=datetime.timedelta(seconds=900),
    start_time=datetime.datetime(2023, 3, 17, 15, 0),
    duration=datetime.timedelta(days=1),
    uncertain=False,

The start_time defaults to the current time, which will usually need to be adjusted. The other defaults
may work, but it is helpful to include them in the initialization to make it easier to adjust them.

By default the model uncertainty is off. To run with uncertainty included set uncertain=True.
Wind, currents, and diffusion all have uncertainty parameters with default values.
You can set the coefficients that control the size and distribution of the uncertainty on the individual movers.
The uncertainty only applies to the model transport. Uncertainty for weathering is under development.

To run the model backwards, set both the time_step and the duration to negative values

Configuring the Model
---------------------


Running the Model
-----------------


Capturing Results As the Model Runs
...................................

Sometimes we want to do this iteratively step-by-step to view results
along the way without outputting to a file.
There are some helper utilities to extract data associated with the elements.
These data include properties such as mass, age, and position or weathering information such as the mass of oil evaporated (if the simulation has specified an oil type rather than a conservative substance as in this example).

For example, if we want to extract the element positions as a function of time, we can use the :func:`gnome.model.get_spill_property` convenience function, as shown below::

    x=[]
    y=[]
    for step in model:
        positions = model.get_spill_property('positions')
        x.append(positions[:,0])
        y.append(positions[:,1])

To see a list of properties associated with elements use::

    model.list_spill_properties()

Note: this list will be empty until after the model has been run for at least one timestep.


Examining the Results
---------------------

Save and Reload a Model Setup
-----------------------------

.. todo:: Create a new page to talk about Save Files?

The PyGNOME uses "save files" as a way to save a model setup to use again or to share with another user.
The save files are a zip file that contain all the configuration information as JSON files and any needed data files all in one archive.
They are usually given the `.gnome` file extension but they are, in fact, regular zip files.

Save files are used by the WebGNOME application, so that users can save and reload a model setup that they have created via the interactive GUI interface.
For the most part, when you are running ``gnome`` via Python scripts, you don't need to use save files, as your script can rebuild the model when it runs.
However, there are use cases for save files with scripting, particularly if you want to work on the same model via scripting and WebGNOME.

A model can be created from a save file via the :func:`scripting.load_model()` function:

.. code-block:: python

  import gnome.scripting as gs
  model = gs.load_model("the_savefile.gnome")

You can save out a configured model using the save method:

.. code-block:: python

  model.save("the_savefile.gnome")

The resulting file can be loaded into WebGNOME, or a PyGNOME script.


