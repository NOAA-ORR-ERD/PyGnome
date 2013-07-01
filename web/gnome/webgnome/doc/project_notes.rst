Project Notes
=============

Miscellaneous project notes with no obvious home.

.. contents:: `Table of contents`
   :depth: 2


Creating a Location File
------------------------

Location files are directories in webgnome's ``location_files`` directory that
contains JSON files and data needed to load a py_gnome model from disk.

There are two files that every location file needs to contain:

    **config.json**: this contains UI-related details like the latitude and
    longitude of the location for display on a map, and its name

    **location.json**: this is the serialized py_gnome model for the location
    file and contains the parameters loaded into the model when the user chooses
    this location


Creating the config.json File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A script exists that allows a command-line user to create a new directory for a
location file and generate its ``config.json`` file. If you've run
``python setup.py develop`` and``webgnome`` is installed, then you can run the
script by typing ``create_location_file`` with various options.

This command takes the following required options:

    --lat (the latitude of the location file)
    --lon (the longitude of the location file)
    --name (the name that the user should see in the location file map and menu)
    --filename (the filename the location file should receive)

Creating the location.json File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process for creating the ``location.json`` file for a new location is as follows:

    - Create a Python script that builds up the model and saves it to the
    location file's directory. For an example see
    ``webgnome/webgnome/scripts/create_boston_model.py``.

    - Run the script to create the file.

There are a couple of differences between this script and a typical py_gnome
script.

It should "bootstrap" the Pyramid application to get the configured location
file data directory, so that it can write the new ``location.json`` file there.
The code to do so starts with this line (after the import)::

    from pyramid.paster import bootstrap
    env = bootstrap('../development.ini')

See http://pyramid.readthedocs.org/en/latest/narr/commandline.html#writing-a-script
for more details on writing command-line scripts that work with Pyramid.

After creating the model object and adding movers, etc. to it, the script should
serialize the model object to JSON and write that data, with lines like the
following (taken from ``webgnome/webgnome/scripts/create_boston_model.py``::

    serialized_model = ModelSchema().bind().serialize(model.to_dict())
    model_json = json.dumps(serialized_model, default=util.json_encoder,
                            indent=4)
