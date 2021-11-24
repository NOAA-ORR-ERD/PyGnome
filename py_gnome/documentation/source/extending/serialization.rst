########################################
Notes on serialization / deserialization
########################################

.. _serialization_overview:

Overview
========

PYGNOME used a JSON serialization system for two things: saving model configuration in "save files", and communication with the Web Client. The two JSON forms are very similar and set up with the same system.


There are three key processes:

Serializing:
  - Saving the configuration of an Object to a json-compatible dictionary
  - Saving the json-compatible dict to actual json -- in files or strings to be passed via the API. For the most part, any new code only needs to concern itself with the JSON-compatible dict.

Deserializing:
  - From actual json (in string or file) to a json-compatible dict
  - From a json-compatible dict to a Object-compatible dict.
  - From an object compatible dict to a configured Object.

Updating:
  - Changing an already existing object with a JSON compatible dict
  - Very similar to deserializing, except that the object already exists. The JSON may only contain the attributes that need to be updated.

Methods
=======

Basic serialization methods are provided by the ``GnomeObject`` base class.

``Object.serialize()`` returns a "json-compatible" dict.


Making a new object serializable
================================

In order to make a new GnomeObject compatible with the serialization system, a ``Schema`` must be provided. The schema specifies which attributes need to be saved and loaded, and the limitations of how they can be set e.g. some are read-only, some can be updated, etc.

The serialization system uses an extension of the `colander <https://docs.pylonsproject.org/projects/colander/en/latest/>`_ package. Schema nodes are defined that specify how the object is to be (de)serialized.

[Examples here]
