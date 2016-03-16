*****************************************
Notes on serialization / deserialization:
*****************************************

Overview
########

The process involves multiple steps:

Serializing:
  - From Object to a json-compatible dict
  - From json-compatible dict to actual json

Deserializing:
  - From actual json to a json-compatible dict
  - From a json-compatible dict to a Object-compatible dict.
  - From an object compatible dict to an Object.

This is a "cycle", so could be in a different order depending on where you start!

Also -- "deserializing" can be createing a new object from json, or updating teh object from json.

Methods
#######

``Object.serialize()`` returns a "json-compatible" dict.


Making a new object serializable
################################


