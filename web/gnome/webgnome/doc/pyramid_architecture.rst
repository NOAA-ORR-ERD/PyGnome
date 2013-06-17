Pyramid Application Architecture
================================

A high-level overview of the architecture of the Python/Pyramid layer of
WebGNOME.

.. contents:: `Table of contents`
   :depth: 2


The Model
---------

A description of our use of the py_gnome.model.Model object, including the use
of a dict to contain running models.


Model Persistence
-----------------

A description of our proposed mechanism for reading and writing Model instances
via JSON files.


Validation Web Services
-----------------------

Validation web services were created especially for handling form errors in
Location File Wizards. Instead of saving form data to a service like
``/model/<id>/movers/wind`` immediately when the user submits, these forms only
perform validation on submit. Model data is saved later, when the user finishes
the wizard.

There is one validation service for each of the normal web service APIs. The
validation service checks the input for that API and does nothing if it is
correct.

On the server-side these services are essentially empty views that reuse the
same validation code from the normal web service they represent. E.g.::

    @resource(path='/model/{model_id}/validate/map', renderer='gnome_json',
              description='Validate Map JSON.')
    class MapValidator(BaseResourceValidator):
        schema = MapSchema

:class:`webgnome.views.services.schema_validation.BaseResourceValidator` uses a
metaclass that creates the necessary code to validate any incoming data to this
web service against `MapSchema`.

This code is in ``webgnome/webgnome/views/services/schema_validation.py``.

