JavaScript Application Architecture
===================================

A high-level overview of the architecture of the JavaScript/client-side layer of
WebGNOME.

.. contents:: `Table of contents`
   :depth: 2


Summary
-------

The WebGNOME JavaScript application is loosely structured based on the MVC
pattern using Backbone.js. Functionality is spread between View and Model
objects, with Views often receiving user input in a style more like a
traditional Controller -- as is the fashion when using Backbone.

Views listen to Models and refresh themselves when models change. A model may
represent GUI state, such as the type of action the user is currently engaged
in (playing an animation, drawing a spill onto the map) or they may exist
as client-side representation of server-side objects, like Wind Movers,
Surface Release Spills and the Gnome Model itselt -- or, at least, the settings
values that it makes sense for a user to change from a web GUI.


High-level Goals
----------------

- Always request model data from the server before displaying it. In-app changes
  change an intermdiary client-side model which is validated server-side and
  only persisted if validation succeeds.

- Keep browser memory use low for long-running sessions of the web application
  by swapping out the data for form-related views from models when forms are
  displayed, rather than creating multiple View objects, one for each model.

- Validate data in the client and on the server. See VALIDATION for more details.


RequireJS Modules
-----------------

Code for the JavaScript application is separated into modules using the
RequireJS library. This means that each module declares its dependencies at
the top of the file, and those JavaScript modules (or files, in the case of non-
RequireJS-enabled code) will load before the code of the module is executed.


Backbone.js Models and Views
----------------------------

The application uses Backbone.js client-side Model and View objects for
animation controls and to represent input forms.


Two-Way Data Binding with Rivets.js
-----------------------------------

Form inputs are bound to a specific model through the Rivets.js library, in a
manner similar to that found in Angular.js. This means that as users fill out a
form, the underlying model the form uses is updated.

If the user cancels a form without saving, the model is retrieved from the
server and refreshed.


Data Validation
---------------

Data users input may be validated one of three different ways -- all of which
happen during form submission. The workflow for a form submission is as follows:

- The user edits a form, updating the model as they do so
- The user clicks a "Save" button
- The View responsible for the form executes its ``submit`` event handler
- If a JSON Schema is defined for the model, user input is checked agains the
  schema by the JSV JavaScript library. Any errors are displayed on the form
  and the submissions is canceled.
- Data for the form is sent to the View's validation web service if one has
  been defined. Any errors returned by the validation service are displayed
  on the form for the user to correct.
- If no validation service is defined, the model is saved to its usual web
  service API, e.g., `/model/<model_id>/movers/wind/<wind_id>`. Again, if any
  errors are returned, they are displayed on the form next to the appropriate
  input fields.

A validation web service is normally only used if the save operation the user is
attempting to do will be deferred until a later time, as is the case when the
user is filling out a multi-step form. We wait to send each "save" operation
until the user finishes the multi-step form, to make it easier for us to back
out of changes to multiple models during the course of the form.


Running the Model
-----------------

"Running" the :class:`gnome.model.Model` sets up a chain of animations driven
by the client by which the client requests the next available image, displays
it, and requests another image, until the server reports that there are no more
images remaining.

A client-side ``TimeStep`` Backbone.js model receives data about individual
time steps during the run. A ``Model`` object acts as a collection of
``TimeSteps`` and is bound to animation controls, such as the slider, which
update when the Backbone.js changes (e.g. new ``TimeStep`` objects are added),
and allow the user to start and stop a "run" of the :class:`gnome.model.Model`.


The Navigation Tree
-------------------

- The tree renders root items ("movers", "spills", "settings") and child items,
  which are settings values, instantiated movers, etc., for the active model.

- Each item in the tree is linked to a form by a string known as the ``form_id``
  that is the HTML ID of the form that should open the item. A View with that ID
  will open if the user double-clicks on the item, e.g. the Edit Wind Mover form.
  Each item also has an ``object_id`` which refers to a Model that will then
  be loaded as the dataset for the form.

- The tree view listens for a successful submission event of any form, and if
  that happens, it makes an AJAX request for the new representation of the tree
  and redisplays itself.

