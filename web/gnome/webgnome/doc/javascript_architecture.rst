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
represent GUI state, such as the type of action the user is currently engaged in
(playing an animation, drawing a spill onto the map) or they may exist as
client-side representation of server-side objects, like Wind Movers, Surface
Release Spills and the Gnome Model itself -- at least, the settings values
that it makes sense for a user to change from a web GUI.


High-level Goals
----------------

- Validate before saving anything. In-app changes change an intermdiary
  client-side model which is validated server-side and only persisted if
  validation succeeds. See VALIDATION for more details.

- One view, many models. Keep browser memory use low for long-running sessions
  of the web application by swapping out the data for form-related views from
  models when forms are displayed, rather than creating multiple View objects,
  one for each model.


Application Entry-Point
-----------------------

The server-side view that loads the JavaScript application is:

   webgnome/webgnome/views/model.py

The template it renders, which performs the client-side loading using RequireJS,
is:

  webgnome/webgnome/templates/model.mak

That template renders all of the HTML for the dialog forms in a hidden div and
loads RequireJS, which handles loading all of the individual JavaScript
"modules."

It also configures Rivets, discussed in `Two-Way Data Binding with Rivets.js`.


RequireJS Modules
-----------------

Code for the JavaScript application is separated into modules using the
RequireJS library. Each module declares its dependencies at the top of the file.

A RequireJS module is basically an anonymous function. Dependencies for each
module are listed at the top of the module as parameters to the anonymous
function. RequireJS resolves the dependencies before executing the function body
and passes the requirements into the function in the position you defined.

It looks like this::

    define([
        'jquery',
        'lib/underscore',
        'models',
        'util',
        'views/forms/base',
    ], function($, _, models, util, deferreds, multi_step, base) {

        // The body of your module goes here
    });

The location of dependencies is relative to the root you configure for
RequireJS. Configuration is done in webgnome/webgnome/static/js/config.js.

In the example, we're loading some library code from the lib/ directory and some
of our own code from the root directory and the views/forms directory.


Backbone.js and Underscore.js
-----------------------------

These are the most-used libraries after jQuery. The application uses Backbone.js
client-side Model and View objects for animation controls and to represent input
forms.

Underscore is a swiss-army utility library that Backbone relies on. It provides
the "_" symbol. This is used in the application to iterate over collections and
to render client-side templates. It is also used by Backbone to provide various
APIs on its Collection prototype.

These libraries both have decent API docs for any method the application uses.


Two-Way Data Binding with Rivets.js
-----------------------------------

http://rivetsjs.com/

Rivets provides two-way data binding between forms and Backbone models. It's
used in most of the form dialogs to keep the form and the form's model in sync.

Configured in model.mak
~~~~~~~~~~~~~~~~~~~~~~~

The first is that it is configured in model.mak to use a Backbone-specific
adapter -- a JavaScript object that tells Rivets how to save values found in the
HTML forms it is bound to, to the model objects it is bound to. Conceptually
this is a map of the Rivets API to the Backbone API -- "read(obj, field_name)"
becomes "obj.get(field_name)", "subscribe(obj, event_name, callback)" becomes
"obj.on(event_name, callback)" and so forth.

Bound by FormViews
~~~~~~~~~~~~~~~~~~

Each form in the application has a FormView object bound to it. These are
Backbone views. When a form is "shown" (the "show" method is called), usually
when someone double-clicks on an item in the tree, the FormView object asks
Rivets to bind any form inputs in the view's HTML element to a particular model.

This process happens when a form is shown and not just once during
initialization because while there are multiple model objects, the app uses only
one FormView for each type of model (me trying to optimize memory use).

So when a model is hidden and shown again, its model may have changed. We bind
Rivets when showing the form and unbind when hiding it.


Data bindings and "getDataBindings" method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the moment forms are shown and Rivets binds them to the FormView's model, it
looks for all inputs with certain data- values.

How this works is you'll have something like this in a server-side template for
a form view::

    refloat_halflife = h.text('refloat_halflife',
                              data_value='map.refloat_halflife',
                              class_='input-small')

This calls the 'h' object, a Python helper, to create an HTML text input with
the name 'refloat_halflife'. The HTML input gets a data- attribute consumed by
Rivets: ``data-value='map.refloat_halflife'``.

When Rivets is bound to the HTML form containing this input,
``data-value='map.refloat_halflife'`` tells Rivets to set the 'refloat_halflife'
field on the object 'map' to the input's value whenever the input changes, and
to set the input whenever the model's value changes (2-way binding).

The Rivets docs list the various data- attributes you can use.

Behind the scenes, whenever anyone changes this field, Rivets executes a call
like this::

    map.set('refloat_halflife', newValue);

So, long story ... long, in order for Rivets to resolve "map.refloat_halflife"
to an object, each FormView that uses Rivets has to provide a data bindings
dict-like object, e.g. {'map': actualMapObject} when it calls ``rivets.bind()``.
There are examples of this in the FormViews that use Rivets.

The end result is that when the user clicks "Save" we persist the current values
in the model without much haggling because they should reflect the user's
choices. However, there is one problem...


jQuery Deferreds
----------------

jQuery now returns Deferred objects for any async operations. That means when
you perform a $.get or $.ajax operation, you get back a deferred. This is a new
API that they intend to help cut back on the number of deeply-nested callbacks
in asyn code.

Backbone uses jQuery as its transport mechanism for any model persistence, so
when you call ``model.save()`` the return value is a Deferred.

This is helpful in a lot of cases where you need to wait to do something until
an AJAX call returns, but you still want the call to happen async. I've
incorporated the use of deferreds in a few places where you would normally see a
success callback applied. So you will see things like this::

    object.save().then(function() {
        _this.doSomethingElse();
    });

Anytime you see ".then()" or ".resolve()" the object is a Deferred. And I've
tried to add a note to the docstring of functions that return Deferreds.


Location File Wizards
---------------------

I found Deferreds helpful in creating the FormView code that handles Location
File Wizards (webgnome/webgnome/static/js/views/forms/location_file_wizard.js).
These are basically multi-step form wizards where the FormView handles
interaction and the HTML markup is defined elsewhere, each location file's
wizard.mak file.

The way they work is to hide various "steps" of the form in the same HTML div.
Whatever step you are on is the one that is visible. Each step has a button to
go back or to save and continue to the next step. When the user goes to the next
step, they are usually submitting a form, like the WindMover form. However, we
don't necessarily want to submit the forms as they work through the wizard, in
case they decide to abandon it half-way through, because they'd be back at a
non-working model that may have replaced one that was working.

Instead, we queue up all of the form submits as Deferreds and then execut them
all in the same order when the user finishes the form.


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
  service API, e.g., ``/model/<model_id>/movers/wind/<wind_id>``. Again, if any
  errors are returned, they are displayed on the form next to the appropriate
  input fields.

A validation web service is normally only used if the save operation the user is
attempting to do will be deferred until a later time, as is the case when the
user is filling out a multi-step form. We wait to send each "save" operation
until the user finishes the multi-step form, to make it easier for us to back
out of changes to multiple models during the course of the form.

For more details on the server-side component, see `Validation Web Services`.


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

