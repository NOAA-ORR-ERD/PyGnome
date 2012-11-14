JavaScript Application Architecture
===================================

A high-level overview of the architecture of the JavaScript/client-side layer of
WebGNOME.

.. contents:: `Table of contents`
   :depth: 2


Summary
-------

The WebGNOME JavaScript application uses two approaches for managing UI
elements. The first is AJAX requests for "view partials," which are server-side
API endpoints (known as "views" in Pyramid) that render HTML that the JavaScript
application then inserts into the DOM. The second is the Backbone.js pattern of
client-side ``View`` objects that listen to client-side ``Model`` objects and
refresh themselves when these models change.


High-level Goals
----------------

- Keep as little state about the model in the browser as possible.

- Restrict the use of hard-coded API URLs in JavaScript. URLs should be passed
  into the application from the server.

- Always request model data from the server before displaying it. In-app changes
  do not change an intermediary client-side model which then persists to the
  server, as is usual in Backbone.js applications; they change the server-side
  model directly via form submissions and displays refresh from AJAX calls to the
  server.

- Render forms on the server ("view partials") and return via AJAX calls. Forms
  are not rendered by client-side templates in the browser, though we may use
  lightweight client-side validation in addition to server-side validation.

- Backbone.js models and views are used to render read-only data.


Server-side View Partials
-------------------------

The JavaScript application uses view partials for all form-handling code. Forms
are submitted via AJAX requests to the server, rendered in the server, and sent
back to the JavaScript application if there are errors. A Backbone.js ``View``
called the ``ModalFormView`` is responsible for displaying the form HTML
returned by the server in a modal display. Form submissions immediately change
the value of settings on the user's server-side :class:`gnome.model.Model`.


Backbone.js Models and Views
----------------------------

The application uses Backbone.js client-side models and views for animation
controls. The convention used in the application is that only form submissions
may change the underlying :class:`gnome.model.Model` object, so Backbone.js
``Model`` objects are used in a read-only way: i.e., these "models" are not
saved back to the server, but typically refresh when the
:class:`gnome.model.Model` changes after a form submission.


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
  that is the HTML ID of the form. A form class is responsible for providing this
  string both when a form is rendered and on demand via a class method, which is
  used during the construction of the navigation tree.

- Each item in the tree is given a ``delete_url`` that the client may use to
  delete the resource the item refers to.

- The tree view listens for a successful submission event of any form, and if
  that happens, it makes an AJAX request for the new representation of the tree
  and redisplays itself.


Forms
-----

- All forms are rendered as HTML and included in the page on the first load.

- Invoking an "Add" event on a root item in the tree, such as "Movers" or "Spills"
  displays the add form for that item.

- Invoking an "Edit" event on an item displays the edit form for the item, which
  exists on the page in a hidden div.

- Submitting a form passes the serialized form values to a Pyramid view in an
  AJAX call. If there were form errors, the rendered form is sent back from the
  server in a JSON response and the client will display it again.

- After every form submission, the hidden <div> of form HTML is refreshed from a
  partial view, to make sure all forms reflect the current state of the model.
