JavaScript Application Architecture
===================================

A high-level overview of the architecture of the JavaScript/client-side layer of
WebGNOME.

.. contents:: `Table of contents`
   :depth: 2


Summary
--------

The WebGNOME JavaScript application employs a hybrid architecture that uses AJAX
requests for "view partials" for form markup rendering and submission, and
Backbone.js models and views for animation controls.

All forms are rendered on the server and retrieved and submitted by the app
using AJAX calls, and displayed using Backbone.js views. Form submissions
immediately change the value of settings on the user's server-side model.

"Running" the model sets up a chain of animations driven by the client by which
the client requests the next available image, displays it, and requests another
image, until the server reports that there are no more images remaining.

A client-side ``TimeStep`` model receives data about individual time steps during
the run. A ``Model`` object acts as a collection of ``TimeSteps`` and is bound to
animation controls, such as the slider, which update when the ``Model`` changes
(e.g. new ``TimeStep`` objects are added), and allow the user to start and stop a
"run" of the model.


Goals
-----

- Keep as little state about the model in the browser as possible.

- Restrict the JavaScript app from direct knowledge of URLs. URLs should be
  passed into the application from the server.

- Always request model data from the server before displaying it (i.e., in-app
  changes do not change a local model which persists; they change the server-side
  model and displays refresh from AJAX calls to the server).

- Render forms on the server ("view partials") and return via AJAX calls. Forms
  are not rendered by client- side templates in the browser, though we may use
  lightweight client-side validation in addition to server-side validation.

- Backbone.js models and views are used to render anything related to time
  steps, e.g. each frame in the animation.


The Navigation Tree
-------------------

- The tree renders root items ("movers", "spills", "settings") and child items,
  which are settings values, instantiated movers, etc., for the active model.

- Each item in the tree is linked to a form (add for root items, edit for child
  items) by a string known as the ``form_id`` that is both the HTML ID of the form
  element and the route name for the view that handles GET and POST requests for
  the form.

- The client may construct a delete URL to POST delete requests for child items
  using a URL convention. Note that this design sort of violates one of the
  above goals,that the JavaScript app should not have knowledge of URLs.

- The tree view listens for a successful submission event of any form, and if
  that happens, it makes an AJAX request for the new representation of the tree
  and redisplays itself.


Forms
-----

- All forms are rendered as HTML and included in the page on the first load.

- Invoking an "Add" event on a root item in the tree, such as "Movers" or "Spills"
  opens the "add" form for that item, if one exists.

- Invoking an "Edit" event on an item refreshes the form from the server using
  the ID of the item in question to populate the form fields.

- Submitting a form passes the serialized form values to a view via an AJAX call.
  If there were form errors, the rendered form is sent back from the server in a
  JSON response and the client will display it again.

- Displaying an "add" form for the same item subsequently requests a new copy of
  the form via an AJAX call, to clear out and reset any potential hidden fields on
  the form.
