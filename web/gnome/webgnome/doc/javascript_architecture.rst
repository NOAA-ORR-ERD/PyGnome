JavaScript Application Architecture
===================================

A high-level overview of the architecture of the JavaScript/client-side layer of
WebGNOME.

.. contents:: `Table of contents`
   :depth: 2


Summary
--------

The WebGNOME JavaScript application uses a hybrid architecture that uses AJAX
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

- Keep as little state about the model in the browser as possible

- Always request model data from the server before displaying it (i.e., in-app
  changes do not change a local model which persists; they change the server-side
  model and displays refresh from AJAX calls to the server)

- Render forms on the server ("view partials") and return via AJAX calls. Forms
  are not rendered by client- side templates in the browser, though we may use
  lightweight client-side validation in addition to server-side validation.

- Backbone.js models and views are used to render anything related to time
  steps, e.g. each frame in the animation.


The Navigation Tree
-------------------

- The tree renders root items ("movers", "spills", "settings") and child items,
  which are actual settings values, instantiated movers, etc., for the active
  model

- Each item in the tree is linked to a form (add for root items, edit for child
  items)

- The client may construct a delete URL to POST delete requests for child items
  using a URL convention (what is the best convention here?)

- The tree view listens for a successful submission event of any form, and if
  that happens, it makes an AJAX request for the new representation of the tree
  and redisplays itself


Forms
-----

- All forms are rendered as HTML and included in the page on the first load

- Adding an item shows the initial form

- Editing an item refreshes the form from the server using the ID of the item in
  question to populate the form

- Displaying an "add" form for the same item subsequently requests a new copy of
  the form via an AJAX call, to clear out and reset any potential hidden fields on
  the form
