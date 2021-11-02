The logging system
==================

``pygnome`` comes with the python logging system pre-configured for use.

It is used in the code to capture model state and warnings. When used as the back-end behind WebGNOME, the logging messages are passed to the client for display to the user, and parsed for capturing warnings etc.

But you may also want to use the logging system in scripting mode, to capture what's going on with the model.

Configuring the logger
----------------------

The logger can be initialized and configured in the usual way with the python ``logging`` module. However, pygnome comes with a couple utilities to make it easy to do standard configuration:

``gnome.initialize_console_log(level='debug')``
................................................

Initializes the logger to simply log everything to the console (stdout). Likely what you want for scripting use.

you can set the logging level to the level you want your log to show. options are, in order of importance: "debug", "info", "warning", "error", "critical"

You will only get the logging messages at or above the level you set.

(see ``script_columbia_river`` for an example)


``gnome.initialize_log(config, logfile=None)``
..............................................

Helper function to initialize a log - this should be called by the application using PyGnome. This is used to set up the log for the WEbGNOME API, for instance. ``config`` can be a file containing json or it can be a Python dict -- in the dict config format used by ``logging.dictConfig``:

https://docs.python.org/2/library/logging.config.html#logging-config-dictschema

logfile is the optional file name to log to.

(this requires a bit of knowledge of the logging system)

Using the logger
----------------

If you want to use the logger in your scripts (or are writing your own mover, etc.), you can do:

``import logger``

and then use the regular logging functions, passing in the message you want to log::

    logger.debug("this is info you'd only want for debugging")
    logger.info("this is some arbitrary information")
    logger.warning("this is just a friendly warning")
    logger.error("oh oh! an error occurred")
    logger.critical("whoops! a critical error -- you really don't want to miss this!" )

These are in order of importance -- debug is not important, critical is critically important.

Logging levels in WebGNOME
--------------------------

In WebGNOME, The logger is used to pass information to the Web client through the WebAPI. In this case, the logging levels are used to tell the client how to process the log message:

``logger.debug``:

``logger.info``:

``logger.warning``:

``logger.error``:

``logger.critical``:





