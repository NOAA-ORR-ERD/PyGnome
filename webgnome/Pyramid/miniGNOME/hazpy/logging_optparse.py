"""An OptionParser that can also configure logging.

The API is experimental and subject to change.  Please list all applications
using this module here:
- 2009-04-30: ResponseLINK scripts.
"""

import logging
import logging.config
import optparse

TRACE = 5   # Trace log level.

#### Exceptions
class LogLevelError(KeyError):
    def __init__(self, level):
        self.level = level
        msg = "logging level '%s' is undefined" % level
        KeyError.__init__(self, msg)

#### Utility functions
def resolve_log_level(level):
    add_trace_level()
    if isinstance(level, int):
        return level
    level = level.upper()
    if level.isdigit():
        return int(level)
    elif level in logging._levelNames:
        return logging._levelNames[level]
    raise LogLevelError(level)
    
    
def add_trace_level():
    if "TRACE" not in logging._levelNames:
        logging.addLevelName(TRACE, "TRACE")
    if not hasattr(logging.Logger, "trace"):
        logging.Logger.trace = trace

def trace(self, *args, **kw):
    self.log(TRACE, *args, **kw)

#### LoggingOptionParser class
class OptionParser(optparse.OptionParser):
    def __init__(self, arg_names=None, varargs_name=None, **kw):
        add_trace_level()
        self.arg_names = arg_names or []
        self.varargs_name = varargs_name
        self.nargs_error_messages = {}
        optparse.OptionParser.__init__(self, **kw)
        self.add_option("--log", action="append",
            metavar="LEVEL or LOGGER:LEVEL",
            help="set the default log level or specified logger's level")

    def parse(self, *args, **kw):
        opts, args = self.parse_args(*args, **kw)
        def args_len_error():
            default = "wrong number of command-line arguments"
            msg = self.nargs_error_messages.get(len(args), default)
            self.error(msg)
        for i in range(len(self.arg_names)):
            try:
                setattr(opts, self.arg_names[i], args[i])
            except IndexError:
                args_len_error()
        if self.varargs_name:
            setattr(opts, self.varargs_name, args[len(self.arg_names):])
        elif len(args) > len(self.arg_names):
            args_len_error()
        return opts

    def on_nargs_error(self, nargs, message):
        self.nargs_error_messages[nargs] = message

    def init_logging_from_ini(self, filename):
        logging.config.fileConfig(filename)

    def add_logging_options(self, **kw):
        """More options may be added in the future."""
        self._add_special_logging_options(**kw)

    def init_logging(self, opts, log_date=False, **basic_config_kw):
        basic_config_kw.setdefault("level", logging.INFO)
        basic_config_kw.setdefault("format", 
            "%(asctime)s %(levelname)s [%(name)s] %(message)s")
        if log_date:
            basic_config_kw.setdefault("datefmt", "%Y-%m-%d %H:%M:%S")
        else:
            basic_config_kw.setdefault("datefmt", "%H:%M:%S")
        logging.basicConfig(**basic_config_kw)
        self._init_logging_from_special_options(opts)
        if opts.log:
            self._init_logging_from_specs(opts.log)

    #### Private methods
    def _add_special_logging_options(self, **kw):
        def add(name, help):
            self.add_option(name, action="store_true", help=help)
        if kw.pop("debug", False):
            add("--debug", "enable debug logging")
        if kw.pop("quiet", False):
            add("--quiet", "disable status logging")
        if kw.pop("trace", False):
            add("--trace", "enable trace logging")
        if kw.pop("sql", False):
            add("--sql", "log SQLAlchemy statements")
        if kw:
            unknown = " ".join(sorted(kw.keys()))
            raise TypeError("unknown keyword args: %s" % unknown)


    def _init_logging_from_special_options(self, opts):
        if getattr(opts, "quiet", False):
            logging.getLogger().setLevel(logging.WARN)
        if getattr(opts, "debug", False):
            logging.getLogger().setLevel(logging.DEBUG)
        if getattr(opts, "trace", False):
            logging.getLogger().setLevel(0)
        if getattr(opts, "sql", False):
            logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    def _init_logging_from_specs(self, specs):
        try:
            for spec in specs:
                parts = spec.split(":", 1)
                if len(parts) == 1:
                    logger = "__main__"
                    level = parts[0].upper()
                else:
                    logger = parts[0]
                    level = parts[1].upper()
                level = resolve_log_level(level)
                logging.getLogger(logger).setLevel(level)
                if logger == "__main__":
                    # Set root logger to same level.
                    logging.getLogger().setLevel(level)
        except LogLevelError, e:
            parser.error("log level '%s' not defined" % e.level)
