from paste.deploy.loadwsgi import NicerConfigParser
from unipath import Path

class ConfigParser(NicerConfigParser):
    def read(self, filename):
        NicerConfigParser.read(self, filename)
        ini_file = Path(filename).absolute()
        self._defaults.setdefault("__file__", ini_file)
        self._defaults.setdefault("here", ini_file.parent)


def get_config_section(filename, section):
    """Extract a section from an INI file as a dict.

    Supports interpolation and "set " the way Pylons does.
    """
    cp = ConfigParser(filename)
    cp.read(filename)
    ret = cp.defaults().copy()
    for option in cp.options(section):
        if option.startswith("set "):  # Override a global option.
            option = option[4:]
        ret[option] = cp.get(section, option)
    return ret

