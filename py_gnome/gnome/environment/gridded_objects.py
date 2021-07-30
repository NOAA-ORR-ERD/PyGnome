
from gnome.persist import (ObjTypeSchema, FilenameSchema, SchemaNode, Boolean)

# from gnome.persist.extend_colander import FilenameSchema
# from colander import (SchemaNode, Boolean)

from .environment_objects import GridCurrent
from .gridcur import GridcurCurrent


class FileGriddedCurrentSchema(ObjTypeSchema):
    filename = FilenameSchema(
        isdatafile=True, test_equal=False, update=False
    )
    extrapolation_is_allowed = SchemaNode(Boolean())


class FileGriddedCurrent(GridCurrent):
    """
    class that presents an interface for gridded currents loaded from
    files of various formats

    Done as a class to provide a Schema for the persistence system
    """
    _schema = FileGriddedCurrentSchema

    # def __new__(cls, filename, extrapolation_is_allowed=False, **kwargs):

    def __init__(self, filename, extrapolation_is_allowed=False):

        # determine what file format this is
        filename = str(filename)  # just in case it's a Path object
        if filename.endswith(".nc"):  # should be a netCDF file
            try:
                current = GridCurrent.from_netCDF(filename=filename,
                                                  extrapolation_is_allowed=extrapolation_is_allowed)

            except Exception as ex:
                raise ValueError(f"{filename} is not a valid netcdf file") from ex
            else:
                # and now the total kludge!
                # maybe need to set this? name=f"gridcur {data_type}",
                GridCurrent.__init__(self,
                                     units=current.units,
                                     time=current.time,
                                     variables=current.variables,
                                     varnames=current.varnames,
                                     extrapolation_is_allowed=extrapolation_is_allowed)
                self.filename = filename

        else:  # maybe it's a gridcur file -- that's the only other option
            try:
                GridcurCurrent.__init__(self, filename, extrapolation_is_allowed)
            except Exception as ex:
                raise ValueError(f"{filename} is not a valid gridcur file") from ex

    @classmethod
    def new_from_dict(cls, serial_dict):
        return cls(filename=serial_dict["filename"],
                   extrapolation_is_allowed=serial_dict["extrapolation_is_allowed"])


