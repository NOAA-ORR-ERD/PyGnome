"""
Outputter for dumping teh oil budget as a CSV file

(Or maybe other formats in the future)

"""

from .weathering import BaseMassBalanceOutputter
from .outputter import OutputterFilenameMixin


class OilBudgetOutput(BaseMassBalanceOutputter, OutputterFilenameMixin):
    """
    Outputter for the oil budget table
    """
    _valid_file_formats = ('csv')

    def __init__(self,
                 filename,
                 file_format='csv',
                 cache=None,
                 on=True,
                 output_timestep=None,
                 *args,
                 **kwargs):

        if file_format not in self._valid_file_formats:
            raise ValueError("Invalid format: {}\n"
                             "Formats allowed: {}".format(file_format,
                                                          self._valid_file_formats))
        self.file_format = file_format

        super(OilBudgetOutput, self).__init__(filename=filename,
                                              cache=None,
                                              on=True,
                                              output_timestep=output_timestep,
                                              # these always should be this way for this
                                              # outputter
                                              output_zero_step=True,
                                              output_last_step=True,
                                              output_start_time=None,
                                              surface_conc=None,
                                              *args,
                                              **kwargs)






