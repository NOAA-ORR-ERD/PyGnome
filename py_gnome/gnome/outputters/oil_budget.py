"""
Outputter for dumping teh oil budget as a CSV file

(Or maybe other formats in the future)

"""

import csv

from .weathering import BaseMassBalanceOutputter
from .outputter import OutputterFilenameMixin


class OilBudgetOutput(BaseMassBalanceOutputter, OutputterFilenameMixin):
    """
    Outputter for the oil budget table
    """
    _valid_file_formats = ('csv')

    # Fixme: what is the 'non_weathering' field ??
    budget_categories = ['beached',
                         'dispersed'
                         'chem_dispersed',
                         'amount_released',
                         'off_maps',
                         'skimmed',
                         'burned',
                         'evaporated',
                         'floating']

    #                    'time_stamp',

    def __init__(self,
                 filename="gnome_oil_budget.csv",
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

    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              **kwargs):
        """
        start the csv file
        """
        outfile = open(self.filename, 'w')
        self.csv_writer = csv.writer(outfile)

    def write_output(self, step_num, islast_step=False):
        """
        Oil budget is only output for forecast spill container, not
        the uncertain spill container. This is because Weathering has its
        own uncertainty and mixing the two was giving weird results. The
        cloned models that are modeling weathering uncertainty do not include
        the uncertain spill container.
        """
        if not self._write_step:
            return None
        super(OilBudgetOutput, self).write_output(step_num, islast_step)

        mass_balance_data = self.gather_mass_balance_data(step_num)

        print "data:"
        print mass_balance_data.keys()
        print mass_balance_data['non_weathering']

    # write the header:







