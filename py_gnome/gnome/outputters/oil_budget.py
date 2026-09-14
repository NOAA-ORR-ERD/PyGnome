"""
Outputter for dumping the oil budget as a CSV file

(Or maybe other formats in the future)

"""

import csv
from pathlib import Path

from gnome.persist import Boolean, SchemaNode

from . import BaseOutputterSchema
from .outputter import OutputterFilenameMixin
from .weathering import BaseMassBalanceOutputter


class OilBudgetOutputSchema(BaseOutputterSchema):
    # these fields are saved in the base class so need to override
    #pass
    output_zero_step = SchemaNode(
        Boolean(), save=False, update=False
    )
    output_last_step = SchemaNode(
        Boolean(), save=False, update=False
    )
    output_single_step = SchemaNode(
        Boolean(), save=False, update=False
    )

class OilBudgetOutput(BaseMassBalanceOutputter, OutputterFilenameMixin):
    """
    Outputter for the oil budget table
    """
    _valid_file_formats = {'csv': '.csv'}

    # These go in the oil budget table
    # note: these need to be kept in sync!
    budget_categories = ['amount_released',
                         'evaporated',
                         'natural_dispersion',
                         'sedimentation',
                         'beached',
                         'floating',
                         'off_maps',
                         ]
    header_row = ['Model Time',
                  'Hours Since Model Start',
                  'Amount Released (kg)',
                  'Evaporated (kg)',
                  'Dispersed (kg)',
                  'Sedimentation (kg)',
                  'Beached (kg)',
                  'Floating (kg)',
                  'Off_maps (kg)',
                  ]


    _schema = OilBudgetOutputSchema

    def __init__(self,
                 filename="gnome_oil_budget.csv",
                 file_format='csv',
                 cache=None,
                 on=True,
                 output_timestep=None,
                 *args,
                 **kwargs):

        if file_format not in self._valid_file_formats:
            raise ValueError(f"Invalid format: {file_format}\n"
                             f"Formats allowed: {list(self._valid_file_formats.keys())}"
                             )
        self.file_format = file_format

        # add a csv if there's no extension
        filename = Path(filename).with_suffix(self._valid_file_formats[file_format])

        super().__init__(filename=filename,
                                              cache=None,
                                              on=True,
                                              output_timestep=output_timestep,
                                              # these always should be this way for this
                                              # outputter
                                              output_zero_step=True,
                                              output_last_step=True,
                                              output_single_step=False,
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
        super().prepare_for_model_run(model_start_time,
                                                           spills,
                                                           **kwargs)
        self._check_is_dir(self.filename)
        self._outfile = open(self.filename, 'w', newline='', encoding="utf-8")  # noqa: SIM115
        self.csv_writer = csv.writer(self._outfile)
        # write the header
        self.csv_writer.writerow(self.header_row)

    def write_output(self, step_num, islast_step=False):
        """
        Oil budget is only output for forecast spill container, not
        the uncertain spill container. This is because Weathering has its
        own uncertainty and mixing the two was giving weird results. The
        cloned models that are modeling weathering uncertainty do not include
        the uncertain spill container.
        """

        super().write_output(step_num, islast_step)

        # print "self._model_start_time:", self._model_start_time

        if self.on is False or not self._write_step:
            return

        mass_balance_data = self.gather_mass_balance_data(step_num)

        model_time = mass_balance_data['model_time']
        run_time = (model_time - self._model_start_time).total_seconds()
        hours = int(run_time // 3600)
        minutes = int((run_time - hours * 3600) // 60)
        row = [model_time.strftime("%Y-%m-%d %H:%M"),
               hours + minutes / 60]
        for category in self.budget_categories:
            try:
                row.append(f" {mass_balance_data[category]:.8g}")
            except KeyError:
                val = ''
                row.append(val)
        self.csv_writer.writerow(row)

    def post_model_run(self):
        """
        Called  after a model run is complete

        remove the csv file - hopefully resulting
        in the file being closed.

        """
        try:
            self._outfile.close()
        # I don't like a bare exception but it this fails,
        # it's either non-existent or already closed, so no harm done
        except:
            pass
        self.csv_writer = None








