'''
Weathering Outputter
'''

import os
from glob import glob

from geojson import dump
from colander import SchemaNode, String, drop

from .outputter import Outputter, BaseOutputterSchema


class BaseMassBalanceOutputter(Outputter):
    """
    Base class for outputters that need to return results of the mass balance:

    i.e. averaged properties of the LEs
    """
    units = {'default': 'kg',
             'avg_density': 'kg/m^3',
             'avg_viscosity': 'm^2/s'}

    def gather_mass_balance_data(self, step_num):
        # return a json-compatible dict of the mass_balance data
        # only applies to forecast spill_container (Not uncertain)
        sc = self.cache.load_timestep(step_num).items()[0]
        output_info = {'model_time': sc.current_time_stamp}
        output_info.update(sc.mass_balance)

        self.logger.debug(self._pid + 'step_num: {0}'.format(step_num))

        for name, val in output_info.items():
            msg = ('\t{0}: {1}'.format(name, val))
            self.logger.debug(msg)

        return output_info


class WeatheringOutputSchema(BaseOutputterSchema):
    output_dir = SchemaNode(
        String(), missing=drop, save=True, update=True
    )


class WeatheringOutput(BaseMassBalanceOutputter):
    '''
    class that outputs GNOME weathering results on a time step by time step basis

    The output is the aggregation of properties for all LEs (aka Mass Balance)
    for a particular time step.
    There are a number of different things we would like to graph:
    - Evaporation
    - Dissolution
    - Dissipation
    - Biodegradation
    - ???

    '''
    _schema = WeatheringOutputSchema

    # Fixme: -- this is a do-nothing __init__
    #        only here to document the interface
    # may need it in the future if we refactor out the output_dir handling
    def __init__(self,
                 output_dir=None,   # default is to not output to file
                 **kwargs):
        '''
        :param str output_dir=None: output directory for the json files.
                If not directory is provided, files will not be written.

        other arguments as defined in the Outputter class
        '''
        super(WeatheringOutput, self).__init__(output_dir=output_dir,
                                               **kwargs)

    def write_output(self, step_num, islast_step=False):
        '''
        Weathering data is only output for forecast spill container, not
        the uncertain spill container. This is because Weathering has its
        own uncertainty and mixing the two was giving weird results. The
        cloned models that are modeling weathering uncertainty do not include
        the uncertain spill container.
        '''
        super(WeatheringOutput, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        output_info = self.gather_mass_balance_data(step_num)
        # convert to string
        output_info['time_stamp'] = output_info.pop('model_time').isoformat()

        if self.output_dir:
            output_filename = self.output_to_file(output_info, step_num)
            output_info.update({'output_filename': output_filename})

        return output_info

    def output_to_file(self, json_content, step_num):
        file_format = 'mass_balance_{0:06d}.json'
        filename = os.path.join(self.output_dir,
                                file_format.format(step_num))

        with open(filename, 'w', encoding='utf-8') as outfile:
            dump(json_content, outfile, indent=4)

        return filename

    def clean_output_files(self):
        if self.output_dir:
            files = glob(os.path.join(self.output_dir,
                                      'mass_balance_*.json'))
            for f in files:
                os.remove(f)

    # just use the base class(s) one -- nothing to do here
    # cleaning out the files is done in prepare_for_model_run
    # def rewind(self):
    #     'remove previously written files'
    #     super(WeatheringOutput, self).rewind()

    #     self.clean_output_files()

    def __getstate__(self):
        '''
            This is to support pickle.dumps() inside the uncertainty model
            subprocesses.
            We need to be able to pickle our weathering outputters so that
            our uncertainty subprocesses can send them back to the parent
            process through a message queue.
            And the cache attribute (specifically, the ElementCache.lock
            attribute) can not be pickled, and instead produces a
            RuntimeError.

            (Note: The __setstate__() probably doesn't need to recreate the
                   ElementCache since it will be created inside the
                   Model.setup_model_run() function.)
        '''
        odict = self.__dict__.copy()  # copy the dict since we change it

        del odict['cache']  # remove cache entry

        return odict
