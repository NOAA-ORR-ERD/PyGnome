'''
Weathering Outputter
'''
import copy
import os
from glob import glob

from geojson import dump
from colander import SchemaNode, String, drop

from gnome.utilities.serializable import Serializable, Field

from .outputter import Outputter, BaseSchema

from gnome.basic_types import oil_status


class WeatheringOutputSchema(BaseSchema):
    output_dir = SchemaNode(String(), missing=drop)


class WeatheringOutput(Outputter, Serializable):
    '''
    class that outputs GNOME weathering results.
    The output is the aggregation of properties for all LEs (aka Mass Balance)
    for a particular time step.
    There are a number of different things we would like to graph:
    - Evaporation
    - Dissolution
    - Dissipation
    - Biodegradation
    - ???

    However at this time we will simply try to implement an outputter for the
    halflife Weatherer.
    Following is the output format.

        {
        "type": "WeatheringGraphs",
        "half_life": {"properties": {"mass_components": <Component values>,
                                     "mass": <total Mass value>,
                                     }
                      },
            ...
        }

    '''
    _state = copy.deepcopy(Outputter._state)

    # need a schema and also need to override save so output_dir
    # is saved correctly - maybe point it to saveloc
    _state += [Field('output_dir', update=True, save=True)]
    _schema = WeatheringOutputSchema

    def __init__(self,
                 output_dir=None,   # default is to not output to file
                 **kwargs):
        '''
        :param str output_dir='./': output directory for geojson files

        use super to pass optional \*\*kwargs to base class __init__ method
        '''
        self.output_dir = output_dir
        self.units = {'default': 'kg',
                      'avg_density': 'kg/m^3',
                      'avg_viscosity': 'm^2/s'}
        super(WeatheringOutput, self).__init__(**kwargs)

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

        # return a dict - json of the mass_balance data
        # weathering outputter should only apply to forecast spill_container
        sc = self.cache.load_timestep(step_num).items()[0]
        dict_ = {}
        dict_.update(sc.mass_balance)

        output_info = {'step_num': step_num,
                       'time_stamp': sc.current_time_stamp.isoformat()}
        output_info.update(dict_)
        self.logger.debug(self._pid + 'step_num: {0}'.format(step_num))
        for name, val in dict_.iteritems():
            msg = ('\t{0}: {1}'.format(name, val))
            self.logger.debug(msg)

        if self.output_dir:
            output_filename = self.output_to_file(output_info, step_num)
            output_info.update({'output_filename': output_filename})

        return output_info

    def output_to_file(self, json_content, step_num):
        file_format = 'mass_balance_{0:06d}.json'
        filename = os.path.join(self.output_dir,
                                file_format.format(step_num))

        with open(filename, 'w') as outfile:
            dump(json_content, outfile, indent=True)

        return filename

    def clean_output_files(self):
        if self.output_dir:
            files = glob(os.path.join(self.output_dir,
                                      'mass_balance_*.json'))
            for f in files:
                os.remove(f)

    def rewind(self):
        'remove previously written files'
        super(WeatheringOutput, self).rewind()
        self.clean_output_files()

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
        del odict['cache']               # remove cache entry

        return odict
