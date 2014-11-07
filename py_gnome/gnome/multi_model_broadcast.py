from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import sys
import traceback

import multiprocessing
mp = multiprocessing

from gnome.environment import Wind
from gnome.outputters import WeatheringOutput


class ModelConsumer(mp.Process):
    '''
        This is a consumer process that makes the model available
        upon process creation so that registered commands can act upon
        the model.
        Program flow:
        - Read a command from the task queue
        - if there is a None command, we exit the process.
        - Parse the data received in the format:
            ('registeredcommand', {arg1: val1,
                                   arg2: val2,
                                   ...
                                   },
             )
        - Attempt to perform the registered command.  Registered commands
          are defined as private methods of this class.
        - Returns the results in a results queue

    '''
    def __init__(self, task_queue, result_queue, model):
        mp.Process.__init__(self)

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model = model

    def run(self):
        proc_name = self.name
        while True:
            data = self.task_queue.get()
            if data is None:
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                break

            try:
                cmd, kwargs = data
                result = getattr(self, '_' + cmd)(**kwargs)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                fmt = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback)
                result = fmt

            self.result_queue.put(result)
        return

    def _rewind(self):
        return self.model.rewind()

    def _step(self):
        return self.model.step()

    def _num_time_steps(self):
        return self.model.num_time_steps

    def _full_run(self, rewind=True, logger=False):
        return self.model.full_run(rewind=rewind, logger=logger)

    def _get_wind_timeseries(self):
        '''
            just some model diag
        '''
        res = []
        wind_objs = [e for e in self.model.environment
                     if isinstance(e, Wind)]

        for obj in wind_objs:
            ts = obj.get_timeseries()
            for tse in ts:
                res.append(tse['value'])

        return res

    def _get_spills(self):
        return self.model.spills

    def _get_spill_amounts(self):
        return [s.amount for s in self.model.spills]

    def _set_wind_speed_uncertainty(self, up_or_down):
        winds = [e for e in self.model.environment
                 if isinstance(e, Wind)]
        res = [w.set_speed_uncertainty(up_or_down) for w in winds]

        return all(res)

    def _set_spill_container_uncertainty(self, uncertain):
        self.model.spills.uncertain = uncertain
        return self.model.spills.uncertain

    def _set_spill_amount_uncertainty(self, up_or_down):
        res = [s.set_amount_uncertainty(up_or_down) for s in self.model.spills]

        return all(res)

    def _get_cache_dir(self):
        return self.model._cache._cache_dir

    def _set_cache_dir(self):
        return self.model._cache.create_new_dir()

    def _get_outputters(self):
        return self.model.outputters

    def _set_weathering_output_only(self):
        del_list = [o for o in self.model.outputters
                    if not isinstance(o, WeatheringOutput)]
        for dl in del_list:
            del self.model.outputters[dl.id]


class ModelBroadcaster(object):
    '''
        Here is where we spawn an array of model consumer processes
        based on the variations in the model configurations we would like.

        More specifically, the model variations we are interested in are
        uncertainty variations.
    '''
    def __init__(self, model,
                 wind_speed_uncertainties,
                 spill_amount_uncertainties):
        self.tasks = []
        self.results = []
        self.lookup = {}

        idx = 0
        for wsu in wind_speed_uncertainties:
            for sau in spill_amount_uncertainties:
                self.tasks.append(mp.Queue())
                self.results.append(mp.Queue())

                model_consumer = ModelConsumer(self.tasks[idx],
                                               self.results[idx],
                                               model)
                model_consumer.start()

                self._set_uncertainty(idx, wsu, sau)
                self._set_new_cache_dir(idx)
                self._set_weathering_output_only(idx)

                self.lookup[(wsu, sau)] = idx
                idx += 1

    def __del__(self):
        self.stop()

    def cmd(self, command, args, key=None):
        if key is None:
            [t.put((command, args)) for t in self.tasks]
            return [r.get() for r in self.results]
        else:
            idx = self.lookup[key]
            self.tasks[idx].put((command, args))
            return self.results[idx].get()

    def stop(self):
        [t.put(None) for t in self.tasks]
        self.tasks = []
        self.results = []
        self.lookup = {}

    def _set_uncertainty(self, index,
                         wind_speed_uncertainty,
                         spill_amount_uncertainty):
        # py_gnome spill container uncertainty is not used here
        # so we turn it off always
        self.tasks[index].put(('set_spill_container_uncertainty',
                               dict(uncertain=False)))
        self.results[index].get()

        self.tasks[index].put(('set_wind_speed_uncertainty',
                               dict(up_or_down=wind_speed_uncertainty)))
        self.results[index].get()

        self.tasks[index].put(('set_spill_amount_uncertainty',
                               dict(up_or_down=spill_amount_uncertainty)))
        self.results[index].get()

    def _set_new_cache_dir(self, index):
        self.tasks[index].put(('set_cache_dir', {}))
        self.results[index].get()

    def _set_weathering_output_only(self, index):
        self.tasks[index].put(('set_weathering_output_only', {}))
        self.results[index].get()
