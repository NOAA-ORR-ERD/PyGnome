from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import sys
import traceback
import multiprocessing
mp = multiprocessing

from gnome.environment import Wind


class ModelConsumer(mp.Process):
    '''
        This is a consumer process that:
        - reads a command from a queue
        - acts on the data received in the format:
            ('registeredcommand', {arg1: val1,
                                   arg2: val2,
                                   ...
                                   },
             )
        - returns the results in a results queue

        The model is passed into the child process,
        and all registered commands presumably act upon the model
    '''
    def __init__(self, task_queue, result_queue, model):
        mp.Process.__init__(self)

        self.commands = {'full_run': self.full_run,
                         'step': self.step,
                         'get_wind_timeseries': self.get_wind_timeseries
                         }

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
                result = self.commands[data[0]](**data[1])
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                fmt = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback)
                result = fmt

            self.result_queue.put(result)
        return

    def get_wind_timeseries(self):
        '''
            just some model diag
        '''
        res = []
        time_objs = [e for e in self.model.environment
                     if isinstance(e, Wind)]
        for obj in time_objs:
            ts = obj.get_timeseries()
            for tse in ts:
                res.append(tse['value'])
        return res

    def full_run(self, rewind=True, logger=False):
        return self.model.full_run(rewind=rewind, logger=logger)

    def step(self):
        return self.model.step()


class ModelBroadcaster(object):
    '''
        Here is where we spawn an array of model consumer processes
        based on the variations in the model configurations we would like.

        Specifically, the variations we would like to use are uncertainty
        variations.
    '''
    def __init__(self, model,
                 wind_speed_uncertainty):
        self.tasks = []
        self.results = []

        for i in range(num_instances):
            self.tasks.append(mp.Queue())
            self.results.append(mp.Queue())

            model_consumer = ModelConsumer(self.tasks[i],
                                           self.results[i],
                                           model)
            model_consumer.start()

    def cmd(self, command, args):
        [t.put((command, args)) for t in self.tasks]
        return [r.get() for r in self.results]

    def stop(self):
        [t.put(None) for t in self.tasks]
