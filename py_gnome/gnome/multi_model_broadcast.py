from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import sys
import time
import traceback

from cPickle import loads, dumps

import multiprocessing
mp = multiprocessing


import zmq
from zmq.eventloop import ioloop, zmqstream

from gnome import GnomeId
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
    def __init__(self, task_port, model):
        mp.Process.__init__(self)

        self.task_port = task_port
        self.model = model

    def run(self):
        context = zmq.Context()

        self.loop = ioloop.IOLoop.instance()

        sock = context.socket(zmq.REP)
        sock.bind('ipc://ModelConsumerTask{0}'.format(self.task_port))

        # We need to create a stream from our socket and
        # register a callback for recv events.
        self.stream = zmqstream.ZMQStream(sock, self.loop)
        self.stream.on_recv(self.handle_cmd)

        self.loop.start()

        context.destroy(linger=0)
        print '{0}: exiting...'.format(self.name)
        sys.exit()

    def handle_cmd(self, msg):
        '''
            the IOLoop only uses recv_multipart(), so we will always get
            a list of byte strings.
        '''
        cmd = loads(msg[0])
        if cmd is None:
            # Poison pill means shutdown
            self.loop.stop()
        else:
            try:
                res = getattr(self, '_' + cmd[0])(**cmd[1])
                self.stream.send_unicode(dumps(res))
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                fmt = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback)
                self.stream.send_unicode(dumps(fmt))

    def _rewind(self):
        return self.model.rewind()

    def _step(self):
        begin = time.clock()
        ret = self.model.step()
        end = time.clock()
        ret['response_time'] = end - begin
        return ret

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

    def _get_weatherer_attribute(self, idx, attr):
        return getattr(self.model.weatherers[idx], attr)

    def _set_weathering_output_only(self):
        del_list = [o for o in self.model.outputters
                    if not isinstance(o, WeatheringOutput)]
        for dl in del_list:
            del self.model.outputters[dl.id]


class ModelBroadcaster(GnomeId):
    '''
        Here is where we spawn an array of model consumer processes
        based on the variations in the model configurations we would like.

        More specifically, the model variations we are interested in are
        uncertainty variations.
    '''
    def __init__(self, model,
                 wind_speed_uncertainties,
                 spill_amount_uncertainties):
        self.model = model
        self.context = None
        self.consumers = []
        self.tasks = []
        self.lookup = {}

        self._get_available_ports(wind_speed_uncertainties,
                                  spill_amount_uncertainties)

        self._spawn_consumers()

        self._spawn_tasks()

        for wsu in wind_speed_uncertainties:
            for sau in spill_amount_uncertainties:
                self._set_uncertainty(wsu, sau)

        for t in self.tasks:
            self._set_new_cache_dir(t)
            self._set_weathering_output_only(t)

    def __del__(self):
        self.stop()

    def _get_available_ports(self,
                             wind_speed_uncertainties,
                             spill_amount_uncertainties):
        self.task_ports = []
        idx = 0

        for wsu in wind_speed_uncertainties:
            for sau in spill_amount_uncertainties:
                self.task_ports.append(idx)
                self.lookup[(wsu, sau)] = idx
                idx += 1

    def _spawn_consumers(self):
        for p in self.task_ports:
            model_consumer = ModelConsumer(p, self.model)
            model_consumer.start()
            self.consumers.append(model_consumer)

    def _spawn_tasks(self):
        self.context = zmq.Context()

        for p in self.task_ports:
            task = self.context.socket(zmq.REQ)
            task.connect('ipc://ModelConsumerTask{0}'.format(p))

            self.tasks.append(task)

    def cmd(self, command, args, key=None):
        if key is None:
            [t.send(self._to_buff(command, args)) for t in self.tasks]
            return [loads(t.recv()) for t in self.tasks]
        else:
            idx = self.lookup[key]
            self.tasks[idx].send(self._to_buff(command, args))
            return loads(self.tasks[idx].recv())

    def stop(self):
        [t.send(dumps(None)) for t in self.tasks]
        [t.close() for t in self.tasks]

        for c in self.consumers:
            c.join()

        self.context.destroy()

        self.tasks = []
        self.lookup = {}

    def _to_buff(self, cmd, args):
        return dumps((cmd, args))

    def _set_uncertainty(self,
                         wind_speed_uncertainty,
                         spill_amount_uncertainty):
        # py_gnome spill container uncertainty is not used here
        # so we turn it off always
        index = self.lookup[(wind_speed_uncertainty, spill_amount_uncertainty)]
        cmd = self._to_buff('set_spill_container_uncertainty',
                            dict(uncertain=False))
        self.tasks[index].send(cmd)
        self.tasks[index].recv()

        cmd = self._to_buff('set_wind_speed_uncertainty',
                            dict(up_or_down=wind_speed_uncertainty))
        self.tasks[index].send(cmd)
        self.tasks[index].recv()

        cmd = self._to_buff('set_spill_amount_uncertainty',
                            dict(up_or_down=spill_amount_uncertainty))
        self.tasks[index].send(cmd)
        self.tasks[index].recv()

    def _set_new_cache_dir(self, task):
        task.send(self._to_buff('set_cache_dir', {}))
        task.recv()

    def _set_weathering_output_only(self, task):
        task.send(self._to_buff('set_weathering_output_only', {}))
        task.recv()
