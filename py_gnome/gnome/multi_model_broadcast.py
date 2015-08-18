
import sys
import os
import psutil
import time
import traceback
import logging

from cPickle import loads, dumps
import uuid

import multiprocessing as mp


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
    def __init__(self, task_port, model,
                 ipc_folder='.'):
        mp.Process.__init__(self)

        self.task_port = task_port
        self.model = model
        self.ipc_folder = ipc_folder

    def run(self):
        print '{0}: starting...'.format(self.name)

        # remove any root handlers else we get IOErrors for shared file
        # handlers
        # todo: find a better way to capture log messages for child processes
        root_logger = logging.getLogger()
        handler_list = root_logger.handlers[:]

        root_logger.setLevel(logging.CRITICAL)
        [root_logger.removeHandler(h) for h in handler_list]

        self.cleanup_inherited_files()

        context = zmq.Context()

        self.loop = ioloop.IOLoop.instance()

        sock = context.socket(zmq.REP)
        sock.bind('ipc://{0}/Task-{1}'.format(self.ipc_folder, self.task_port))

        # We need to create a stream from our socket and
        # register a callback for recv events.
        self.stream = zmqstream.ZMQStream(sock, self.loop)
        self.stream.on_recv(self.handle_cmd)

        self.loop.start()

        sock.close()
        context.destroy(linger=0)
        print '{0}: exiting...'.format(self.name)

    def cleanup_inherited_files(self):
        proc = psutil.Process(os.getpid())
        [os.close(c.fd) for c in proc.get_connections()]
        # [os.close(f.fd) for f in proc.get_open_files()]

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
                cmd, args = cmd[:2]
                res = getattr(self, '_' + cmd)(**args)

                self.stream.send_unicode(dumps(res))
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                fmt = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback)

                self.stream.send_unicode(dumps(fmt))

    def _rewind(self):
        return self.model.rewind()

    def _step(self):
        begin = time.time()
        ret = self.model.step()
        end = time.time()

        if 'WeatheringOutput' in ret:
            ret['WeatheringOutput']['response_time'] = end - begin
        return ret

    def _num_time_steps(self):
        return self.model.num_time_steps

    def _full_run(self, rewind=True):
        return self.model.full_run(rewind=rewind)

    def _get_wind_timeseries(self):
        '''
            just some model diag
        '''
        res = []
        wind_objs = [e for e in self.model.environment
                     if isinstance(e, Wind)]

        for obj in wind_objs:
            ts = obj.get_wind_data()
            for tse in ts:
                res.append(tse['value'])

        return res

    def _get_spill_amounts(self):
        return [s.amount for s in self.model.spills]

    def _set_wind_speed_uncertainty(self, up_or_down):
        winds = [e for e in self.model.environment
                 if isinstance(e, Wind)]
        res = [w.set_speed_uncertainty(up_or_down) for w in winds]

        return all(res)

    def _get_spill_container_uncertainty(self):
        return self.model.spills.uncertain

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

    def _get_cache_enabled(self):
        return self.model._cache.enabled

    def _set_cache_enabled(self, enabled):
        self.model._cache.enabled = enabled

    def _get_outputters(self):
        return [o for o in self.model.outputters]

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
                 spill_amount_uncertainties,
                 ipc_folder='.'):
        self.model = model
        self.ipc_folder = ipc_folder
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

        for i in range(len(self.tasks)):
            self._set_new_cache_dir(i)
            self._disable_cache(i)
            self._set_weathering_output_only(i)

    def __del__(self):
        self.stop()

    def _get_available_ports(self,
                             wind_speed_uncertainties,
                             spill_amount_uncertainties):
        self.task_ports = []
        idx = 0

        for wsu in wind_speed_uncertainties:
            for sau in spill_amount_uncertainties:
                self.task_ports.append(uuid.uuid4())
                self.lookup[(wsu, sau)] = idx
                idx += 1

    def _spawn_consumers(self):
        for p in self.task_ports:
            model_consumer = ModelConsumer(p, self.model, self.ipc_folder)
            model_consumer.start()
            self.consumers.append(model_consumer)

    def _spawn_tasks(self):
        self.context = zmq.Context()

        for p in self.task_ports:
            task = self.context.socket(zmq.REQ)
            task.connect('ipc://{0}/Task-{1}'.format(self.ipc_folder, p))

            self.tasks.append(task)

    def cmd(self, command, args, key=None, idx=None, in_parallel=True):
        request = dumps((command, args))

        if idx is not None:
            self.tasks[idx].send(request)
            return loads(self.tasks[idx].recv())
        elif key is not None:
            idx = self.lookup[key]
            self.tasks[idx].send(request)
            return loads(self.tasks[idx].recv())
        else:
            if in_parallel:
                [t.send(request) for t in self.tasks]
                return [loads(t.recv()) for t in self.tasks]
            else:
                out = []
                for t in self.tasks:
                    t.send(request)
                    out.append(loads(t.recv()))
                return out

    def stop(self):
        [t.send(dumps(None)) for t in self.tasks]
        [t.close() for t in self.tasks]

        for c in self.consumers:
            c.join()
        print 'joined all consumers!!!'

        self.context.destroy()

        self.consumers = []
        self.tasks = []
        self.lookup = {}

    def _set_uncertainty(self,
                         wind_speed_uncertainty,
                         spill_amount_uncertainty):
        # py_gnome spill container uncertainty is not used here
        # so we turn it off always
        idx = self.lookup[(wind_speed_uncertainty, spill_amount_uncertainty)]

        self.cmd('set_spill_container_uncertainty',
                 dict(uncertain=False), idx=idx)

        self.cmd('set_wind_speed_uncertainty',
                 dict(up_or_down=wind_speed_uncertainty), idx=idx)

        self.cmd('set_spill_amount_uncertainty',
                 dict(up_or_down=spill_amount_uncertainty), idx=idx)

    def _set_new_cache_dir(self, idx):
        self.cmd('set_cache_dir', {}, idx=idx)

    def _disable_cache(self, idx):
        self.cmd('set_cache_enabled', dict(enabled=False), idx=idx)

    def _set_weathering_output_only(self, idx):
        self.cmd('set_weathering_output_only', {}, idx=idx)
