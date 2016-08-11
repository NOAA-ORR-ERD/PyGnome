from pprint import PrettyPrinter
pp = PrettyPrinter(indent=2)

import sys
import traceback

from timeit import Timer
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
    def __init__(self, task_port):
        mp.Process.__init__(self)

        self.task_port = task_port

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
                self.stream.send_unicode(dumps('world' * 20))
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                fmt = traceback.format_exception(exc_type, exc_value,
                                                 exc_traceback)
                self.stream.send_unicode(dumps(fmt))


class ModelBroadcaster(GnomeId):
    '''
        Here is where we spawn an array of model consumer processes
        based on the variations in the model configurations we would like.

        More specifically, the model variations we are interested in are
        uncertainty variations.
    '''
    def __init__(self,
                 wind_speed_uncertainties,
                 spill_amount_uncertainties):
        self.context = None
        self.tasks = []
        self.lookup = {}

        self._get_available_ports(wind_speed_uncertainties,
                                  spill_amount_uncertainties)

        self._spawn_consumers()

        self._spawn_tasks()

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
            model_consumer = ModelConsumer(p)
            model_consumer.start()

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
        self.context.destroy()

        self.tasks = []
        self.lookup = {}

    def _to_buff(self, cmd, args):
        return dumps((cmd, args))


if __name__ == '__main__':
    model_broadcaster = ModelBroadcaster(('down', 'normal', 'up'),
                                         ('down', 'normal', 'up'))

    for step in range(1000):
        pp.pprint(model_broadcaster.cmd('step', {}))

    model_broadcaster.stop()
