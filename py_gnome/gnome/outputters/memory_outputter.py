"""
An outputter that stores in memory

This can be handy for tests, etc.

NOTE: not complete!
"""

from gnome.outputters.outputter import Outputter


class DataBuffer(object):
    def __init__(self):
        self.certain = []
        self.uncertain = []


class MemoryOutputter(Outputter):

    arrays_to_output = ["mass", "positions", "age", "status_codes"]

    def prepare_for_model_run(self, *args, **kwargs):
        super(MemoryOutputter, self).prepare_for_model_run(*args, **kwargs)

        self.data_buffer = DataBuffer()

    def write_output(self, step_num, islast_step=False):
        """
        Save data at each output timestep

        :param int step_num: the model step number you want rendered.
        :param bool islast_step: Default is False.
                                 Flag that indicates that step_num is
                                 last step.
                                 If 'output_last_step' is True then this is
                                 written out

        Use super to call base class write_output method
        """
        super(MemoryOutputter, self).write_output(step_num, islast_step)

        if self.on is False or not self._write_step:
            return None

        for sc in self.cache.load_timestep(step_num).items():
            if sc.uncertain:
                raise NotImplementedError("MemoryOutputter Doesn't handle "
                                          "uncertainty yet.")
            # if sc.uncertain and self._u_filename is not None:
            #     file_ = self._u_filename
            # else:
            #     file_ = self.filename

            time_stamp = sc.current_time_stamp
            data = {'time': time_stamp}
            self.data_buffer.certain.append(data)
            # with nc.Dataset(file_, 'a') as rootgrp:
            #     rg_vars = rootgrp.variables
            #     idx = len(rg_vars['time'])

            #     rg_vars['time'][idx] = nc.date2num(time_stamp,
            #                                        rg_vars['time'].units,
            #                                        rg_vars['time'].calendar)
            #     pc = rg_vars['particle_count']
            #     pc[idx] = len(sc)

            #     _end_idx = self._start_idx + pc[idx]

            # add the data:
            for var_name in self.arrays_to_output:
                data[var_name] = sc[var_name]

            # write mass_balance data
            if sc.mass_balance:
                for key, val in sc.mass_balance.items():
                    print(sc.mass_balance)
        #             if key not in grp.variables:
        #                 self._create_nc_var(grp,
        #                                     key, 'float', ('time', ),
        #                                     (self._chunksize,)
        #                                     )
        #             grp.variables[key][idx] = val

        # self._start_idx = _end_idx  # set _start_idx for the next timestep

        return {'buffer': "memory",
                'time_stamp': time_stamp}

