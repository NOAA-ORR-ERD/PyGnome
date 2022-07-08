'''
Binary Outputter
For use in Gnome Analyst
'''

import os
import copy
from glob import glob
import zipfile

import numpy as np

from colander import SchemaNode, Boolean, String, drop

from gnome.utilities.time_utils import date_to_sec
from datetime import datetime
from gnome.basic_types import oil_status

from .outputter import Outputter, OutputterFilenameMixin, BaseOutputterSchema


le_dtype = np.dtype(np.dtype([('Lat', np.float32),
                              ('Lon', np.float32),
                              ('Release_time', np.float32),
                              ('AgeWhenReleased', np.float32),
                              ('BeachHeight', np.float32),
                              ('nMap', np.int32),
                              ('pollutant', np.int32),
                              ('WindKey', np.int32),
                              ]))

le_dtype = le_dtype.newbyteorder('B') # or ">"

header_dtype = np.dtype(np.dtype([('name', np.string_, 10),
                              ('day', np.int16),
                              ('month', np.int16),
                              ('year', np.int16),
                              ('hour', np.int16),
                              ('minute', np.int16),
                              ('current_time', np.float32),
                              ('version', np.float32),
                              ('num_LE', np.int32),
                              ]))

header_dtype = header_dtype.newbyteorder('B') # or ">"


class BinaryOutputSchema(BaseOutputterSchema):
    filename = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    zip_output = SchemaNode(
        Boolean(), missing=drop, save=True, update=True
    )


class BinaryOutput(OutputterFilenameMixin,Outputter):
    '''
    class that outputs GNOME trajectory results on a time step by time step basis
    this is the format output by desktop GNOME for use in Gnome Analyst
    '''
    _schema = BinaryOutputSchema

    def __init__(self,
                 filename,   
                 zip_output=True,   
                 **kwargs):
        '''
        :param str filename: full path and basename of the zip file

        :param zip_output=True: whether to zip up the output binary files

        other arguments as defined in the Outputter class
        '''
        super(BinaryOutput, self).__init__(filename=filename,
                                               **kwargs)

        name, ext = os.path.splitext(self.filename)
        self.name = name
        dirname, filename = os.path.split(self.filename)
        self.output_dir = dirname
        #self._c_filename = '{0}FORCST'.format(name)
        #self._u_filename = '{0}UNCRTN'.format(name)
        self.zip_output = zip_output
        self.filedir = os.path.dirname(filename)
        self.file_num = 0


    def prepare_for_model_run(self,
                              model_start_time,
                              spills,
                              uncertain = False,
                              **kwargs):
        """
        .. function:: prepare_for_model_run(model_start_time,
                                            cache=None,
                                            uncertain=False,
                                            spills=None,
                                            **kwargs)

        Reset file_num and uncertainty

        This must be done in prepare_for_model_run because if model _state
        changes, it is rewound and re-run from the beginning.

        If there are existing output files, they are deleted here.

        This takes more than standard 'cache' argument. Some of these are
        required arguments - they contain None for defaults because non-default
        argument cannot follow default argument. Since cache is already 2nd
        positional argument for Renderer object, the required non-default
        arguments must be defined following 'cache'.

        If uncertainty is on, then SpillContainerPair object contains
        identical _data_arrays in both certain and uncertain SpillContainer's,
        the data itself is different, but they contain the same type of data
        arrays. If uncertain, then data arrays for uncertain spill container
        are written to separate files.

        .. note::
            Does not take any other input arguments; however, to keep the
            interface the same for all outputters, define kwargs in case
            future outputters require different arguments.
        """
        if not self.on:
            return

        super(BinaryOutput, self).prepare_for_model_run(model_start_time,
                                                       spills,
                                                       **kwargs)
        self.file_num = 0
        self.uncertain = uncertain


    def write_output(self, step_num, islast_step=False):
        '''
        Write data from time step to binary file 
        '''
        super(BinaryOutput, self).write_output(step_num, islast_step)

        #if not self._write_step:
        if self.on is False:
            return None

        if self._write_step:
            for sc in self.cache.load_timestep(step_num).items():
                # loop through uncertain and certain LEs
                # extract the data
                if sc.uncertain:
                    #self._u_filename = '{0}UNCRTN.{1:03d}'.format(self.name,self.file_num)
                    filename = '{0}UNCRTN.{1:03d}'.format(self.name,self.file_num)
                else:
                    #self._c_filename = '{0}FORCST.{1:03d}'.format(self.name,self.file_num)
                    filename = '{0}FORCST.{1:03d}'.format(self.name,self.file_num)

                self._file_exists_error(filename)

                output_filename = self.output_to_file(filename, sc)


        if islast_step:
            num_files = self.file_num + 1
            self._zip_binary_files(num_files)
 
        if not self._write_step:
            return None

        self.file_num += 1
        
        output_info = {'time_stamp': sc.current_time_stamp.isoformat(),
                       'output_filename': self.filename}

        return output_info

    def output_to_file(self, filename, sc):
        """dump a timestep's data into binary files for Gnome Analyst"""

        time_stamp = sc.current_time_stamp
        version = 16.1
        day = time_stamp.day
        month = time_stamp.month
        year = time_stamp.year
        hour = time_stamp.hour
        minute = time_stamp.minute
        second = time_stamp.second
        name = 'L.E. FILE'
        num_LE = len(sc['positions'])

        refYear = year - (year % 4)
	
        ref_day = 0o1
        ref_month = 0o1
        ref_year = refYear	#back to 4 digit year
        ref_hour = 0
        ref_minute = 0
        ref_second = 0

        model_time = date_to_sec(sc.current_time_stamp)
        ossm_datetime = datetime(ref_year, ref_month, ref_day, ref_hour, ref_minute, ref_second)
        ossm_time = date_to_sec(ossm_datetime)
        seconds = model_time - ossm_time
        current_time = seconds/3600.0;

        LE_header = np.zeros((1,), dtype=header_dtype)
        LE_header['name'] = name
        LE_header['day'] = day
        LE_header['month'] = month
        LE_header['year'] = year
        LE_header['hour'] = hour
        LE_header['minute'] = minute
        LE_header['current_time'] = current_time
        LE_header['version'] = version
        LE_header['num_LE'] = num_LE

        LEs = np.zeros((num_LE,), dtype=le_dtype)
        positions = sc['positions']
        longitude = -1 * sc['positions'][:, 0]
        latitude = sc['positions'][:, 1]

        release_time = model_time - sc['age'][:]
        release_time -= ossm_time
        release_time /= 3600

        ageInHrsWhenReleased = 0
        release_age = ageInHrsWhenReleased  - (current_time - release_time)
        release_age = release_age.clip(min=0)

        LEs['Lat'] = latitude
        LEs['Lon'] = longitude
        LEs['Release_time'] = release_time 
        LEs['AgeWhenReleased'] = release_age
        LEs['nMap'] = 1 #the off map LEs are not included, would be set to 7
        LEs['BeachHeight'][sc['status_codes'] == oil_status.on_land] = -50 #ossm value

        with open(filename, 'wb') as binfile:
            LE_header.tofile(binfile)
            LEs.tofile(binfile)

        return filename

    def _zip_binary_files(self, num_files):
        if self.zip_output is True:
            zfilename = self.name + '.zip'
            zipf = zipfile.ZipFile(zfilename, 'w')

            for file_num in range(0, num_files):
                forcst_file = '{0}FORCST.{1:03d}'.format(self.name,file_num)
                dir, file_to_zip = os.path.split(forcst_file)
                zipf.write(forcst_file,
                           arcname=file_to_zip)
                os.remove(forcst_file)
                if self.uncertain is True:
                    uncrtn_file = '{0}UNCRTN.{1:03d}'.format(self.name,file_num)
                    dir, file_to_zip = os.path.split(uncrtn_file)
                    zipf.write(uncrtn_file,
                               arcname=file_to_zip)
                    os.remove(uncrtn_file)

            zipf.close()

    def clean_output_files(self):
        if self.output_dir:
            files = glob(os.path.join(self.output_dir,
                                      '*FORCST.*'))
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass

            files = glob(os.path.join(self.output_dir,
                                      '*UNCRTN.*'))
            for f in files:
                try:
                    os.remove(f)
                except:
                    pass

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
