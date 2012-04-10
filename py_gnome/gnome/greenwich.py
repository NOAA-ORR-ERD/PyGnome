import time
import calendar
import re
import os

class gwtm:

    """ 
        Handles date time strings across the pyGNOME package. 
        Strings are expected in the 'mm/dd/yyyy hh:mm:ss' format.
        Some more experimentation required, to find why some of the
        following tweaking of values is necessary for compatibility
        with stand-alone GNOME. reference_time is the standard epoch.
    """
                    
    reference_time = '01/01/1970 00:00:00'
    pattern = re.compile('^(\d\d)/(\d\d)/(\d\d\d\d)\s(\d\d):(\d\d):(\d\d)$')

    def __init__(self, time_string, epoch = reference_time):
        """ 
            Initializes the time by parsing the supplied date-time string
            and grouping the resulting tokens so that they can be treated
            more easily.
        """
        tokens = self.parse_datetime(time_string)
        self.date_time = {'month': tokens.group(1), 'day': tokens.group(2), \
                                    'year': tokens.group(3), 'hour': tokens.group(4), \
                                    'minute': tokens.group(5), 'second': tokens.group(6)  }
        self.__set_epoch(epoch)
        self.__to_seconds()
        
    def __set_epoch(self, epoch):
        """ 
            Alters epoch. 
        ++args:
           epoch must be in the expected date-time string fmt.
        """
        self.epoch_date_time = epoch
        if(epoch != self.reference_time):
            self.epoch_seconds = gwtm(epoch).time_seconds
        else:
            self.epoch_seconds = 0
            
    def __to_seconds(self):
        """ converts the object's date-time struct to seconds. """
        dt = self.date_time
        #os.environ['TZ'] = 'US/Eastern'
        #time.tzset()
        time_str = dt['year'] + ' ' + dt['month'] + ' ' + dt['day'] + ' ' + dt['hour'] + ' ' + dt['minute'] + ' ' + dt['second']
        self.struct_time = time.strptime(time_str, "%Y %m %d %H %M %S")
        self.time_seconds = calendar.timegm(self.struct_time)
        self.time_seconds += -self.epoch_seconds
        self.time_seconds += 28800 # temporary hack, but not really sure. seems to work. need to look into this some more. 

    def parse_datetime(self, datetime_string):
        """ 
            checks to ensure that the supplied date-time string adheres to the expected fmt,
            and if it does not, we raise an exception and don't try to salvage the process.
        """
        mtch = self.pattern.match(datetime_string)
        if not mtch:
            raise Exception
        else:
            return mtch
