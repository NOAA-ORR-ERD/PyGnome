'''
Functions to download data from remote server if required
'''
import os
import urllib2

from progressbar import ProgressBar, Percentage, FileTransferSpeed, ETA, Bar

data_server = 'http://gnome.orr.noaa.gov/py_gnome_testdata'
CHUNKSIZE = 1024*1024   # read 1 MB at a time

def get_datafile(file_):
    if os.path.exists(file_):
        return file_
    else:
        # download file, then return file_ path
        (path_, fname) = os.path.split(file_)
        resp = urllib2.urlopen(os.path.join(data_server, fname))
        
        # progress bar
        widgets = [fname+':      ', Percentage(), ' ', Bar(),' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=int(resp.info().getheader('Content-Length'))).start()
        
        if not os.path.exists(path_):
            os.makedirs(path_)
        
        sz_read = 0
        with open(file_, 'wb') as fh:
            while True: # while sz_read < resp.info().getheader('Content-Length') goes into infinite recursion so break loop for len(data) == 0
                data = resp.read(CHUNKSIZE)
                
                if len(data) == 0:
                    break
                else:
                    fh.write(data)
                    sz_read += len(data)
                    if sz_read >= CHUNKSIZE:
                        pbar.update( CHUNKSIZE)
                    
        pbar.finish()
        return file_     