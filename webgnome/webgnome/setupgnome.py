from gnome import c_gnome
import md5
import os
import datetime
import views

'''sets up gnome backend'''

def gnomehash(pyson):
    m = md5.new() 
    m.update(str(pyson)) 
    return m.hexdigest()

def gnomesetup(pyson):
    '''sets the model data'''
    movers, params, spills = pyson["movers"], pyson["params"], pyson["spills"] 
    return movers, params, spills 

import location_files
curdir = os.getcwd()
location_files.LongIslandSound.map_file_name = curdir+"/webgnome/locationdata/LongIslandSound/LongIslandSoundMap.BNA"
location = location_files.LongIslandSound()

def run_gnome(pyson):
    """
    computes dir name, runs gnome, and returns the dirname
    """
    movers, params, spills = gnomesetup(pyson)
    dirname = gnomehash(pyson)
    imgpath = os.path.join(curdir+'/webgnome/static/hashes',dirname)
    try:
        os.mkdir(imgpath)    
    except OSError:
        print 'OS Error: Directory exists'

    location.reset()
    location.replace_constant_wind_mover(speed = float(movers['velocity']), direction=float(movers['direction']))
    location.set_spill(start_time=datetime.datetime(2012, 2, 14, 14),
                       location = (float(spills['longitude']),float(spills['latitude'])),
                       )
    print spills 
    png_files = location.run(imgpath)
    
    return dirname, len(png_files)
        
if __name__=='__main__':
    import location_files
    run_gnome({u'movers': {u'velocity': 2.3, u'direction': 275, u'type': u'constant_wind'}, u'spills': {u'latitude': u'0', u'date': u'01/01/2012', u'start_time': u'00:00:00', u'type': u'point_source', u'longitude': u'0'}, u'params': {u'type': u'none'}}
    )
    
    
    
    
    
    
    