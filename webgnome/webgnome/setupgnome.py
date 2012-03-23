from gnome import c_gnome
import md5
import os
import sys
import datetime
import views
import location_files
from math import cos, sin, radians

'''sets up gnome backend'''

curdir = os.getcwd()
location_files.LongIslandSound.map_file_name = curdir+"/locationdata/LongIslandSound/LongIslandSoundMap.BNA"
location_files.LongIslandSound.topology_file = curdir+"/locationdata/LongIslandSound/tidesWAC.CUR"
location_files.LongIslandSound.shio_file = curdir+"/locationdata/LongIslandSound/CLISShio.txt"
location_files.LowerMississippiRiver.map_file_name = curdir+"/locationdata/LowerMississippiRiver/LMiss.bna"
location_files.LowerMississippiRiver.topology_file = curdir+"/locationdata/LowerMississippiRiver/LMiss.CUR"

def gnomehash(pyson):
    m = md5.new() 
    m.update(str(pyson)) 
    return m.hexdigest()

def gnomesetup(pyson):
    '''sets the model data'''
    movers, params, spills = pyson["movers"], pyson["params"], pyson["spills"]
    return movers, params, spills 

def instantiate_location(params):
    location_name = params['location'].strip().lower()
    if(location_name == "longislandsound"):
        constructor = location_files.LongIslandSound
    elif(location_name == "lowermississippi"):
        constructor = location_files.LowerMississippiRiver
    else:
        print 'Unknown location.'
        exit(-1)
    
    model_start_time = params['model_start_date'] + " " + params['model_start_time']
    model_stop_time = params['model_stop_date'] + " " + params['model_stop_time']
    try:
        timestep = int(params['model_time_step'])
    except:
        print 'exception!'
        exit(-1)

    return constructor(model_start_time, model_stop_time, timestep)
        
def handle_movers(movers, location):
    for mover in movers:
        try:
            type = mover['type'].strip().lower()
            if(type == 'constant_mover'):
                velocity = float(mover['velocity'])
                direction = radians(float(mover['direction']))
                location.add_wind_mover((cos(direction), sin(direction))) #refactor this.
            elif(type == 'cats_mover'):
                pass # this should be handled by default, with the exception of the scale factor, which we're going to leave alone for now.
        except:
            print 'exception in handle_movers!'
            exit(-1)

def run_gnome(pyson):
    """
    computes dir name, runs gnome, and returns the dirname
    """
    movers, params, spills = gnomesetup(pyson)
    dirname = gnomehash(pyson)
    imgpath = os.path.join(curdir+'/static/hashes/',dirname)
    try:
        os.mkdir(imgpath)
    except OSError:
        print 'directory exists!!'

    location = instantiate_location(params)
    
    for spill in spills:
        try:
            num_particles = int(spill['num_particles'])            
            time = spill['date'] + " " + spill['start_time']
            xy = (float(spill['longitude']), float(spill['latitude']))
            location.set_spill(num_particles, time, xy)
        except:
            print 'exception in run_gnome!'
            exit(-1)

    handle_movers(movers, location)
    png_files = location.run(imgpath)
    return dirname, len(png_files)
    
if __name__=='__main__':
    import location_files
    run_gnome({u'movers': [{u'velocity': u'2.3', u'direction': u'275', u'type': u'constant_wind'},], \
                u'spills': [{u'longitude': u'-72.419882', u'latitude': u'41.202120', u'date': u'01/01/2012', u'start_time': u'01:02:00', u'type': u'point_source',  u'num_particles': u'1000'},], \
                    u'params': {u'location': u'LongIslandSound', u'model_start_date': '01/01/2012', u'model_start_time': '01:00:00', u'model_stop_date': '01/01/2012', u'model_stop_time': '12:00:00', u'model_time_step': '900'}})
