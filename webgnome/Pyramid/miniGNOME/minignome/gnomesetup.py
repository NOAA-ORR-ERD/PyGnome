import c_gnome
import md5
import os
import datetime

'''dict that contains CGNOME keys with JSON values'''
bridge = {} 

def gnomehash(pyson):
    m = md5.new() 
    m.update(str(pyson)) 
    return m.hexdigest()

def gnomesetup(pyson):
    '''sets up a prototypal instance of the model'''
    spills, params, movers = pyson[2], pyson[1], pyson[0] 
#    print 
##    print spills 
#    print
    
    #print params 
    #print movers
    #movers 
    #m = {}
    #m["movers"] = {}
    #m["movers"] = movers[movers.index({'movers': 'movers'})+1: movers.index({'params': 'params'})]
    #m = m.pop() 
    #params
    #p = {} 
    #p["params"] = params[params.index({'params': 'params'})+1: params.index({'spills': 'spills'})]
    #p = p.pop()
    #spills 
    s  = {}
    for item in spills:
        for key in item:
            s[key] = item[key]
#    print s
    m  = {}
    for item in movers:
        for key in item:
            m[key] = item[key]
#    print m
    p  = {}
    for item in params:
        for key in item:
            p[key] = item[key]
 #   print s
    #s["movers"] = spills[movers.index({'spills': 'spills'})+1: spills.index({'params': 'params'})]
    #s.pop()
    return m, p, s


import location_files
location_files.LongIslandSound.map_file_name = "/Users/dan.smith-mathews/Desktop/miniGNOME/minignome/locationdata/LongIslandSound/LongIslandSoundMap.BNA"
location = location_files.LongIslandSound()

def run_gnome(pyson):
    """
    computes dir name, runs gnome, and returns the dirname
    """
    movers, params, spills = gnomesetup(pyson)
    dirname = gnomehash(pyson)
    imgpath = os.path.join('/Users/dan.smith-mathews/Desktop/miniGNOME/minignome/static/hashes',dirname)
    try:
        os.mkdir(imgpath)    
    except OSError:
        print 'OS Error: Directory exists'

    location.reset()
    location.replace_constant_wind_mover(speed = movers['velocity'], direction=movers['direction'])
    location.set_spill(start_time=datetime.datetime(2012, 2, 14, 14),
                       location = (spills['longitude'],spills['latitude']),
                       )
    png_files = location.run(imgpath)
    
    return dirname, len(png_files)
        
if __name__=='__main__':
    a = [[{"movers" : "movers"},  {"type" : "constant_wind"}, {"velocity" : 2.3}, {"direction" : 275}], 
    [{"params" : "params"},  {"type" : "none"},	{"p1" : "1"}, {"p2" : "2"}, {"p3" : "3"}], 
    [{"spills" : "spills"},  {"type" : "point_source"}, {"latitude" : 41.112120}, {"longitude" : -72.719832}, {"date" : "2012:2:14"}, {"start time" : "1400:00"}]]
     #location = (-72.719832,
     #               41.112120))
    import location_files
    run_gnome(a)
    
    
    
    
    
    
    