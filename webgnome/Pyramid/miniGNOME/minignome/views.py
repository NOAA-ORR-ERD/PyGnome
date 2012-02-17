from minignome.models import DBSession
from minignome.models import MyModel
from pyramid.renderers import render_to_response
from pyramid.response import Response
from pyramid.view import view_config 
import os
import gnomebridge
import location_files
import gnomesetup


def my_view(request):
    dbsession = DBSession()
    root = dbsession.query(MyModel).filter(MyModel.name==u'root').first()
    return {'root':root, 'project':'miniGNOME'}

data = []

def sync_data(d):
    global data
    data = d
    return(d)

#gnomedata = [{'speed': '0'}, {'direction': '0'}, {'latitude': '0'}, {'longitude': '0'}, {'date': '00:00:00'}, {'time': '01/01/2012'}]
#[{"movers" : "movers"},  {"type" : "constant_wind"}, {"velocity" : 2.3}, {"direction" : 275}], 
#[{"params" : "params"},  {"type" : "none"},	{"p1" : "1"}, {"p2" : "2"}, {"p3" : "3"}], 
#[{"spills" : "spills"},  {"type" : "point source"}, {"latitude" : "latitude"}, {"longitude" : "longitude"}, {"date" : "date"}, {"start time" : "time"}]

#@view_config(renderer='string')
def gnome_view(request):

    response = render_to_response('templates/gnome.pt',
                                 {'gnome':1},
                                 request=request)
    #print 'JSON recieved'
    #os.system('pwd')
    #os.system('python map_from_bna.py LongIslandSoundMap.BNA')
    #os.system('mv LongIslandSoundMap.png ./minignome/static/LongIslandSoundMap.png')
    #print 'Processing spill request'
    #os.system('python particlepng.py')
    #os.system('mv ppng.png ./minignome/static/ppng.png')
    #print 'Send spill response to GNOME client...'
    #print request.json_bodylocal 
    #global gnomedata 
    #gnomedata = request.json_body
    #gnomefeed(request.json_body)
    #print 'this is our json request',request.json_body
    #print 'this is type', type(request.json_body) 
    #gnomebridge.setupgnome(request.json_body)
    #f = open('model.txt','w')
    #f.write(str(gnomedata))
    #f.close()
    #print "gnomedata: ",gnomedata
    #print "gnomestring:", str(gnomedata)
    
    #location = request.json_body
    #location_instance = location_files.Location(location)
    #location_files.map(gnomesetup(location))
    
    return response
    
#@view_config(renderer='string')

count = 1

@view_config(renderer ='json')
def run_spill_view(request):
        global count
        count += 1
        print 'run spill view, count: %i'%count
        '''
        response = render_to_response('templates/spill.pt',
                                        {'run_spill':1},
                                        request=request)
        '''
        

        # orent the POST worked:
        jsondata = [[{"movers" : "movers"},  {"type" : "constant_wind"}, {"velocity" : 2.3}, {"direction" : 275}], 
        [{"params" : "params"},  {"type" : "none"},	{"p1" : "1"}, {"p2" : "2"}, {"p3" : "3"}], 
        [{"spills" : "spills"},  {"type" : "point_source"}, {"latitude" : 41.112120}, {"longitude" : -72.719832}, {"date" : "2012:2:14"}, {"start time" : "1400:00"}]]


        
        #hash, numframes = gnomesetup.run_gnome(jsondata)  
        
        return {'hash':'hash', 'numberframes':'numframes'}

#@view_config(renderer ='string')
def data_view(request):
 
    response = render_to_response('templates/data.pt',
                                 {'gnome':1},
                                  request=request)
    
    #print request.json_body
    print request.POST 
    #print request.GET
    #print request.body
    return response
    

