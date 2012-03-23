from pyramid.view import view_config
from pyramid.httpexceptions import exception_response
import setupgnome 
import os
from gnomefs import gnome_proc

@view_config(route_name='home', renderer='templates/mytemplate.pt')
def my_view(request):
    return {}
    
@view_config(route_name='welcome', renderer='templates/welcome.pt')
def welcome(request):
    return {}
        
@view_config(route_name='location', renderer='templates/location.pt')
def location(request):
    return {}

@view_config(route_name='gnome', renderer='templates/gnome.pt')
def gnome(request): 
    return {} 

jobs = []
@view_config(route_name='run', renderer='json')
def run(request):
    try: 
        jsondata = request.json_body
        print "json request recieved in the run view: "
        print jsondata
        print 'attempting subproc setupgnome proc: '
        hash, numframes = gnome_proc('gnomeproc', setupgnome.run_gnome, jsondata)
        jobs.append(hash)
        print 'current jobs: ', jobs
    except: 
        print "json request failed" 

    return {'hash': hash, 'numframes': numframes}
