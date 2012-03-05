from pyramid.view import view_config
from pyramid.httpexceptions import exception_response
import setupgnome 
import os

@view_config(route_name='home', renderer='templates/mytemplate.pt')
def my_view(request):
    return {'project':'webgnome'}

@view_config(route_name='gnome', renderer='templates/gnome.pt')
def gnome(request): 
    return {} 

@view_config(route_name='run', renderer='json')
def run(request):
    try: 
        jsondata = request.json_body
        print "json request: "
        print jsondata
    except: 
        print "json request failed" 
    
    hash, numframes = setupgnome.run_gnome(jsondata)
    return {'hash': hash, 'numframes': numframes}

