from minignome.models import DBSession
from minignome.models import MyModel
from pyramid.renderers import render_to_response
from pyramid.response import Response
from pyramid.view import view_config 
import os

def my_view(request):
    dbsession = DBSession()
    root = dbsession.query(MyModel).filter(MyModel.name==u'root').first()
    return {'root':root, 'project':'miniGNOME'}

@view_config(renderer='string')
def gnome_view(request):
    response = render_to_response('templates/gnome.pt',
                                 {'gnome':1},
                                 request=request)
    print 'JSON recieved'
    os.system('pwd')
    os.system('python map_from_bna.py LongIslandSoundMap.BNA')
    os.system('mv LongIslandSoundMap.png ./minignome/static/LongIslandSoundMap.png')
    print 'Processing spill request'
    os.system('python particlepng.py')
    os.system('mv ppng.png ./minignome/static/ppng.png')
    print 'Send spill response to GNOME client...'
    #print request.json_body
    #gnomedata = request.json_body
    #print gnomedata
    return response

#@view_config(renderer='string')
def run_spill_view(request):
    #'''
    response = render_to_response('templates/spill.pt',
                                    {'run_spill':1},
                                    request=request)
    
    #'''
    #modeldata = request.json_body
    print 'reading json request as: '
    #print request.json_body
    return response
    