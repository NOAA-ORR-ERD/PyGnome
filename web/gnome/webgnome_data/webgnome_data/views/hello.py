""" Cornice services.
"""
from cornice import Service


hello = Service(name='hello', path='/', description="Simplest app")


@hello.get()
def get_info(request):
    """Returns Hello in JSON."""
    print 'request.session:', request.session
    if 'my_counter' in request.session:
        request.session['my_counter'] += 1
    else:
        request.session['my_counter'] = 0

    request.session.changed()

    return {'Hello': 'World',
            'Counter': request.session['my_counter']}
