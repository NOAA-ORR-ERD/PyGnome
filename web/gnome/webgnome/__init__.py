from pyramid.config import Configurator

def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('welcome', '/welcome')
    config.add_route('location', '/location')
    config.add_route('gnome', '/gnome')
    config.add_route('run', '/run')
    config.add_route('longislandsound', '/longislandsound')
    config.add_route('lmiss', '/lmiss')
    config.scan()
    return config.make_wsgi_app()
