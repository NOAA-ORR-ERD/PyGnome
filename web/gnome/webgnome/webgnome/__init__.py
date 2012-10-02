from pyramid.config import Configurator
from pyramid.session import UnencryptedCookieSessionFactoryConfig


session_factory = UnencryptedCookieSessionFactoryConfig('ibjas45u3$@#$++slkjf__22134bbb')


def main(global_config, **settings):
    """ This function returns a Pyramid WSGI application.
    """
    config = Configurator(settings=settings, session_factory=session_factory)
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('index', '/')
    config.add_route('model', '/model')
    config.scan()
    return config.make_wsgi_app()
