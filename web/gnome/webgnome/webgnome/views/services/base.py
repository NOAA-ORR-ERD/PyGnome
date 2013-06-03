import logging
from functools import wraps


log = logging.getLogger(__name__)


def lock_model():
    """
    A class decorator that wraps any 'get', 'post', 'put' and 'delete' methods
    found in the class with a method decorator that locks a 'model' object if
    found in `self.request.validated` on entry to the method and releases it on
    exit.
    """
    def decorate(cls):
        def decorator(fn):
            @wraps(fn)
            def inner(*args, **kwargs):
                # This is a class decorator and its targets are methods, so
                # the first argument will be ``self``.
                self = args[0]
                model = self.request.validated.get('model', None)
                method_name = '%s.%s' % (self.__class__.__name__, fn.__name__)

                # Lock the model before entering the method body.
                if model:
                    model.lock.acquire()
                    log.info('Model locked by %s' % method_name)

                try:
                    result = fn(*args, **kwargs)
                except:
                    # Release the lock if an exception occurs and propagate
                    # the exception up the stack.
                    if model:
                        model.lock.release()
                        log.exception('Model unlocked after view exception '
                                      'by %s' % method_name)
                    raise

                # Release the lock after the method completes.
                if model:
                    model.lock.release()
                    log.info('Model unlocked by %s' % method_name)

                return result
            return inner

        targets = ['get', 'put', 'post', 'delete']
        for method in [attr for attr in cls.__dict__
                       if attr in targets and callable(getattr(cls, attr))]:
            setattr(cls, method, decorator(getattr(cls, method)))

        return cls
    return decorate


class BaseResource(object):
    def __init__(self, request):
        self.request = request

    @property
    def id(self):
        _id = self.request.matchdict.get('id', None)
        if _id:
            return _id

    @property
    def settings(self):
        return self.request.registry.settings
