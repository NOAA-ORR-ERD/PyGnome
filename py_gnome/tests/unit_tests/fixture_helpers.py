"""
Classes primarily used for testing. 
Test fixtures instantiate these objects, that are passed
to tests.
"""
class MagDirectionUV:
    """
    stores the (r,theta) and corresponding wind_uv provided by user
    """
    def __init__(self,rq,uv):
        self.rq = rq
        self.uv = uv

    def get_rq(self):
        return self.rq

    rq = property(get_rq)

    def get_uv(self):
        return self.uv

    uv = property(get_uv)
