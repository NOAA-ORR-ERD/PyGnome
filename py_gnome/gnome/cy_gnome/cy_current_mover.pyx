'''
Base class for all current movers
No CyCurrentMover object is ever defined or used; however, this is a good
place to keep all the common properties of current movers
'''

from cy_mover cimport CyMover
from current_movers cimport CurrentMover_c

cdef class CyCurrentMover(CyMover):
    '''
    This is kind of like a virtual base class - only children should
    instantiate the curr_mover object
    '''
    def __cinit__(self):
        if type(self) == CyCurrentMover:
            self.mover = new CurrentMover_c()
            self.curr_mover = dc_mover_to_cmover(self.mover)

    def __dealloc__(self):
        if self.mover is not NULL:
            del self.mover

        self.curr_mover = NULL

    def __init__(self,
                 uncertain_duration=172800,
                 uncertain_time_delay=0,
                 up_cur_uncertain=.3,
                 down_cur_uncertain=-.3,
                 right_cur_uncertain=.1,
                 left_cur_uncertain=-.1,
                 ):
        '''
        :param uncertain_duration: how often does a given uncertain element
            get reset. Default (48 hours = 48*3600 sec)
        :param uncertain_time_delay: when does the uncertainly kick in.
        :param up_cur_uncertain: Scale for uncertainty along the flow
        :param down_cur_uncertain: Scale for uncertainty along the flow
        :param right_cur_uncertain: Scale for uncertainty across the flow
        :param left_cur_uncertain: Scale for uncertainty across the flow
        '''
        # move following two to Mover base class
        self.curr_mover.fDuration = uncertain_duration
        self.curr_mover.fUncertainStartTime = uncertain_time_delay

        self.curr_mover.fUpCurUncertainty = up_cur_uncertain
        self.curr_mover.fDownCurUncertainty = down_cur_uncertain
        self.curr_mover.fLeftCurUncertainty = left_cur_uncertain
        self.curr_mover.fRightCurUncertainty = right_cur_uncertain

    property uncertain_duration:
        def __get__(self):
            return self.curr_mover.fDuration

        def __set__(self, value):
            self.curr_mover.fDuration = value

    property uncertain_time_delay:
        def __get__(self):
            return self.curr_mover.fUncertainStartTime

        def __set__(self, value):
            self.curr_mover.fUncertainStartTime = value

    property up_cur_uncertain:
        def __get__(self):
            return self.curr_mover.fUpCurUncertainty

        def __set__(self, value):
            self.curr_mover.fUpCurUncertainty = value

    property down_cur_uncertain:
        def __get__(self):
            return self.curr_mover.fDownCurUncertainty

        def __set__(self, value):
            self.curr_mover.fDownCurUncertainty = value

    property right_cur_uncertain:
        def __get__(self):
            return self.curr_mover.fRightCurUncertainty

        def __set__(self, value):
            self.curr_mover.fRightCurUncertainty = value

    property left_cur_uncertain:
        def __get__(self):
            return self.curr_mover.fLeftCurUncertainty

        def __set__(self, value):
            self.curr_mover.fLeftCurUncertainty = value

    def __repr__(self):
        """
        Return an unambiguous representation of this object so it can be
        recreated
        """
        return ('{0.__class__.__name__}('
                'uncertain_duration={0.uncertain_duration}, '
                'uncertain_time_delay={0.uncertain_time_delay}, '
                'up_cur_uncertain={0.up_cur_uncertain}, '
                'down_cur_uncertain={0.down_cur_uncertain}, '
                'right_cur_uncertain={0.right_cur_uncertain}, '
                'left_cur_uncertain={0.left_cur_uncertain})'
                .format(self))

    def __str__(self):
        """Return string representation of this object"""
        return ('{0.__class__.__name__} object '
                '- see attributes for more info\n'
                '  uncertain_duration = {0.uncertain_duration}\n'
                '  uncertain_time_delay = {0.uncertain_time_delay}\n'
                '  up_cur_uncertain = {0.up_cur_uncertain}\n'
                '  down_cur_uncertain = {0.down_cur_uncertain}\n'
                '  right_cur_uncertain = {0.right_cur_uncertain}\n'
                '  left_cur_uncertain = {0.left_cur_uncertain}\n'
                .format(self))

    def __reduce__(self):
        return (CyCurrentMover, (self.uncertain_duration,
                                 self.uncertain_time_delay,
                                 self.up_cur_uncertain,
                                 self.down_cur_uncertain,
                                 self.right_cur_uncertain,
                                 self.left_cur_uncertain))
