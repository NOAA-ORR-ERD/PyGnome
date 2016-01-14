import os
from os.path import basename
import numpy as np

from colander import SchemaNode, String, drop

from gnome.persist import base_schema, class_from_objtype

from . import Renderer
import py_gd
from gnome.utilities.map_canvas import MapCanvas
from gnome.utilities.serializable import Field
from gnome.utilities.file_tools import haz_files
from gnome.utilities import projections

from gnome.basic_types import oil_status


class Animation(Renderer):
    def __init__(self, *args, **kwargs):
        '''
        TODO: Recheck this!
        Animation renderer. This creates .gif animations using py_gd. 

        :param repeat: Whether the animation will repeat or not
        :type repeat: Boolean

        :param delay: The delay between frames in 1/100s of a second
        :type delay: int

        :param filename: The name of the animation output file
        :type filename: String
        '''
        self.repeat = True
        self.delay = 50
        if 'repeat' in kwargs:
            self.repeat = kwargs['repeat']
        if 'delay' in kwargs:
            self.delay = kwargs['delay']
        Renderer.__init__(self, *args, **kwargs)
        if 'filename' in kwargs:
            self.anim_filename = kwargs['filename']
        else:
            self.anim_filename = '%s_anim.gif' % os.path.splitext(self._filename)[0]

    def clean_output_files(self):
        # clear out the output dir:
        try:
            os.remove(os.path.join(self.output_dir,
                                   self.background_map_name))
        except OSError:
            # it's not there to delete..
            pass

        anim_file = os.path.join(self.output_dir, self.anim_filename)
        try:
            os.remove(anim_file)
        except OSError:
            # it's not there to delete..
            pass

    def start_animation(self, filename):
        self.animation = py_gd.Animation(filename, self.delay)
        l = 0 if self.repeat else -1
        print 'Starting animation'
        self.animation.begin_anim(self.back_image, l)

    def prepare_for_model_run(self, *args, **kwargs):
        """
        prepares the renderer for a model run.

        Parameters passed to base class (use super): model_start_time, cache

        Does not take any other input arguments; however, to keep the interface
        the same for all outputters, define ``**kwargs`` and pass into the
        base class

        In this case, it draws the background image and clears the previous
        images. If you want to save the previous images, a new output dir
        should be set.
        """
        super(Renderer, self).prepare_for_model_run(*args, **kwargs)
        self.clean_output_files()
        self.draw_background()
        self.start_animation(os.path.join(self.anim_filename))

    def save_foreground_frame(self, animation, delay=50):
        """
        save the foreground image to the specified animation with the specified delay

        :param animation: py_gd animation object to add the frame to
        :type animation: py_gd.Animation

        :param delay: delay after this frame in 1/100s
        :type delay: integer > 0
        """

        self.animation.add_frame(self.fore_image, delay)

    def write_output(self, step_num, islast_step=False):
        """
        Render the map image, according to current parameters.

        :param step_num: the model step number you want rendered.
        :type step_num: int

        :param islast_step: default is False. Flag that indicates that step_num
            is last step. If 'output_last_step' is True then this is written
            out
        :type islast_step: bool

        :returns: A dict of info about this step number if this step
            is to be output, None otherwise.
            'step_num': step_num
            'image_filename': filename
            'time_stamp': time_stamp # as ISO string

        use super to call base class write_output method

        If this is last step, then data is written; otherwise
        prepare_for_model_step determines whether to write the output for
        this step based on output_timestep
        """

        super(Renderer, self).write_output(step_num, islast_step)

        if not self._write_step:
            return None

        self.clear_foreground()
        if self.draw_back_to_fore:
            self.copy_back_to_fore()

        # draw data for self.draw_ontop second so it draws on top
        scp = self.cache.load_timestep(step_num).items()
        if len(scp) == 1:
            self.draw_elements(scp[0])
        else:
            if self.draw_ontop == 'forecast':
                self.draw_elements(scp[1])
                self.draw_elements(scp[0])
            else:
                self.draw_elements(scp[0])
                self.draw_elements(scp[1])

        time_stamp = scp[0].current_time_stamp
        self.draw_timestamp(time_stamp)
        self.save_foreground_frame(self.animation, self.delay)

    def write_output_post_run(self, **kwargs):
        print 'closing animation'
        self.animation.close_anim()