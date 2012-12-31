import gnome.basic_types
import gnome.utilities.map_canvas
import datetime
import numpy
import os
import time

from cornice.resource import resource, view
from hazpy.file_tools import haz_files
from pyramid.httpexceptions import HTTPNotFound

from webgnome import util, WebPointReleaseSpill, WebWindMover
from webgnome.navigation_tree import NavigationTree
from webgnome.schema import ModelSettingsSchema
from webgnome.views.services.base import BaseResource


@resource(path='/model', renderer='gnome_json',
          description='Model settings, movers and spills.')
class Model(BaseResource):

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Return a JSON tree representation of the entire model, including movers
        and spills.
        """
        model_settings = self.model.to_dict()
        return model_settings
    
    @view()
    def post(self):
        """
        Create a new model, deleting the user's current model if one exists.
        """
        if self.model:
            self.settings.Model.delete(self.model.id)

        model = self.settings.Model.create()

        self.request.session[self.settings.model_session_key] = model.id
        message = util.make_message('success', 'Created a new model.')
    
        return {
            'success': True,
            'model_id': model.id,
            'message': message
        }
    
    @view(validators=util.valid_model_id)
    def delete(self):
        """
        Delete the current model.
        """
        self.settings.Model.delete(self.model.id)
        message = util.make_message('success', 'Deleted the current model.')
    
        return {
            'success': True,
            'model_id': self.model.id,
            'message': message
        }


@resource(path='/model/settings', renderer='gnome_json',
          description='Model settings without movers or spills.')
class ModelSettings(BaseResource):

    @view(validators=util.valid_model_id)
    def get(self):
        return self.model.to_dict(include_movers=False, include_spills=False)

    @view(schema=ModelSettingsSchema, validators=util.valid_model_id)
    def post(self):
        """
        Update settings for the current model.
        """
        app_struct = self.request.validated
        self.model.uncertain = app_struct['uncertain']
        self.model.start_time = app_struct['start_time']
        self.model.time_step = app_struct['time_step']
        self.model.duration = datetime.timedelta(
            days=app_struct['duration_days'],
            seconds=app_struct['duration_hours'] * 60 * 60)
        
        return {
            'success': True
        }


@resource(path='/model/tree', renderer='gnome_json',
          description='A Dynatree JSON representation of the current model.')
class ModelTree(BaseResource):

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Return a JSON representation of the current state of the model, to be used
        to create a tree view of the model in the JavaScript application.
        """
        return NavigationTree(self.model).render()
    
    
@resource(path='/model/run', renderer='gnome_json',
          description='A model run.')
class ModelRun(BaseResource):
    def _get_timestamps(self):
        """
        TODO: Move into ``gnome.model.Model``?
        """
        timestamps = []

        # XXX: Why is _num_time_steps a float? Is this ok?
        for step_num in range(int(self.model._num_time_steps) + 1):
            if step_num == 0:
                dt = self.model.start_time
            else:
                delta = datetime.timedelta(
                    seconds=step_num * self.model.time_step)
                dt = self.model.start_time + delta
            timestamps.append(dt)

        return timestamps

    def _make_runtime(self):
        return time.strftime("%Y-%m-%d-%H-%M-%S")

    @view(validators=util.valid_model_id)
    def post(self):
        """
        Start a run of the user's current model and return a JSON object
        containing the first time step.
        """
        data = {}

        # TODO: This should probably be a method on the model.
        timestamps = self._get_timestamps()
        data['expected_time_steps'] = timestamps
        self.model.timestamps = timestamps
        self.model.uncertain = True

        if not self.model.runtime:
            self.model.runtime = self._make_runtime()

        # TODO: Set separately in spill view.
        if not self.model.spills:
            spill = WebPointReleaseSpill(
                name="Long Island Spill",
                num_LEs=1000,
                start_position=(-72.419992, 41.202120, 0.0),
                release_time=self.model.start_time)

            self.model.add_spill(spill)

        if not self.model.movers:
            start_time = self.model.start_time

            r_mover = gnome.movers.RandomMover(diffusion_coef=500000)
            self.model.add_mover(r_mover)

            series = numpy.zeros((5,), dtype=gnome.basic_types.datetime_value_2d)
            series[0] = (start_time, (30, 50) )
            series[1] = (start_time + datetime.timedelta(hours=18), (30, 50))
            series[2] = (start_time + datetime.timedelta(hours=30), (20, 25))
            series[3] = (start_time + datetime.timedelta(hours=42), (25, 10))
            series[4] = (start_time + datetime.timedelta(hours=54), (25, 180))

            w_mover = WebWindMover(timeseries=series, is_constant=False,
                                   units='mps')
            self.model.add_mover(w_mover)


        # TODO: Set separately in map configuration form/view.
        if not self.model.map:
            map_file = os.path.join(
                self.settings['project_root'],
                'sample_data', 'LongIslandSoundMap.BNA')

            # the land-water map
            self.model.map = gnome.map.MapFromBNA(
                map_file, refloat_halflife=6 * 3600)

            canvas = gnome.utilities.map_canvas.MapCanvas((800, 600))
            polygons = haz_files.ReadBNA(map_file, "PolygonSet")
            canvas.set_land(polygons)
            self.model.output_map = canvas

        # The client requested no cached images, so rewind and clear the cache.
        if self.request.POST.get('no_cache', False):
            self.model.runtime = self._make_runtime()
            self.model.rewind()
            self.model.time_steps = []

        first_step = self._get_time_step()

        if not first_step:
            return {}

        self.model.time_steps.append(first_step)
        data['time_step'] = first_step

        data['background_image'] = self._get_model_image_url(
            'background_map.png')
        data['map_bounds'] = self.model.map.map_bounds.tolist()

        return data

    @view(validators=util.valid_model_id)
    def get(self):
        """
        Get the next step in the model run.
        """
        step = self.get_time_step()

        if not step:
            raise HTTPNotFound

        self.model.time_steps.append(step)

        return {
            'time_step': step
        }
