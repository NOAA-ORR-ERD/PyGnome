#!/usr/bin/env python
"""
Script to test GNOME with all weatherers and response options

Note: this version is using the WEatherers individually
      see script_weathering_run to see an easier way
"""






from gnome import scripting as gs

from gnome.basic_types import datetime_value_2d
import os

from gnome import scripting as gs


from gnome.environment import constant_wind, Water, Waves


from gnome.weatherers import (Emulsification,
                              Evaporation,
                              NaturalDispersion,
                              ChemicalDispersion,
                              Burn,
                              Skimmer,)

# define base directory
base_dir = os.path.dirname(__file__)

water = Water(280.928)
wind = constant_wind(20., 117, 'knots')
waves = Waves(wind, water)


def make_model(images_dir=os.path.join(base_dir, 'images')):
    print('initializing the model')

    # start_time = datetime(2015, 5, 14, 0, 0)

    start_time = gs.asdatetime("2015-05-14")
    # 1 day of data in file
    # 1/2 hr in seconds
    model = gs.Model(start_time=start_time,
                     duration=gs.days(1.75),
                     time_step=60 * 60,
                     uncertain=True)

# #     mapfile = get_datafile(os.path.join(base_dir, './ak_arctic.bna'))
# #
# #     print 'adding the map'
# #     model.map = gs.MapFromBNA(mapfile, refloat_halflife=1)  # seconds
# #
# #     # draw_ontop can be 'uncertain' or 'forecast'
# #     # 'forecast' LEs are in black, and 'uncertain' are in red
# #     # default is 'forecast' LEs draw on top
# #     renderer = Renderer(mapfile, images_dir, image_size=(800, 600),
# #                         output_timestep=timedelta(hours=2),
# #                         draw_ontop='forecast')
# #
# #     print 'adding outputters'
# #     model.outputters += renderer

#     model.outputters += gs.WeatheringOutput(os.path.join(base_dir, 'output'))

    print('adding outputters')

    model.outputters += gs.WeatheringOutput(os.path.join(base_dir, 'output'))

    netcdf_file = os.path.join(base_dir, 'script_weatherers.nc')
    gs.remove_netcdf(netcdf_file)
    model.outputters += gs.NetCDFOutput(netcdf_file,
                                        which_data='all',
                                        output_timestep=gs.hours(1),
                                        surface_conc=None,
                                        )

    print('adding a spill')
    # for now subsurface spill stays on initial layer
    # - will need diffusion and rise velocity
    # - wind doesn't act
    # - start_position = (-76.126872, 37.680952, 5.0),
    end_time = start_time + gs.hours(24)
    oil_file = os.path.join(base_dir, 'alaska-north-slope.json')
    substance = gs.GnomeOil(filename=oil_file)
    spill = gs.surface_point_line_spill(num_elements=100,
                                        start_position=(-164.791878561,
                                                        69.6252597267, 0.0),
                                        release_time=start_time,
                                        end_release_time=end_time,
                                        amount=1000,
                                        #substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE, 1997)',
                                        #substance='oil_ans_mp',
                                        substance=substance,
                                        units='bbl')

    # set bullwinkle to .303 to cause mass goes to zero bug at 24 hours (when continuous release ends)
    spill.substance._bullwinkle = .303
    model.spills += spill

    print('adding a RandomMover:')
    model.movers += gs.RandomMover(diffusion_coef=50000)

    print('adding a wind mover:')

    # series = np.zeros((2,), dtype=datetime_value_2d)
    # series[0] = (start_time, (20, 0))
    # series[1] = (start_time + timedelta(hours=23), (20, 0))

    # wind2 = gs.Wind(timeseries=series, units='knot')

    w_mover = gs.PointWindMover(wind)
    model.movers += w_mover

    print('adding weatherers and cleanup options:')

    # define skimmer/burn cleanup options
    skim1_start = start_time + gs.hours(15.58333)
    skim2_start = start_time + gs.hours(16)

    skim1_active_range = (skim1_start, skim1_start + gs.hours(8))
    skim2_active_range = (skim2_start, skim2_start + gs.hours(12))

    units = spill.units

    skimmer1 = Skimmer(80, units=units, efficiency=0.36,
                       active_range=skim1_active_range)
    skimmer2 = Skimmer(120, units=units, efficiency=0.2,
                       active_range=skim2_active_range)

    burn_start = start_time + gs.hours(36)

    burn = Burn(1000., .1,
                active_range=(burn_start, gs.InfTime()),
                efficiency=.2)

    chem_start = start_time + gs.hours(24)
    c_disp = ChemicalDispersion(fraction_sprayed=0.5,
                                efficiency=0.4,
                                active_range=(chem_start, chem_start + gs.hours(8)),
                                waves=waves,
                                )

    model.environment += water
    model.environment += wind
    model.environment += waves

    model.weatherers += Evaporation(water, wind)
    model.weatherers += Emulsification(waves)
    model.weatherers += NaturalDispersion(waves, water)
    model.weatherers += skimmer1
    model.weatherers += skimmer2
    model.weatherers += burn
    model.weatherers += c_disp

    return model


if __name__ == "__main__":
    gs.make_images_dir()
    model = make_model()
    model.full_run()
    model.save('.')
