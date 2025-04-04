#!/usr/bin/env python
'''
Test code for what happens when a model stops mid-run

e.g. outputters should close properly.

'''

from pathlib import Path
import numpy as np

import pytest

import gnome.scripting as gs
import zipfile

from gnome.environment import Wind
from gnome.model import Model

from gnome.spills.spill import Spill, point_line_spill

from gnome.outputters import KMZOutput


HERE = Path(__file__).parent
data_dir = Path(__file__).parent / "sample_data"
output_dir = Path(__file__).parent / "output_dir"
output_dir.mkdir(exist_ok=True)


@pytest.fixture(scope='function')
def model():
    '''
    Utility to setup up a simple, but complete model for tests

    has a single spill and no movers or map.
    '''
    start_time = "2023-03-02T12:00:00"

    pos1 = (-125.16, 48.41)
    pos2 = (-126.01, 48.79)

    model = Model(start_time = start_time,
                  time_step = gs.hours(1),
                  duration = gs.hours(2),
                  )

    model.spills += gs.point_line_spill(num_elements=10,
                                        start_position=pos1,
                                        release_time=start_time,
                                        end_position=pos2,
                                        )

    return model


def test_run_out_of_data_point_wind(model):
    """
    see what happens when the model runs out of data before the run is done
    use different methods of running the model - full_run, step, and next
    """

    print(model.movers)

    filename = output_dir / "stop_in_middle"

    start_time = model.start_time

    # set up a WindMover that's too short.
    times = [model.start_time + (gs.minutes(30) * i) for i in range(3)]
    # long enough record
    # times = [model.start_time + (gs.minutes(30) * i) for i in range(5)]

    winds = gs.wind_from_values([(dt, 5, 90) for dt in times])

    model.movers += gs.WindMover(winds)

    assert len(model.movers) == 1

    kmz = KMZOutput(filename,
                      output_timestep=gs.hours(1),
                      output_zero_step=False,
                      output_last_step=False,
                      output_single_step=False,
                      output_start_time=model.start_time,
                      )
    model.outputters += kmz

    with pytest.raises(ValueError, match="not within the bounds"):
        for step in model:
            print('just ran time step: %s' % model.current_time_step)
            assert step['step_num'] == model.current_time_step

    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created

    # rewind and run again using full_run:
    print('rewinding')
    model.rewind()
    with pytest.raises(ValueError, match="not within the bounds"):
        model.full_run()

    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created

    # rewind and run again using next:
    print('rewinding')
    model.rewind()
    with pytest.raises(ValueError, match="not within the bounds"):
        while True:
            next(model)

    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created


def count_files_in_zip(zip_filepath):
    """
    Counts the number of files in a ZIP archive.

    Args:
        zip_filepath (str): The path to the ZIP file.

    Returns:
        int: The number of files in the ZIP archive.
             Returns -1 if the file is not found or is not a valid ZIP file.
    """
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
            return len(zip_file.namelist())
    except FileNotFoundError:
        print(f"Error: File not found: {zip_filepath}")
        return -1
    except zipfile.BadZipFile:
         print(f"Error: Not a valid ZIP file: {zip_filepath}")
         return -1


def test_run_out_of_data_gridded_movers(model):
    '''
    This tests that the model will output files if the
    model stops mid run (out of data error) with gridded movers
    there are six hours of data starting at model start time
    '''

    filename = output_dir / "stop_in_middle_gridded"

    model.duration = gs.hours(8)
    start_time = model.start_time

    fn = data_dir / 'gridded_wind.nc'
    wind = gs.GridWind.from_netCDF(filename=fn)
    wind_mover = gs.WindMover(wind)
    model.movers += wind_mover

    # create a current mover that's too short
    fn = data_dir / 'gridded_current.nc'
    current_mover = gs.CurrentMover.from_netCDF(filename=fn)
    model.movers += current_mover

    assert len(model.spills) == 1

    kmz = KMZOutput(filename,
                      output_timestep=gs.hours(1),
                      output_zero_step=False,
                      output_last_step=False,
                      output_single_step=False,
                      output_start_time=model.start_time,
                      )
    model.outputters += kmz

    with pytest.raises(ValueError, match="not within the bounds"):
        for step in model:
            print('just ran time step: %s' % model.current_time_step)
            assert step['step_num'] == model.current_time_step

    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created


def test_run_out_of_data_backwards(model):
    '''
    This tests that the model will output files if the
    model stops mid run (out of data error) in a backward run
    '''

    filename = output_dir / "stop_in_middle_backwards"
    model.run_backwards = True

    start_time = model.start_time

    # set up a WindMover that's too short.
    times = [(model.start_time - gs.minutes(60)) + (gs.minutes(30) * i) for i in range(3)]
    # long enough record
    # times = [(start_time - gs.minutes(60*2)) + (gs.minutes(30) * i) for i in range(5)]

    winds = gs.wind_from_values([(dt, 5, 90) for dt in times])

    model.movers += gs.WindMover(winds)

    assert len(model.movers) == 1

    kmz = KMZOutput(filename,
                      output_timestep=gs.hours(1),
                      output_zero_step=False,
                      output_last_step=False,
                      output_single_step=False,
                      output_start_time=model.start_time,
                      )
    model.outputters += kmz

    with pytest.raises(ValueError, match="not within the bounds"):
        for step in model:
            print('just ran time step: %s' % model.current_time_step)
            assert step['step_num'] == model.current_time_step

    len_zip = count_files_in_zip(filename.with_suffix('.kmz'))
    assert len_zip == 3	# two icons and kml, just check it got created


