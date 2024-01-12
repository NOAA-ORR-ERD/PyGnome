"""
tests of how the model and map interact
"""
from pathlib import Path

import gnome.scripting as gs

DATA_DIR = Path(__file__).parent.parent / "sample_data"

# output_dir = os.path.normpath(os.path.join(basedir, "output_dir"))
# testbnamap = os.path.join(datadir, 'MapBounds_Island.bna')
# bna_with_lake = os.path.join(datadir, 'florida_with_lake_small.bna')
# test_tri_grid = os.path.join(datadir, 'small_trigrid_example.nc')


def test_off_map_removed_GnomeMap():
    """
    When elements go off the map, they are marked as off_map

    And then they should be removed by the model before the next time step
    """
    start_time = gs.asdatetime("2023-01-01T12:00")

    model = gs.Model(start_time=start_time,
                     duration=gs.hours(12),
                     time_step=gs.hours(1),
                     )

    model.map = gs.GnomeMap(map_bounds=[(0.1, 0.1),
                                        (0.1, -0.1),
                                        (-0.1, -0.1),
                                        (-0.1, 0.1)])

    model.spills += gs.surface_point_line_spill(num_elements=10,
                                                start_position=(0.0, 0.0),
                                                release_time=model.start_time,
                                                end_release_time=model.start_time + gs.hours(5),
                                                amount=10,
                                                )
    model.movers += gs.CurrentMover(gs.SteadyUniformCurrent(speed=0.5,
                                                            direction=90,
                                                            units="m/s"))

    # there are a different number of elements at each step
    num_elements = [0, 2, 4, 6, 8, 10, 10, 9, 7, 5, 3, 1, 0]
    mass_balance_expected = {'beached': 0.0, 'off_maps': 10.0, 'floating': 0.0, 'amount_released': 10.0}

    for step in model:
        step_num = step['step_num']
        pos = model.get_spill_property('positions')
        sc = model.get_spill_property('status_codes')
        assert len(pos) == num_elements[step_num]
        mass_balance = model.spills.items()[0].mass_balance
    # check the final mass balance.
    for key, expected in mass_balance_expected.items():
       assert mass_balance[key] == expected, f"{key}, {expected=}, {mass_balance[key]}"


def test_off_map_removed_MapFromBNA():
    """
    When elements go off the map, they are marked as off_map

    And then they should be removed by the model before the next time step
    """
    start_time = gs.asdatetime("2023-01-01T12:00")

    model = gs.Model(start_time=start_time,
                     duration=gs.hours(12),
                     time_step=gs.hours(1),
                     )

    map = gs.MapFromBNA(DATA_DIR / 'MapBounds_Island.bna')
    model.map = map

    # print(model.map.map_bounds)
    # [[-127.465333   48.3294  ]
    #  [-126.108847   48.3294  ]
    #  [-126.108847   47.44727 ]
    #  [-127.465333   47.44727 ]]

    # model.outputters += gs.Renderer((DATA_DIR / 'MapBounds_Island.bna'),
    #                                 draw_map_bounds=True,
    #                                 formats=['gif']
    #                                 )


    model.spills += gs.surface_point_line_spill(num_elements=10,
                                                start_position=(-126.2, 48.0),
                                                release_time=model.start_time,
                                                end_release_time=model.start_time + gs.hours(5),
                                                amount=10.0
                                                )
    model.movers += gs.CurrentMover(gs.SteadyUniformCurrent(speed=0.5,
                                                            direction=90,
                                                            units="m/s"))


    # there are a different number of elements at each step
    num_elements = [0, 2, 4, 6, 7, 7, 5, 3, 1, 0, 0, 0, 0]

    for step in model:
        step_num = step['step_num']
        pos = model.get_spill_property('positions')
        sc = model.get_spill_property('status_codes')
        mass = model.get_spill_property('mass')
        print(f"{pos=}")
        print(f"{sc=}")
        print(f"{mass=}")
        mass_balance = model.spills.items()[0].mass_balance
        print(mass_balance)
        assert len(pos) == num_elements[step_num]


def test_mass_balance_MapFromBNA():
    """
    This has elements beaching and also going off map
    Checking that the mass balance works for those.
    """
    start_time = gs.asdatetime("2023-01-01T12:00")

    model = gs.Model(start_time=start_time,
                     duration=gs.hours(12),
                     time_step=gs.hours(1),
                     )

    map = gs.MapFromBNA(DATA_DIR / 'MapBounds_Island.bna')
    model.map = map

    # print(model.map.map_bounds)
    # [[-127.465333   48.3294  ]
    #  [-126.108847   48.3294  ]
    #  [-126.108847   47.44727 ]
    #  [-127.465333   47.44727 ]]

    model.outputters += gs.Renderer((DATA_DIR / 'MapBounds_Island.bna'),
                                    draw_map_bounds=True,
                                    point_size=4,
                                    formats=['gif']
                                    )


    # print(model.map.map_bounds)
    # [[-127.465333   48.3294  ]
    #  [-126.108847   48.3294  ]
    #  [-126.108847   47.44727 ]
    #  [-127.465333   47.44727 ]]

    # This goes off the map
    model.spills += gs.surface_point_line_spill(num_elements=10,
                                                start_position=(-126.2, 48.2),
                                                release_time=model.start_time,
                                                end_release_time=model.start_time + gs.hours(5),
                                                amount=10.0
                                                )

    # this one hits land
    model.spills += gs.surface_point_line_spill(num_elements=10,
                                                start_position=(-127.0, 47.7),
                                                release_time=model.start_time,
                                                end_release_time=model.start_time + gs.hours(5),
                                                amount=10.0
                                                )

    model.movers += gs.CurrentMover(gs.SteadyUniformCurrent(speed=0.5,
                                                            direction=90,
                                                            units="m/s"))

    # there are a different number of elements at each step
    num_elements = [0, 4, 8, 12, 15, 17, 15, 13, 11, 10, 10, 10, 10]
    mass_balance_expected = {'beached': 10.0, 'off_maps': 10.0, 'floating': 0.0, 'amount_released': 20.0}


    for step in model:
        step_num = step['step_num']
        print(f"{step_num=}")
        pos = model.get_spill_property('positions')
        sc = model.get_spill_property('status_codes')
        mass = model.get_spill_property('mass')
        # print(f"{pos=}")
        # print(f"{sc=}")
        print(f"{mass=}")
        print(len(mass))
        assert len(pos) == num_elements[step_num]
        mass_balance = model.spills.items()[0].mass_balance
        print(mass_balance)

    # check the final mass balance.
    for key, expected in mass_balance_expected.items():
       assert mass_balance[key] == expected, f"{key}, {expected=}, {mass_balance[key]}"


