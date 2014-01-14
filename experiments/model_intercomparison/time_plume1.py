import timeit



print timeit.timeit("[compare_le_count(plume_gen, n) for n in range(100, 401)]",
                    '''
from datetime import datetime, timedelta
from plume_generator import Plume, PlumeGenerator, get_plume_data

release_time = datetime.now()
end_release_time = release_time + timedelta(hours=24)
time_step_delta = timedelta(hours=1).total_seconds()

plume = Plume(position=(28, -78, 0.),
              plume_data=get_plume_data())
plume_gen = PlumeGenerator(release_time=release_time,
                           end_release_time=end_release_time,
                           time_step_delta=time_step_delta,
                           plume=plume)

def compare_le_count(plume_generator, le_count):
    plume_generator.set_le_mass_from_total_le_count(le_count)
    return le_count, sum([sum([r[1] for r in step[1]]) for step in plume_generator])
                    ''',
                    number=1)

