'''
Constants are mostly used internally. They are all documented here to keep them
in one place. The 'units' serve as documentation **do not mess with them**.
There is no unit conversion when using these constants - they are used as is
in the code, implicitly assuming the units are SI and untouched.

ToDo:
 - add a few comments saying what these are, and fill in the rest of the units
 - update with more meaningful names
'''

gas_constant = 8.314
atmos_pressure = 101325.0
drop_min = 1.0e-6
drop_max = 1.0e-5
volume_entrained = 3.9e-8
ka = 1.0e-4
gravity = 9.80665  # gravitation acceleration m/s^2
water_kinematic_viscosity = 0.000001 # m^2/s

units = {'gas_constant': 'J/(K mol)',
         'pressure': 'Pa',
         'min emul drop diameter': 'm',
         'max emul drop diameter': 'm',
         'volume of oil entrained': 'm^3',
         'oil sticking term': 'm^3/kg',
         'acceleration': 'm/s^2'}
