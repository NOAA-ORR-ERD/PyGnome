#!/usr/bin/env python2

"""
tests for teh json outputter

VERY incomplete!
"""
import json
import pytest

from gnome.outputters.json import SpillJsonOutput
from gnome.utilities.appearance import SpillAppearance, Colormap

def test_deserialize():
    '''
    attempting to test if you can create one from the
    kind of json we'd get from teh WebAPI
    '''

    json_from_api = {#'on': True,
                     'obj_type': 'gnome.outputters.json.SpillJsonOutput',
                     'name': None,
                     'output_zero_step': True,
                     'output_start_time': None,
                     'output_last_step': True,
                     'surface_conc': 'kde',
                     'json_': 'webapi',
                     '_additional_data': [],
                     # 'id': u'4c64ca4f-4cbc-11e8-8899-acbc32795771',
                     }

    # create an outputter:

    sjo = SpillJsonOutput.deserialize(json_from_api)

    # fixme -- need more tests!
    print(sjo)


def test_generate_cutoff_struct_missing():
    """
    This Appearance is missig key attributes

    It shouldn't fail, but should produce and empty dict
    """
    class MockSpill:
        _appearance = OLD_APPEARANCEOBJECT
    sjo = SpillJsonOutput()
    sjo.spills = [MockSpill()]

    cutoff_struct = sjo.generate_cutoff_struct()

    print(cutoff_struct)

    assert cutoff_struct == {}

def test_generate_cutoff_struct_full():
    """
    This should work properly with a fully populated spill appearance
    object
    """
    class MockSpill:
        _appearance = FULL_APPEARANCEOBJECT
    sjo = SpillJsonOutput()
    sjo.spills = [MockSpill()]

    cutoff_struct = sjo.generate_cutoff_struct()

    print(cutoff_struct)
    # result: {0: {'param': 'mass', 'cutoffs': [{'cutoff': 15.898729999999999, 'cutoff_id': 0, 'color': '#000000', 'label': ''}]}}
    # {'param': 'mass',
    #  'cutoffs': [{'cutoff': 15.898729999999999, 'cutoff_id': 0, 'color': '#000000', 'label': ''}]}

    assert len(cutoff_struct) == 1
    # not the best way to test, but it's something ...
    cs = cutoff_struct[0]
    assert cs == {'param': 'mass', 'cutoffs': [{'cutoff': 15.898729999999999, 'cutoff_id': 0, 'color': '#000000', 'label': ''}]}



# as of 5/8/2025
OLD_APPEARANCEOBJECT = SpillAppearance(**{
 # 'obj_type': 'gnome.utilities.appearance.SpillAppearance',
 # 'id': 'be34b3be-2a38-11f0-b23c-02420a00033f',
 'name': 'SpillAppearance_10',
 'colormap': 'Colormap_10.json',
 'pin_on': True,
 'les_on': True,
 'scale': 1,
 'data': 'Mass',
 'beached_color': '#000000',
 'units': 'kg',
 'preset_scales': [{'name': '12 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 43200],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [14400, 28800],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-4', '4-8', '8-12+']}},
  {'name': '24 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 86400],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [21600, 43200],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-6', '6-12', '12-24+']}},
  {'name': '48 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 172800],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [86400],
    'colorScaleRange': ['#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-24', '24-48']}},
  {'name': 'Bonn Agreement Appearance',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.25],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    'colorScaleRange': ['#1f77b4',
     '#2ca02c',
     '#bcbd22',
     '#ff7f0e',
     '#d62728',
     '#654321',
     '#000000'],
    'scheme': 'Bonn',
    'colorBlockLabels': ['Silver (<1)',
     'Rainbow (<5)',
     'Metallic (<10)',
     'Metallic (<25)',
     'Metallic (<50)',
     'Dark (<100)',
     'Dark (>100)']}},
  {'name': 'Response Relevant',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.25],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.05, 0.1],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['Light', 'Medium', 'Heavy']}},
  {'name': 'Biologically Relevant',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.05],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.001, 0.01],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['Light', 'Medium', 'Heavy']}},
  {'name': 'Response Relevant',
   'data': 'Viscosity',
   'units': 'cst',
   'colormap': {'units': 'cst',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.1],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.002, 0.005, 0.0075, 0.01, 0.015, 0.02],
    'colorScaleRange': ['#1f77b4',
     '#2ca02c',
     '#bcbd22',
     '#ff7f0e',
     '#9467bd',
     '#d62728',
     '#000000'],
    'scheme': 'Custom',
    'colorBlockLabels': ['', '', '', '', '', '', '', '']}}],
 'ctrl_names': {'pin_on': 'Spill Location',
  'les_on': 'Particles',
  'scale': 'Particle Size',
  'beached_color': 'Beached'},
 '_available_data': ['Mass', 'Surface Concentration', 'Age', 'Viscosity']}
 )

cmap = Colormap(**{
 # 'obj_type': 'gnome.utilities.appearance.Colormap',
 # 'id': '05448858-2a9b-11f0-8767-02420a0004bd',
 'name': 'Colormap_10',
 'units': 'kg',
 'alphaType': 'mass',
 'useAlpha': True,
 'map_type': 'discrete',
 'scheme': 'Custom',
 'endsConfigurable': 'none',
 'numberScaleType': 'linear',
 'numberScaleDomain': [0, 15.898729999999999],
 'numberScaleRange': [0, 1],
 'colorScaleType': 'threshold',
 'colorScaleDomain': [],
 'colorScaleRange': ['#000000'],
 'colorBlockLabels': [''],
 '_customScheme': ['#000000'],
 '_discreteSchemes': ['Custom',
  'Greys',
  'Reds',
  'Blues',
  'Purples',
  'YlOrBr',
  'Dark2',
  'Bonn'],
 '_continuousSchemes': ['Viridis',
  'Inferno',
  'Magma',
  'Plasma',
  'Warm',
  'Cool']})


# # as of 5/8/2025
FULL_APPEARANCEOBJECT =SpillAppearance(**{
# 'obj_type': 'gnome.utilities.appearance.SpillAppearance',
#  'id': '0544c0ac-2a9b-11f0-8767-02420a0004bd',
 'name': 'SpillAppearance_10',
 'colormap': cmap,
 'pin_on': True,
 'les_on': True,
 'scale': 1,
 'data': 'Mass',
 'beached_color': '#000000',
 'units': 'kg',
 'preset_scales': [{'name': '12 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 43200],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [14400, 28800],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-4', '4-8', '8-12+']}},
  {'name': '24 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 86400],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [21600, 43200],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-6', '6-12', '12-24+']}},
  {'name': '48 Hour',
   'data': 'Age',
   'units': 'hrs',
   'colormap': {'units': 'hrs',
    'numberScaleType': 'linear',
    'numberScaleDomain': [0, 172800],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [86400],
    'colorScaleRange': ['#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['0-24', '24-48']}},
  {'name': 'Bonn Agreement Appearance',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.25],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    'colorScaleRange': ['#1f77b4',
     '#2ca02c',
     '#bcbd22',
     '#ff7f0e',
     '#d62728',
     '#654321',
     '#000000'],
    'scheme': 'Bonn',
    'colorBlockLabels': ['Silver (<1)',
     'Rainbow (<5)',
     'Metallic (<10)',
     'Metallic (<25)',
     'Metallic (<50)',
     'Dark (<100)',
     'Dark (>100)']}},
  {'name': 'Response Relevant',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.25],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.05, 0.1],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['Light', 'Medium', 'Heavy']}},
  {'name': 'Biologically Relevant',
   'data': 'Surface Concentration',
   'units': 'g/m^2',
   'colormap': {'units': 'g/m^2',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.05],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.001, 0.01],
    'colorScaleRange': ['#fdbea0', '#fb6b40', '#b5211c'],
    'scheme': 'Reds',
    'colorBlockLabels': ['Light', 'Medium', 'Heavy']}},
  {'name': 'Response Relevant',
   'data': 'Viscosity',
   'units': 'cst',
   'colormap': {'units': 'cst',
    'numberScaleType': 'log',
    'numberScaleDomain': [0.0001, 0.1],
    'numberScaleRange': [0, 1],
    'colorScaleType': 'threshold',
    'colorScaleDomain': [0.002, 0.005, 0.0075, 0.01, 0.015, 0.02],
    'colorScaleRange': ['#1f77b4',
     '#2ca02c',
     '#bcbd22',
     '#ff7f0e',
     '#9467bd',
     '#d62728',
     '#000000'],
    'scheme': 'Custom',
    'colorBlockLabels': ['', '', '', '', '', '', '', '']}}],
 'ctrl_names': {'pin_on': 'Spill Location',
  'les_on': 'Particles',
  'scale': 'Particle Size',
  'beached_color': 'Beached'},
 '_available_data': ['Mass', 'Surface Concentration', 'Age', 'Viscosity'],
 'contours_on': False,
 'include_certain_contours': True}
 )



# @pytest.mark.parametrize(("json_"), ['save', 'webapi'])
# # @pytest.mark.parametrize(("json_"), ['webapi']) # only used for web api fo
# def test_serialize_deserialize(json_, output_filename):
#     '''
#     todo: this behaves in unexpected ways when using the 'model' testfixture.
#     For now, define a model in here for the testing - not sure where the
#     problem lies
#     '''

#     # create an outputter:

#     sjo = SpillJsonOutput(_additional_data=None,
#                           # **kwargs
#                           )

#     serial = sjo.serialize(json_)

#     print serial
#     # dict_ = o_put.deserialize(o_put.serialize(json_))

#     # jso2 = NetCDFOutput.new_from_dict(dict_)

#     assert False


