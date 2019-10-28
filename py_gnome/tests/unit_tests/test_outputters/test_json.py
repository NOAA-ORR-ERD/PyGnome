#!/usr/bin/env python2

"""
tests for teh json outputter

VERY incomplete!
"""

import pytest

from gnome.outputters.json import SpillJsonOutput


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


