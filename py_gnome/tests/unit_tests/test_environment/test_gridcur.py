"""
tests for code that reads teh old gridcur format
"""

from pathlib import Path

import pytest
from datetime import datetime

from gnome.environment.gridcur import read_file

test_data_dir = Path(__file__).parent / "sample_data"


def test_read_single_cell():
    data_type, times, data_u, data_v = read_file(test_data_dir / "grid_cur_1x1.cur")

    assert data_type == 'currents'
    assert times == [datetime(2002, 1, 30, 1, 0)]  # 30 1 2002 1 0

