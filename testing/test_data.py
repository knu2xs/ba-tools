import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ba_tools import data

gdb = Path(os.path.abspath('./test_data.gdb'))
block_groups = gdb/'block_groups'
brand_locs = gdb/'locations_brand'
comp_locs = gdb/'locations_competition'


def test_data_get_master_dataframe():
    df = data.get_master_dataframe(str(block_groups), 'ID', str(brand_locs), 'LOCNUM', str(comp_locs), 'LOCNUM')
    assert(isinstance(pd.DataFrame, df))
