import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ba_tools import enrich

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'


def test_enrich_all():
    enrich_df = enrich.enrich_all(block_groups)
    assert isinstance(enrich_df, pd.DataFrame)
