import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ba_tools import utils

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'

locations = gdb/'locations_competition'
locations_id_fld = 'LOCNUM'
locations_concept_field = 'CONAME'


def test_count_by_polygon():
    count_df = utils.count_by_polygon(
        input_point_features=locations,
        point_sum_fields=[locations_id_fld, locations_concept_field],
        input_grouping_polygons=block_groups,
        polygon_id_field=block_group_id_field
    )
    assert all(count_df.columns == ['ID', 'LOCNUM', 'CONAME', 'count'])