import os
from pathlib import Path
import sys

from arcgis import GeoAccessor
from arcgis.geometry import Geometry
import numpy as np
import pandas as pd
import pytest

# facilitate local resource imports
project_dir_str = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
src_dir_str = project_dir_str/'src'
sys.path.insert(0, str(src_dir_str))
from ba_tools import analysis

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'

brand_locs = gdb/'locations_brand'
brand_id_field = 'LOCNUM'

comp_locs = gdb/'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir/'origin_demographics.csv'
nearest_brand_csv = data_dir/'nearest_locations.csv'

scratch_dir = data_dir/'scratch'

if not scratch_dir.exists():
    scratch_dir.mkdir()

@pytest.fixture
def near_df():
    nearest_df = pd.read_csv(str(data_dir/'nearest_locations.csv'))
    return nearest_df

@pytest.fixture
def new_dest():
    geom = Geometry({'x': -122.7342423, 'y': 45.4383307, 'spatialReference': {'latestWkid': 4326, 'wkid': 4326}})
    return geom


def test_get_add_new_closest_dataframe(near_df, new_dest):
    updated_nearest_df = analysis.get_add_new_closest_dataframe(
        origins=block_groups,
        origin_id_field=block_group_id_field,
        destinations=brand_locs,
        destination_id_field=brand_id_field,
        closest_table=near_df,
        new_destination=new_dest
    )
    # 1 is the new id assigned to the added destination, hence a decent test of functioning
    assert('1' not in near_df.destination_id_01.values and '1' in updated_nearest_df.destination_id_01.values)
