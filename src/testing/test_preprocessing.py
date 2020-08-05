import os
from pathlib import Path
import sys

from arcgis import GeoAccessor
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ba_tools import preprocessing, data, utils

data_dir = Path(__file__).parent/'test_data'
gdb = data_dir/'test_data.gdb'

block_groups = gdb/'block_groups'
block_group_id_field = 'ID'
brand_locs = gdb/'locations_brand'
brand_id_field = 'LOCNUM'
comp_locs = gdb/'locations_competition'
comp_id_field = 'LOCNUM'

origin_demographics_csv = data_dir/'origin_demographics.csv'
scratch_dir = data_dir/'scratch'

if not scratch_dir.exists():
    scratch_dir.mkdir()

cols_df = pd.DataFrame(columns=['test_FY', 'destination_id_01', 'run_fast', 'run_slow', 'walk_fast', 'walk_slow'])


def test_origins_to_dataframe():
    origins_to_dataframe = preprocessing.OriginGeographyFeatureClassToDataframe(
        geography_id_field=block_group_id_field
    )
    origin_df = origins_to_dataframe.fit_transform(block_groups)
    assert isinstance(origin_df, pd.DataFrame)


def test_add_demographics():
    add_demographics_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddDemographicsToOriginDataframe(
            origin_geography_layer=block_groups,
            geography_id_field=block_group_id_field,
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=True
        ))
    ])
    df = add_demographics_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'origin_demographics.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())


def test_add_demograhpics_tapestry_one_hot():
    add_demographics_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddDemographicsToOriginDataframe(
            origin_geography_layer=block_groups,
            geography_id_field=block_group_id_field,
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=True
        ))
    ])
    df = add_demographics_pipe.fit_transform(block_groups)

    assert 'tapestryhouseholdsNEW_TSEGCODE_6C' in df.columns


def test_add_selected_demographics():
    add_demographics_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddSelectedDemographicsToOriginDataframe(
            origin_geography_layer=block_groups,
            geography_id_field=block_group_id_field,
            enrich_variable_list=data.enrich_vars_dataframe.sample(10)['enrich_str'],
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=True
        ))
    ])
    df = add_demographics_pipe.fit_transform(block_groups)

    assert len(df.columns) == 10


def test_add_tapestry_demograhpics():
    add_demographics_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddTapestryDemographicsToOriginDataframe(
            origin_geography_layer=block_groups,
            geography_id_field=block_group_id_field,
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=True
        ))
    ])
    df = add_demographics_pipe.fit_transform(block_groups)

    assert 'tapestryhouseholdsNEW_TSEGCODE_6C' in df.columns


def test_add_nearest_locations():
    add_nearest_loc_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddNearestLocationsToOriginDataframe(
            origin_geography_layer=block_groups,
            origin_id_field=block_group_id_field,
            location_layer=brand_locs,
            location_id_field=brand_id_field,
            destination_count=6,
            interim_data_directory=scratch_dir,
            clobbber_previous_results=False
        ))
    ])
    df = add_nearest_loc_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'nearest_locations.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())


def test_add_nearest_competition_locations():
    add_nearest_comp_loc_pipe = Pipeline([
        ('get_origin_df', preprocessing.OriginGeographyFeatureClassToDataframe(block_group_id_field)),
        ('add_demographics', preprocessing.AddNearestCompetitionLocationsToOriginDataframe(
            origin_geography_layer=block_groups,
            origin_id_field=block_group_id_field,
            competition_location_layer=comp_locs,
            competition_location_id_field=comp_id_field,
            destination_count=6,
            interim_data_directory=scratch_dir,
            rebuild_if_output_exists=False
        ))
    ])
    df = add_nearest_comp_loc_pipe.fit_transform(block_groups)

    out_path = scratch_dir/'nearest_locations.csv'
    assert (isinstance(df, pd.DataFrame) and out_path.exists())


def test_standard_scaler():
    test_df = pd.DataFrame()
    test_df['factor_01'] = np.random.randint(1, 100, 50)
    test_df['factor_02'] = np.random.randint(1, 100, 50)
    test_df['label'] = np.random.randint(1, 10, 50)
    test_df['origin_id'] = test_df.index
    trans = preprocessing.StandardScaler('label')
    out_df = trans.fit_transform(test_df)
    assert(isinstance(out_df, pd.DataFrame))


def test_exclude_by_name():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeColumnsByName('run_fast')
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count - 1)


def test_exclude_startswith():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeColumnsByStartswith('run')
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count-2)


def test_exclude_endswith():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeColumnsByEndswith('slow')
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count-2)


def test_exclude_contains():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeColumnsByContains(['run', 'walk'])
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count-4)


def test_exclude_dest_id():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeDestinationIdColumns()
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count-1)


def test_exclude_fy():
    start_count = len(cols_df.columns)
    trans = preprocessing.ExcludeFutureYearColumns()
    out_df = trans.fit_transform(cols_df)
    out_count = len(out_df.columns)
    assert(out_count == start_count-1)


def test_generate_hexbins():
    output_hexbins = 'memory/test_hexbins'
    hex_pipe = Pipeline([
        ('generate_hexbins', preprocessing.GenerateHexbins(output_hexbins, block_groups))
    ])
    hex_pipe.fit_transform(None)
    hex_df = GeoAccessor.from_featureclass(output_hexbins)
    assert(len(hex_df.index))


@pytest.fixture
def hexbins():
    return str(utils.get_hexbins('memory/testing_hexbins', 3000, block_groups))


def test_flag_accessible(hexbins):
    flag_pipe = Pipeline([
        ('flag_hex', preprocessing.FlagAccessibility())
    ])
    flag_pipe.fit_transform(hexbins)
    hex_df = GeoAccessor.from_featureclass(flag_pipe)
    assert(hex_df.use_for_analysis.isnull().sum() == 0)