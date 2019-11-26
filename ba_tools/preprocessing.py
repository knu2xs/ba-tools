from arcgis.features import GeoAccessor
import arcpy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import data
from .enrich import enrich_all, enrich
from . import proximity
from . import utils


def _join_result_to_X(X, add_df):
    add_df.index = add_df.index.astype('int64')
    if 'origin_id' in X.columns:
        X['origin_id'] = X['origin_id'].astype('int64')
        joined_df = X.join(add_df, on='origin_id')
    else:
        joined_df = X.join(add_df)
    if joined_df.index.name != 'origin_id':
        joined_df.set_index('origin_id', drop=True, inplace=True)
    return joined_df


class _BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer class to save putting the fit method into all the transformers below.
    """
    def fit(self, X, y=None):
        return self


class OriginGeographyFeatureClassToDataframe(_BaseTransformer):
    """
    Use as the starting point to begin building up a dataframe for analysis. This creates a very sparse dataframe with
    one column, object_id, for subsequent use in the analysis pipeline.
    """
    def __init__(self, geography_id_field):
        self.id_fld = geography_id_field

    def transform(self, X, y=None):
        geo_path = str(X)

        geo_id_lst = [r for r in arcpy.da.SearchCursor(geo_path, self.id_fld)]
        geo_df = pd.DataFrame(geo_id_lst, columns=['origin_id'])
        geo_df['origin_id'] = geo_df['origin_id'].astype('int64')

        return geo_df


class AddDemographicsToOriginDataframe(_BaseTransformer):
    """
    Enrich the origin geographies with all available demographics INCLUDING all tapestry factors.
    :param origin_geography_layer: String path to feature Layer with all potential contributing geographies to be
        included in the analysis.
    :param geography_id_field: String column name for column containing integer values uniquely identifying each
        geography.
    :param interim_data_directory: Directory where intermediate results will be stored for subsequent analysis.
    :param tapestry_one_hot_encoding: Optional boolean indicating whether the dominant tapestry (if available) will be
        one hot encoded to be ready for downstream machine learning modeling. Default is True.
    :param rebuild_if_output_exists: Optional boolean indicating if the output should be rebuild if it already exists in
        the interim directory. Default is False
    """
    def __init__(self, origin_geography_layer, geography_id_field, interim_data_directory,
                 tapestry_one_hot_encoding=True, rebuild_if_output_exists=False):
        self.origin_lyr = str(origin_geography_layer)
        self.geography_id_fld = geography_id_field
        self.interim_dir = utils.ensure_path(interim_data_directory)
        self.tap_one_hot = tapestry_one_hot_encoding
        self.rebuild = rebuild_if_output_exists

    def transform(self, X, y=None):
        enrich_csv = self.interim_dir/'origin_demographics.csv'

        if not enrich_csv.exists() or self.rebuild:

            # enrich with all available variables
            enrich_df = enrich_all(self.origin_lyr, id_field=self.geography_id_fld)

            # since there are two columns for the dominant tapestry segment, drop the numeric one for simplicity's sake
            if 'tapestryhouseholdsNEW_TSEGNUM' in enrich_df.columns:
                enrich_df.drop(columns='tapestryhouseholdsNEW_TSEGNUM', inplace=True)

            # if one hot encoding for tapestry, grip it and rip it!
            if self.tap_one_hot and 'tapestryhouseholdsNEW_TSEGCODE' in enrich_df.columns:
                enrich_df = pd.get_dummies(enrich_df, columns=['tapestryhouseholdsNEW_TSEGCODE'])

            # rename the index for consistency
            enrich_df.index.name = 'origin_id'

            enrich_df.to_csv(enrich_csv)
        else:
            enrich_df = pd.read_csv(enrich_csv, index_col=0)

        # join the result to the input, and pass downstream
        enrich_joined_df = _join_result_to_X(X, enrich_df)

        return enrich_joined_df


class AddSelectedDemographicsToOriginDataframe(_BaseTransformer):
    """
    Enrich origin geographies with selected available demographics.
    :param origin_geography_layer: String path to feature Layer with all potential contributing geographies to be
        included in the analysis.
    :param geography_id_field: String column name for column containing integer values uniquely identifying each
        geography.
    :param enrich_variable_list: List of string variable names for enrichment. These can be looked up using
        ba_tools.data.enrich_vars_dataframe. They must match values in the enrich_str column.
    :param interim_data_directory: Directory where intermediate results will be stored for subsequent analysis.
    :param tapestry_one_hot_encoding: Optional boolean indicating whether the dominant tapestry (if available) will be
        one hot encoded to be ready for downstream machine learning modeling. Default is True.
    :param rebuild_if_output_exists: Optional boolean indicating if the output should be rebuild if it already exists in
        the interim directory. Default is False
    """
    def __init__(self, origin_geography_layer, geography_id_field, enrich_variable_list, interim_data_directory,
                 tapestry_one_hot_encoding=True, rebuild_if_output_exists=False):
        self.origin_lyr = str(origin_geography_layer)
        self.geography_id_fld = geography_id_field
        self.enrich_vars = enrich_variable_list
        self.interim_dir = utils.ensure_path(interim_data_directory)
        self.tap_one_hot = tapestry_one_hot_encoding
        self.rebuild = rebuild_if_output_exists

    def transform(self, X, y=None):
        enrich_csv = self.interim_dir/'origin_demographics.csv'

        if not enrich_csv.exists() or self.rebuild:

            # enrich with all available variables
            enrich_df = enrich(self.origin_lyr, self.enrich_vars, id_field=self.geography_id_fld)

            # since there are two columns for the dominant tapestry segment, drop the numeric one for simplicity's sake
            if 'tapestryhouseholdsNEW_TSEGNUM' in enrich_df.columns:
                enrich_df.drop(columns='tapestryhouseholdsNEW_TSEGNUM', inplace=True)

            # if one hot encoding for tapestry, grip it and rip it!
            if self.tap_one_hot and 'tapestryhouseholdsNEW_TSEGCODE' in enrich_df.columns:
                enrich_df = pd.get_dummies(enrich_df, columns=['tapestryhouseholdsNEW_TSEGCODE'])

            # rename the index for consistency
            enrich_df.index.name = 'origin_id'

            enrich_df.to_csv(enrich_csv)
        else:
            enrich_df = pd.read_csv(enrich_csv, index_col=0)

        # join the result to the input, and pass downstream
        enrich_joined_df = _join_result_to_X(X, enrich_df)

        return enrich_joined_df


class AddNearestLocationsToOriginDataframe(_BaseTransformer):
    """
    Add the nearest nth locations to the origin geographies.
    """
    def __init__(self, origin_geography_layer, origin_id_field, location_layer, location_id_field,
                 destination_count, interim_data_directory, clobbber_previous_results=False):
        self.origin_lyr = origin_geography_layer
        self.origin_id_fld = origin_id_field
        self.location_lyr = location_layer
        self.location_id_fld = location_id_field
        self.destination_cnt = destination_count
        self.interim_dir = utils.ensure_path(interim_data_directory)
        self.clobber = clobbber_previous_results

    def transform(self, X, y=None):
        nearest_csv = self.interim_dir/'nearest_locations.csv'

        if not nearest_csv.exists() or self.clobber:
            nearest_df = proximity.closest_dataframe_from_origins_destinations(self.origin_lyr, self.origin_id_fld,
                                                                               self.location_lyr, self.location_id_fld,
                                                                               network_dataset=data.usa_network_dataset,
                                                                               destination_count=self.destination_cnt)
            nearest_df.set_index('origin_id', drop=True, inplace=True)

            nearest_df.to_csv(nearest_csv)
        else:
            nearest_df = pd.read_csv(nearest_csv, index_col=0)

        nearest_joined_df = _join_result_to_X(X, nearest_df)

        return nearest_joined_df


class AddNearestCompetitionLocationsToOriginDataframe(_BaseTransformer):
    """
    Add the nearest nth locations to the origin geographies.
    """

    def __init__(self, origin_geography_layer, origin_id_field, competition_location_layer,
                 competition_location_id_field, destination_count, interim_data_directory,
                 rebuild_if_output_exists=False):
        self.origin_lyr = origin_geography_layer
        self.origin_id_fld = origin_id_field
        self.location_lyr = competition_location_layer
        self.location_id_fld = competition_location_id_field
        self.destination_cnt = destination_count
        self.interim_dir = utils.ensure_path(interim_data_directory)
        self.rebuild = rebuild_if_output_exists

    def transform(self, X, y=None):
        nearest_csv = self.interim_dir / 'nearest_competition_locations.csv'

        if not nearest_csv.exists() or self.rebuild:
            nearest_df = proximity.closest_dataframe_from_origins_destinations(self.origin_lyr, self.origin_id_fld,
                                                                               self.location_lyr, self.location_id_fld,
                                                                               network_dataset=data.usa_network_dataset,
                                                                               destination_count=self.destination_cnt)
            nearest_df.set_index('origin_id', drop=True, inplace=True)

            nearest_df.columns = [c.replace('proximity', 'proximity_competition') for c in nearest_df.columns]
            nearest_df.columns = [c.replace('destination', 'destination_competition') for c in nearest_df.columns]

            nearest_df.to_csv(nearest_csv)
        else:
            nearest_df = pd.read_csv(nearest_csv, index_col=0)

        nearest_joined_df = _join_result_to_X(X, nearest_df)

        return nearest_joined_df


class ExcludeColumnsByStartswith(_BaseTransformer):
    """
    Exclude DataFrame columns based on the string pattern the column starts with.
    :param string_pattern: Either a single, or multiple string patterns as a list to exclude based on.
    """
    def __init__(self, string_pattern:[str, list]):
        if isinstance(string_pattern, str):
            self.str_pattern = [string_pattern]
        else:
            self.str_pattern = string_pattern

    def transform(self, X, y=None):
        keep_cols = [col for col in X.columns if not col.startswith(tuple(self.str_pattern))]
        return X[keep_cols].copy()


class ExcludeColumnsByEndswith(_BaseTransformer):
    """
    Exclude DataFrame columns based on the string pattern the column ends with.
    :param string_pattern: Either a single, or multiple string patterns as a list to exclude based on.
    """
    def __init__(self, string_pattern:[str, list]):
        if isinstance(string_pattern, str):
            self.str_pattern = [string_pattern]
        else:
            self.str_pattern = string_pattern

    def transform(self, X, y=None):
        keep_cols = [col for col in X.columns if not col.endswith(tuple(self.str_pattern))]
        return X[keep_cols].copy()


class ExcludeColumnsByContains(_BaseTransformer):
    """
    Exclude DataFrame columns based on the string pattern the column contains.
    :param string_pattern: Either a single, or multiple string patterns as a list to exclude based on.
    """

    def __init__(self, string_pattern: [str, list]):
        if isinstance(string_pattern, str):
            self.str_pattern = [string_pattern]
        else:
            self.str_pattern = string_pattern

    def transform(self, X, y=None):
        keep_cols = [col for col in X.columns if all([col.find(str_patt) < 0 for str_patt in self.str_pattern])]
        return X[keep_cols].copy()


class ExcludeFutureYearColumns(ExcludeColumnsByEndswith):
    """
    Exclude the future year columns many times included with enriched data.
    """
    def __init__(self):
        super().__init__('_FY')


class ExcludeDestinationIdColumns(ExcludeColumnsByStartswith):
    """
    Exclude the destination_id columns.
    """
    def __init__(self):
        super().__init__('destination_id')


class ExcludeStringColumns(_BaseTransformer):
    """
    Exclude String/Object columns.
    """
    def transform(self, X, y=None):
        str_cols = X.select_dtypes('object').columns
        keep_cols = [col for col in X.columns if not col in str_cols]
        return X[keep_cols].copy()
