"""
Author: Keyan Nasseri
Institution: Berkeley Institute for Data Science
Date: Spring 2020
Website: github.com/keyan3
"""

from collections import defaultdict
from typing import List, Set
import os

import pandas as pd
from utils import (
    clean_phone, add_slug, gen_company_mapping, get_last_appearance_field_value, add_license_field, get_csvs_in_dir,
    convert_str_to_datetime_maker, get_license_df, get_index_of_date, append_tag_conditional, get_slugs_up_to_i_array,
    get_column_difference
)

TAG_CHANGE_FIELDS = ['address', 'dispensary_name', 'email']

LICENSE_INPUT_DIR = 'input/license/'
PANEL_INPUT_DIR = 'input/panel/'
DISP_SCRAPE_INPUT_DIR = 'input/scrapes/'


def clean_files(files: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Clean phone and dispensary name, and add slug
    Tags: slug
    """
    for i in range(file_count):
        files[i] = add_slug(files[i])
        if 'phone' in files[i].columns:
            files[i] = clean_phone(files[i])
        if 'dispensary_name' in files[i].columns:
            files[i]['dispensary_name'] = files[i]['dispensary_name'].str.lower()
            files[i]['dispensary_name'] = files[i]['dispensary_name'].str.strip()
    return files


def add_license_info() -> None:
    """
    Add license information for panels that have it
    Tags: license, license_type, status, status_date, issue_date, adult_use/medicinal
    """
    license_df = get_license_df(LICENSE_INPUT_DIR)
    add_license_field(DISP_SCRAPE_INPUT_DIR, files, filenames)

    for i in range(file_count):
        if 'license' in files[i].columns:
            files[i] = pd.merge(files[i], license_df, on='license', how='left')


def tag_license_status() -> None:
    """
    For licenses that matched with CA license registry, create tags 'active_license', 'canceled_license', etc. This
    requires comparing the date the license was issued to the scrape date.
    If the scrape date is before the issue date, the license info may be inaccurate, so we record 'future license'.
    For licenses that did not match with CA license registry, create 'possible_license' tag.

    Tags: active_license, canceled_license, expired_license, revoked_license, suspended_license, possible_license,
    future_license_explicit
    """
    for i in range(file_count):
        if 'license' in files[i].columns:
            file = files[i]
            for license_status in ['active', 'canceled', 'expired', 'revoked', 'suspended']:
                tag_name = license_status + '_license'
                if license_status in file['status'].unique():
                    file[tag_name] = (
                        file['license'].notnull() &
                        file['issue_date'].notnull() &
                        (
                            file['issue_date'].apply(convert_str_to_datetime_maker('%m/%d/%Y'))
                            <= file['access_date'].apply(convert_str_to_datetime_maker('%Y-%m-%d'))
                        ) &
                        file['status'].str.contains(license_status)
                    ).apply(int)

            file['possible_license'] = (
                file['license'].notnull() &
                file['issue_date'].isna()
            ).apply(int)

            file['future_license_explicit'] = (
                file['license'].notnull() &
                file['issue_date'].notnull() &
                (
                    file['issue_date'].apply(convert_str_to_datetime_maker('%m/%d/%Y'))
                    > file['access_date'].apply(convert_str_to_datetime_maker('%Y-%m-%d'))
                )
            ).apply(int)


def tag_assumed_license_status() -> None:
    """
    Since not all panels have license information, we back-propagate license information from later panels to earlier
    panels with these tags. In particular, if a dispensary eventually displays a verified active license, we tag its
    earlier appearances with "assumed license".

    Tags: assumed_license
    """
    active_license_slugs_after_i = [set()]
    for i in reversed(range(0, file_count - 1)):
        file_after_i = files[i + 1]
        if 'license' in file_after_i.columns:
            file_after_i_active_slugs = set(file_after_i['slug'][file_after_i['active_license'] == 1])
            active_license_slugs_after_i.insert(0, file_after_i_active_slugs.union(active_license_slugs_after_i[0]))
        else:
            active_license_slugs_after_i.insert(0, active_license_slugs_after_i[0])

    active_license_slugs_at_i = (
        list(map(
            lambda x: set(x['slug'][x['active_license'] == 1]) if 'license' in x.columns else {},
            files
        ))
    )

    for i in range(file_count):
        files[i]['assumed_license'] = (
            files[i]['slug'].isin(active_license_slugs_after_i[i]) &
            ~files[i]['slug'].isin(active_license_slugs_at_i[i])
        ).apply(int)


def tag_field_changes() -> None:
    """
    For each of the fields in TAG_CHANGE_FIELDS, tag storefronts that changed the field between consecutive panels 
    Tags: [changed_{column} for column in TAG_CHANGE_FIELDS] (e.g. changed address)
    """
    for i in range(1, file_count):
        curr_file = files[i]
        prev_file = files[i - 1]
        for field in TAG_CHANGE_FIELDS:
            if (field not in files[i].columns) or (field not in files[i - 1].columns):
                break

            prev_field, curr_field = prev_file[field], curr_file[field]
            prev_slug_column, curr_slug_column = prev_file['slug'], curr_file['slug']
            continued = curr_file['continued']

            prev_slug_to_prev_field_value = defaultdict(lambda: None)
            for j in range(len(prev_file)):
                prev_slug = prev_slug_column.iloc[j]
                prev_slug_to_prev_field_value[prev_slug] = prev_field.iloc[j]

            changed_field = []
            for j in range(len(curr_file)):
                prev_field_value = prev_slug_to_prev_field_value[curr_slug_column.iloc[j]]
                curr_field_value = curr_field.iloc[j]
                append_tag_conditional(changed_field, continued.iloc[j] == 1 and prev_field_value != curr_field_value)
            curr_file['changed_' + field] = changed_field


def tag_reappear_field_changes(slugs_at_i: List[Set[str]]) -> None:
    """
    For each of the fields in TAG_CHANGE_FIELDS, tag storefronts that disappeared and reappeared in a later wave with
    the field changed (could be anywhere from 2 to 9 panels apart). Overwrites previous tags from tag_field_changes,
    which only considers changes between consecutive panels.

    Tags: [changed_{column} for column in TAG_CHANGE_FIELDS] (e.g. changed address)
    """
    for i in range(2, file_count):
        curr_file = files[i]
        for field in TAG_CHANGE_FIELDS:
            if not all([field in file.columns for file in files[0: i + 1]]):
                break
            reappeared = curr_file['reappeared']
            curr_field = curr_file[field]
            curr_slug_column = curr_file['slug']
            old_changed_column = curr_file['changed_' + field]

            changed_field = []
            for j in range(len(curr_file)):
                if reappeared.iloc[j] == 1:
                    prev_field_value = get_last_appearance_field_value(slug=curr_slug_column.iloc[j], max_wave=i-2,
                                                                       field=field, slugs_at_i=slugs_at_i, files=files)
                    curr_field_value = curr_field.iloc[j]
                    append_tag_conditional(changed_field, prev_field_value != curr_field_value)
                else:
                    changed_field.append(old_changed_column.iloc[j])
            curr_file['changed_' + field] = changed_field


def tag_continued() -> None:
    """
    For each panel after the first, record whether a storefront appeared in the previous panel
    Tags: continued
    """
    for i in range(1, file_count):
        curr_file = files[i]
        prev_file = files[i - 1]
        prev_slug_set = set(prev_file['slug'])
        curr_file['continued'] = files[i]['slug'].isin(prev_slug_set).apply(int)


def tag_disappear() -> None:
    """
    Add a column for storefronts which disappear after a given wave
    Tags: disappeared
    """
    for i in range(0, file_count - 1):
        curr_file = files[i]
        next_file = files[i + 1]
        next_slug_set = set(next_file['slug'])
        curr_file['disappeared'] = (~files[i]['slug'].isin(next_slug_set)).apply(int)


def tag_reappear(slugs_up_to_i: List[Set[str]]) -> None:
    """
    Add a reappear column for storefronts which reappear in a given wave after disappearing
    Tags: reappear
    """
    for i in range(2, file_count):
        curr_file = files[i]
        prev_file = files[i - 1]
        prev_slug_set = set(prev_file['slug'])
        curr_file['reappeared'] = (files[i]['slug'].isin(slugs_up_to_i[i - 2]) &
                                   ~files[i]['slug'].isin(prev_slug_set)).apply(int)


def tag_illegal_storefronts(slugs_at_i: List[Set[str]], slugs_up_to_i: List[Set[str]]) -> None:
    """
    Weedmaps purged many illegal storefronts between 12/15/19 and 01/12/20; we create two tags to capture this.
    First, we tag storefronts present on 12/15/19 but not present on 01/12/20 as 'illegal_1912', and storefronts
    present before 12/15/19 but not present on 01/12/20 as 'illegal_other.

    Tags: illegal_1912, illegal_other
    """
    dates = list(map(lambda x: x[:6], filenames))
    if ('191221' not in dates) or ('200112' not in dates):
        return

    index_191221 = get_index_of_date('191221', filenames)
    index_200112 = get_index_of_date('200112', filenames)
    in_191221_not_200112 = slugs_at_i[index_191221].difference(slugs_at_i[index_200112])
    before_191221_not_200112 = slugs_up_to_i[index_191221 - 1].difference(slugs_at_i[index_200112])
    before_191221_not_200112_or_191221 = before_191221_not_200112.difference(slugs_at_i[index_191221])

    for i in range(file_count):
        if '200112' not in filenames[i]:
            files[i]['illegal_1912'] = files[i]['slug'].isin(in_191221_not_200112).apply(int)
            files[i]['illegal_other'] = files[i]['slug'].isin(before_191221_not_200112_or_191221).apply(int)


def tag_dispensary_or_delivery() -> None:
    """
    Add columns tagging whether dispensary or delivery
    Tags: is_dispensary, is_delivery
    """
    for i in range(file_count):
        files[i]['is_dispensary'] = files[i]['url'].str.contains("weedmaps.com/dispensaries/").apply(int)
        files[i]['is_delivery'] = files[i]['url'].str.contains("weedmaps.com/deliveries/").apply(int)


def standardize_files(full_files: List[pd.DataFrame]) -> List[str]:
    """
    Make sure all output files have the same tags, even if some are empty. Returns this maximal group of tags.
    """
    max_tag_num = -1
    for i in range(file_count):
        file_tags = get_column_difference(files[i], full_files[i])
        if len(file_tags) > max_tag_num:
            max_tag_num = len(file_tags)
            max_file_tag_set = file_tags

    for i in range(file_count):
        for field in max_file_tag_set:
            if field not in files[i].columns:
                files[i][field] = [''] * len(files[i])

    return max_file_tag_set


def main() -> None:
    global files, filenames, file_count
    full_files, filenames = get_csvs_in_dir(PANEL_INPUT_DIR, return_filenames=True)
    file_count = len(full_files)

    for file in full_files:
        assert 'url' in file.columns, 'Panel files must include url field'

    # Group files by slug to remove unneeded product info (shrinks files to ~2% of full size)
    full_files = clean_files(full_files)
    files = [file.groupby('slug').agg('first').reset_index() for file in full_files]

    if (
            os.path.exists(LICENSE_INPUT_DIR) and len(get_csvs_in_dir(LICENSE_INPUT_DIR)) > 0
            and os.path.exists(DISP_SCRAPE_INPUT_DIR) and len(get_csvs_in_dir(DISP_SCRAPE_INPUT_DIR)) > 0
    ):
        add_license_info()
        tag_license_status()
        tag_assumed_license_status()

    tag_continued()

    tag_field_changes()

    tag_disappear()

    slugs_up_to_i = get_slugs_up_to_i_array(files)
    slugs_at_i = list(map(lambda x: set(x['slug']), files))

    tag_reappear(slugs_up_to_i)

    tag_reappear_field_changes(slugs_at_i)

    tag_illegal_storefronts(slugs_at_i, slugs_up_to_i)

    tag_dispensary_or_delivery()

    tag_list = standardize_files(full_files)

    # Add tags to original, product-containing files
    for i in range(file_count):
        tags_only_file = files[i][['slug'] + tag_list]
        files[i] = pd.merge(full_files[i], tags_only_file, on='slug')

    # Write tagged panels to output directory
    for i in range(file_count):
        files[i].to_csv('output/panel/' + filenames[i][:6] + '_tagged.csv')

    # Write company mappings to output directory
    for i in range(file_count):
        gen_company_mapping(files[i], 'output/company/' + filenames[i][:6] + '_company_mapping.csv')


if __name__ == "__main__":
    main()
