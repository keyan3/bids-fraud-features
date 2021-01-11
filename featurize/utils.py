"""
Author: Keyan Nasseri
Institution: Berkeley Institute for Data Science
Date: Spring 2020
Website: github.com/keyan3
"""

from collections import defaultdict
from datetime import datetime
import os
from typing import List, Set, Tuple, Callable, Union, Any
import uuid

from disjoint_set import DisjointSet
import numpy as np
import pandas as pd
import stringcase

LICENSE_FIELDS = ['License Number', 'License Type', 'Status', 'Status Date', 'Issue Date', 'Adult-Use/Medicinal']


def country_code(n: str) -> str:
    """
    Helper function to add country code to a phone number if it does not have one already.
    NOTE: area codes in North America cannot start with a 0 or 1
    """
    if n[0] != '1':
        return '1' + n
    else:
        return n


def clean_phone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function cleans phone numbers in dataset by:
        1. Removing all numbers that do not start with a parenthesis or digit
        2. Removing all non-digit characters
        3. Standardizing to 1########## format
    """
    df_pclean = df[df['phone'].notnull()]
    df_pclean['phone'] = df_pclean['phone'].astype(str)
    df_pclean = df_pclean[df_pclean['phone'].str.contains('^([0-9]|\().*')]

    df_pclean['stdrd_phone'] = df_pclean['phone'].str.replace("[^0-9]", "")
    df_pclean = df_pclean[df_pclean['stdrd_phone'].apply(len) > 0]
    df_pclean['stdrd_phone'] = df_pclean['stdrd_phone'].apply(country_code)
    df_pclean['stdrd_phone'] = df_pclean['stdrd_phone'].str[:11]

    df_pclean = df_pclean[df_pclean['stdrd_phone'].str.contains('^.{11}$')]
    df_pclean = df_pclean[df_pclean['stdrd_phone'].str.contains('^((?!000000).)*$')]
    return df_pclean


def add_slug(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['url'].notnull()]
    df = df[df['url'].apply(type) == str]
    df = df[df['url'].str.contains('.*/.*')]
    df['slug'] = df['url'].apply(lambda x: x.split('/')[-1])
    return df


def gen_company_mapping(df: pd.DataFrame, path: str) -> None:
    """
    Function generates storefront-company mapping by:
        1. Assigning storefronts with same product offering to same company
        2. Assigning storefronts with same phone to same company
        3. Assigning storefronts with same email to same company
        4. Generating a UUID for each company
        5. Adding these UUIDs to the panel dataframe (grouped by storefront ID)
        6. Dropping all columns beside storefront and company ID
    A union-find data structure is used to keep track of the company assignments
    throughout the function.
    """
    # drop unneeded columns
    if 'phone' in df.columns:
        df_sim = df[['email', 'slug', 'phone', 'product_name']]
    else:
        df_sim = df[['email', 'slug', 'product_name']]

    # groupby name and agg columns appropriately for company grouping
    if 'phone' in df.columns:
        df_sim_2 = df_sim.groupby('slug').agg(
            {'email': 'first', 'phone': 'first', 'product_name': lambda x: frozenset(x)})
    else:
        df_sim_2 = df_sim.groupby('slug').agg({'email': 'first', 'product_name': lambda x: frozenset(x)})
    ds = DisjointSet()

    # add all storefront IDs to DJS
    for sid in df_sim_2.index:
        ds.find(sid)

    # setup hash tables for later unions
    product_ht = defaultdict(lambda: [])
    email_ht = defaultdict(lambda: [])
    phone_ht = defaultdict(lambda: [])

    for sid in df_sim_2.index:
        row = df_sim_2.loc[sid]
        if not row.isnull()['product_name']:
            prod_set = row['product_name']
        if not row.isnull()['email']:
            email = row['email']
        if 'phone' in df.columns:
            if not row.isnull()['phone']:
                phone = row['phone']
            phone_ht[phone].append(sid)
        product_ht[prod_set].append(sid)
        email_ht[email].append(sid)

    # union storefront IDs with same product offering
    for key in product_ht:
        first_sid = product_ht[key][0]
        for curr_sid in product_ht[key][1:]:
            ds.union(first_sid, curr_sid)

    # union storefront IDs with same email (check NaN)
    for key in email_ht:
        first_sid = email_ht[key][0]
        for curr_sid in email_ht[key][1:]:
            ds.union(first_sid, curr_sid)

    # union storefront IDs with same phone
    if 'phone' in df.columns:
        for key in phone_ht:
            first_sid = phone_ht[key][0]
            for curr_sid in phone_ht[key][1:]:
                ds.union(first_sid, curr_sid)

    # add company_id column to df
    df_sim_2['company_id'] = np.zeros(df_sim_2.index.shape)

    for comp in ds.itersets():
        comp_id = str(uuid.uuid1())
        for sid in comp:
            df_sim_2.loc[sid, 'company_id'] = comp_id

    mapping = df_sim_2[['company_id']]

    mapping.to_csv(path, line_terminator='\n')


def get_last_appearance_field_value(slug: str, max_wave: int, field: str, slugs_at_i: List[Set[str]],
                                    files: List[pd.DataFrame]) -> Any:
    """
    Finds the address, email, and name of that storefront at the latest time it appeared in waves 0 to max_wave
    """
    for i in reversed(range(0, max_wave + 1)):
        if slug in slugs_at_i[i]:
            break

    last_slug_subset = files[i][files[i]['slug'] == slug]
    last_appearance_field_value = last_slug_subset[field].iloc[0]

    return last_appearance_field_value


def add_license_field(path: str, files: List[pd.DataFrame], filenames: List[str]) -> None:
    scrape_files, scrape_filenames = get_csvs_in_dir(path, return_filenames=True)

    scrape_dates_with_licenses = []
    scrape_files_with_licenses = []

    for i in range(len(scrape_files)):
        scrape_files[i] = add_slug(scrape_files[i])
        if 'state_license_number_1' in scrape_files[i].columns:
            scrape_files[i] = scrape_files[i].rename({'state_license_number_1': 'license'}, axis=1)
            scrape_files[i]['license'] = scrape_files[i]['license'].str.strip()
            scrape_files[i]['license'] = scrape_files[i]['license'].str.lower()
            scrape_files[i]['license'] = scrape_files[i]['license'].apply(convert_nonnan_nans)

            scrape_dates_with_licenses.append(scrape_filenames[i][:6])
            scrape_files_with_licenses.append(scrape_files[i])

    for i, date_str in enumerate(scrape_dates_with_licenses):
        for j, filename in enumerate(filenames):
            if date_str in filename:
                matching_scrape = scrape_files_with_licenses[i][['slug', 'license']]
                files[j] = pd.merge(files[j], matching_scrape, on='slug')


def get_csvs_in_dir(dir: str, return_filenames: bool = False,
                    sort: bool = True) -> Union[List[pd.DataFrame], Tuple[List[pd.DataFrame], List[str]]]:
    filenames = os.listdir(dir)
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    if sort:
        filenames = sorted(filenames)
    files = []
    for filename in filenames:
        files.append(pd.read_csv(dir + filename, lineterminator='\n'))
    if return_filenames:
        return files, filenames
    return files


def convert_str_to_datetime_maker(formt: str) -> Callable:
    def convert(string):
        if type(string) is float:
            return string
        else:
            return datetime.strptime(string, formt)
    return convert


def get_license_df(path: str) -> Union[pd.DataFrame, None]:
    license_df = pd.concat(get_csvs_in_dir(path))[LICENSE_FIELDS]
    assert 'License Number' in license_df.columns, 'License files must include license number field'

    license_df = clean_column_names(license_df)
    license_df = license_df.rename({'license_number': 'license', 'license_type': 'license_business_type'}, axis=1)
    for column in license_df.columns:
        if type(license_df[column].iloc[0]) is str:
            license_df[column] = license_df[column].str.lower()

    if 'issue_date' in license_df.columns:
        license_df = license_df[license_df['issue_date'].str.contains('/')]
    return license_df


def convert_nonnan_nans(entry: str) -> Union[None, str]:
    if entry == 'n/a' or entry == 'na':
        return None
    else:
        return entry


def get_index_of_date(date: str, filenames: List[str]) -> int:
    assert len(date) == 6
    idx = 0
    for filename in filenames:
        if date in filename:
            return idx
        idx += 1


def append_tag_conditional(lst: List[int], condition: int) -> None:
    if condition:
        lst.append(1)
    else:
        lst.append(0)


def get_slugs_up_to_i_array(files: List[pd.DataFrame]) -> List[Set[str]]:
    slugs_up_to_i = [set(files[0]['slug'])]
    for i in range(1, len(files)):
        slugs_up_to_i.append(
            set(files[i]['slug']).union(slugs_up_to_i[i - 1])
        )
    return slugs_up_to_i


def clean_column_names(file: pd.DataFrame) -> pd.DataFrame:
    lower_rename_dict = {column_name: column_name.lower() for column_name in file.columns}
    license_df = file.rename(lower_rename_dict, axis=1)
    snakecase_rename_dict = {column_name: stringcase.snakecase(column_name) for column_name in license_df.columns}
    return license_df.rename(snakecase_rename_dict, axis=1)


def get_column_difference(df1: pd.DataFrame, df2: pd.DataFrame) -> List[str]:
    diff = []
    df2_set = set(df2)
    for column in df1:
        if column not in df2_set:
            diff.append(column)
    return diff
