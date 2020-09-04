# Featurizing cannabis storefront data for fraud detection

## Part of [Berkeley Institute for Data Science](https://bids.berkeley.edu) research on the legal cannabis industry

## Running locally

### Environment
If you do not already have conda installed, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
Once you have it: 

    conda env create --file featurize_env.yml
    conda activate featurize_env

### Configuration
Config variables are located in the header of `featurize/featurize.py`. No configuration is required. The defaults are shown below:

    TAG_CHANGE_FIELDS = ['address', 'dispensary_name', 'email']

    LICENSE_INPUT_DIR = 'input/license/'
    PANEL_INPUT_DIR = 'input/panel/'
    DISP_SCRAPE_INPUT_DIR = 'input/scrapes/'

Optionally:
* Modify `TAG_CHANGE_FIELDS` to track storefronts' changes in different kinds of information
* Customize the input paths

### Input
1. Ensure all input files are in CSV format
2. Ensure all product-level panel data files have a non-empty `url` field
3. Place the product-level panel data files in `PANEL_INPUT_DIR` from above
4. If available, place the dispensary-level panel data files in `DISP_SCRAPE_INPUT_DIR`
5. If available, place CA commercial license files in `LICENSE_INPUT_DIR`

### Running
    
    ./run_featurize

### Output

* The featurized panel data files are written to `output/panel`
* The company groupings are written to `output/company`

## Feature key

### Storefront presence features
* **continued**: storefront present in the preceding panel
* **disappeared**: storefront not present in next panel
* **reappeared**: storefront present after disappearing previously

### Change-tracking features
* **changed_name**: storefront changed its name since last appearance
* **changed_address**: storefront changed its address since last appearance

Similar pattern for all features in `TAG_CHANGE_FIELDS` (see configuration section above).

### License identification features
* **license**: license number
* **license_business_type**: e.g. retailer, distributor
* **status**: e.g. active, expired
* **status_date**: date status was recorded
* **issue_date**: date license was issued
* **adult_use/medicinal**: permitted type of customer

### License status features
* **active_license**, **canceled_license**, **expired_license**, **revoked_license**, **suspended_license**: 
record storefront's license state according to CA commercial license database
* **possible_license**: storefront displays a unverified license (does not match with CA database)
* **future_license_explicit**: issue date of storefront's license is after scrape date
* **assumed_license**: storefront displays a verified license in a later panel

### Legality features
* **illegal_1912**: storefront present on 12/15/19 but not present on 01/12/20
* **illegal_other**: storefront present before 12/15/19 but not present on 01/12/20

### Storefront identification features
* **slug**: portion of a storefront's URL after the final slash
* **is_dispensary**: storefront is a brick-and-mortar dispensary
* **is_delivery**: storefront is a delivery service

### Company grouping
For each panel, a mapping of storefront IDs to company IDs is generated using similarity of product offering, 
phone number, and email. This is written as a separate file.