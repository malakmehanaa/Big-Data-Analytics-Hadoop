-- ============================================================
-- SECTION 1 : LOAD RAW DATA WITH FULL SCHEMA IN ONE PASS
-- ============================================================

customers_raw = LOAD '/user/hduser/customers/customers-1000000.csv'
    USING PigStorage(',')
    AS (
        idx:int,
        customer_id:chararray,
        first_name:chararray,
        last_name:chararray,
        company:chararray,
        city:chararray,
        country:chararray,
        phone1:chararray,
        phone2:chararray,
        email:chararray,
        subscription_date:chararray,
        website:chararray
    );

-- ============================================================
-- SECTION 2 : REMOVE HEADER ROW
-- idx = 'Index' casts to NULL as int → safe header filter
-- ============================================================

no_header = FILTER customers_raw
    BY (idx IS NOT NULL) AND (customer_id != 'Customer Id');

-- ============================================================
-- SECTION 3 : DATA CLEANING
-- ============================================================

not_null = FILTER no_header
    BY (customer_id IS NOT NULL)
    AND (TRIM(customer_id) != '');

normalised = FOREACH not_null GENERATE
    idx                     AS idx,
    TRIM(customer_id)       AS customer_id,
    TRIM(first_name)        AS first_name,
    TRIM(last_name)         AS last_name,
    TRIM(company)           AS company,
    TRIM(city)              AS city,
    UPPER(TRIM(country))    AS country,
    TRIM(phone1)            AS phone1,
    TRIM(phone2)            AS phone2,
    LOWER(TRIM(email))      AS email,
    TRIM(subscription_date) AS subscription_date,
    TRIM(website)           AS website;

-- Deduplicate by customer_id (keep first by idx)
grouped_by_id = GROUP normalised BY customer_id;

deduplicated = FOREACH grouped_by_id {
    sorted_group = ORDER normalised BY idx ASC;
    first_row    = LIMIT sorted_group 1;
    GENERATE FLATTEN(first_row);
};

-- FIX: after FLATTEN(first_row), schema prefix is first_row::
cleaned = FOREACH deduplicated GENERATE
    first_row::idx               AS idx,
    first_row::customer_id       AS customer_id,
    first_row::first_name        AS first_name,
    first_row::last_name         AS last_name,
    first_row::company           AS company,
    first_row::city              AS city,
    first_row::country           AS country,
    first_row::phone1            AS phone1,
    first_row::phone2            AS phone2,
    first_row::email             AS email,
    first_row::subscription_date AS subscription_date,
    first_row::website           AS website;

-- ============================================================
-- SECTION 4 : TOTAL CUSTOMER COUNT  (~1 line output)
-- ============================================================

all_customers   = GROUP cleaned ALL;
total_customers = FOREACH all_customers GENERATE
    'TOTAL_CUSTOMERS' AS metric,
    COUNT(cleaned)    AS total;

STORE total_customers
    INTO '/user/hduser/output/total_customers'
    USING PigStorage(',');

-- ============================================================
-- SECTION 5 : CUSTOMERS PER COUNTRY  (~200 lines output)
-- ============================================================

by_country     = GROUP cleaned BY country;
country_counts = FOREACH by_country GENERATE
    group          AS country,
    COUNT(cleaned) AS customer_count;

STORE country_counts
    INTO '/user/hduser/output/country_counts'
    USING PigStorage(',');

-- ============================================================
-- SECTION 6 : CUSTOMERS PER COMPANY  (~large but manageable)
-- ============================================================

by_company     = GROUP cleaned BY company;
company_counts = FOREACH by_company GENERATE
    group          AS company,
    COUNT(cleaned) AS customer_count;

STORE company_counts
    INTO '/user/hduser/output/company_counts'
    USING PigStorage(',');

-- ============================================================
-- SECTION 7 : FILTER – EGYPT CUSTOMERS  (small subset)
-- ============================================================

egypt_customers = FILTER cleaned BY country == 'EGYPT';

-- Store only key columns to keep output small
egypt_slim = FOREACH egypt_customers GENERATE
    customer_id,
    first_name,
    last_name,
    city,
    email,
    subscription_date;

STORE egypt_slim
    INTO '/user/hduser/output/egypt_customers'
    USING PigStorage(',');

-- ============================================================
-- SECTION 8 : DATE RANGE  (2 values output)
-- ============================================================

all_for_dates = GROUP cleaned ALL;
date_range    = FOREACH all_for_dates GENERATE
    MIN(cleaned.subscription_date) AS earliest_subscription,
    MAX(cleaned.subscription_date) AS latest_subscription;

STORE date_range
    INTO '/user/hduser/output/date_range'
    USING PigStorage(',');

-- ============================================================
-- SECTION 9 : TOP 5 COUNTRIES  (5 lines output)
-- ORDER on already-aggregated small relation — safe
-- ============================================================

country_sorted = ORDER country_counts BY customer_count DESC;
top5_countries = LIMIT country_sorted 5;

STORE top5_countries
    INTO '/user/hduser/output/top5_countries'
    USING PigStorage(',');

-- ============================================================
-- SECTION 10 : ML-READY DATASET
-- Store only customer_id + email + country + subscription_date
-- This is the minimum identity+feature set for ML pipelines
-- and produces ~60MB instead of ~300MB
-- ============================================================

ml_ready = FOREACH cleaned GENERATE
    customer_id,
    country,
    city,
    email,
    subscription_date;

STORE ml_ready
    INTO '/user/hduser/output/ml_ready_dataset'
    USING PigStorage(',');

-- ============================================================
-- END OF PIPELINE
-- ============================================================
