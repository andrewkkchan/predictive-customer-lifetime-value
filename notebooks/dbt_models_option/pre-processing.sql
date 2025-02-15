USE DATABASE CLV_DB;
USE WAREHOUSE CLV_WH;
CREATE OR REPLACE TABLE my_transactions (
    id VARCHAR,
    chain VARCHAR,
    dept VARCHAR,
    category VARCHAR,
    company VARCHAR,
    brand VARCHAR,
    date DATE,
    productsize FLOAT,
    productmeasure VARCHAR,
    purchasequantity FLOAT,
    purchaseamount FLOAT
);

COPY INTO my_transactions
FROM @my_internal_stage/transactions_company_1078778070.csv
FILE_FORMAT = (FORMAT_NAME = 'my_csv_format');

select * from my_transactions limit 10;

CREATE OR REPLACE TABLE customer_level_data AS
WITH filtered AS (
  SELECT 
    id,
    chain,
    dept,
    category,
    brand,
    productmeasure,
    CAST(date AS DATE) AS date_dt,
    purchaseamount,
    company
  FROM my_transactions
  WHERE purchaseamount > 0
),
start_dates AS (
  SELECT 
    id, 
    MIN(date_dt) AS start_date
  FROM filtered
  GROUP BY id
),
calibration_value AS (
  SELECT 
    f.id, 
    SUM(f.purchaseamount) AS calibration_value
  FROM filtered f
  JOIN start_dates s ON f.id = s.id
  WHERE f.date_dt = s.start_date
  GROUP BY f.id
),
holdout_value AS (
  SELECT 
    f.id, 
    SUM(f.purchaseamount) AS holdout_value
  FROM filtered f
  JOIN start_dates s ON f.id = s.id
  WHERE f.date_dt > s.start_date
    AND f.date_dt <= s.start_date + INTERVAL '365 days'
  GROUP BY f.id
),
calibration_attributes AS (
  SELECT 
    f.id,
    f.chain,
    f.dept,
    f.category,
    f.brand,
    f.productmeasure,
    ROW_NUMBER() OVER (PARTITION BY f.id ORDER BY f.purchaseamount DESC) AS rn
  FROM filtered f
  JOIN start_dates s ON f.id = s.id
  WHERE f.date_dt = s.start_date
  QUALIFY rn = 1
)
SELECT 
  cv.id,
  cv.calibration_value,
  COALESCE(hv.holdout_value, 0) AS holdout_value,
  ca.chain,
  ca.dept,
  ca.category,
  ca.brand,
  ca.productmeasure,
  LN(cv.calibration_value) AS log_calibration_value,
  COALESCE(hv.holdout_value, 0) AS label
FROM calibration_value cv
LEFT JOIN holdout_value hv ON cv.id = hv.id
LEFT JOIN calibration_attributes ca ON cv.id = ca.id;

select * from customer_level_data;