WITH filtered AS (
  -- Step 1: Filter transactions with positive purchase amounts and cast the date.
  SELECT 
    id,
    chain,
    dept,
    category,
    brand,
    productmeasure,
    CAST(date AS DATE) AS date_dt,
    purchaseamount
  FROM transactions
  WHERE purchaseamount > 0
),
start_dates AS (
  -- Step 2: Compute the earliest (start) date per customer.
  SELECT 
    id, 
    MIN(date_dt) AS start_date
  FROM filtered
  GROUP BY id
),
calibration_value AS (
  -- Step 3: Sum purchase amounts on the start date for each customer.
  SELECT 
    f.id, 
    SUM(f.purchaseamount) AS calibration_value
  FROM filtered f
  JOIN start_dates s ON f.id = s.id
  WHERE f.date_dt = s.start_date
  GROUP BY f.id
),
holdout_value AS (
  -- Step 4: Sum purchase amounts during the holdout period (after start_date up to 365 days later).
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
  -- Step 5: For each customer, select one representative row from the start_date.
  -- We use DISTINCT ON to pick the row with the highest purchaseamount.
  SELECT DISTINCT ON (f.id) 
    f.id,
    f.chain,
    f.dept,
    f.category,
    f.brand,
    f.productmeasure
  FROM filtered f
  JOIN start_dates s ON f.id = s.id
  WHERE f.date_dt = s.start_date
  ORDER BY f.id, f.purchaseamount DESC
)
-- Step 6: Merge all computed pieces.
SELECT 
  cv.id,
  cv.calibration_value,
  COALESCE(hv.holdout_value, 0) AS holdout_value,
  ca.chain,
  ca.dept,
  ca.category,
  ca.brand,
  ca.productmeasure,
  LOG(cv.calibration_value) AS log_calibration_value,
  -- Here, the label is the holdout_value.
  COALESCE(hv.holdout_value, 0) AS label
FROM calibration_value cv
LEFT JOIN holdout_value hv ON cv.id = hv.id
LEFT JOIN calibration_attributes ca ON cv.id = ca.id;