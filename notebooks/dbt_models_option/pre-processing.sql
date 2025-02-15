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
  COALESCE(hv.holdout_value, 0) AS label
FROM calibration_value cv
LEFT JOIN holdout_value hv ON cv.id = hv.id
LEFT JOIN calibration_attributes ca ON cv.id = ca.id;