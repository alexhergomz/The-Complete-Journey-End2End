-- Connect to the database
\c dunnhumby

-- Import household demographics data
COPY household_demographics FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/hh_demographic.csv' DELIMITER ',' CSV HEADER;

-- Import product data
COPY product FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/product.csv' DELIMITER ',' CSV HEADER;

-- Import campaign description data
COPY campaign_desc(description, campaign, start_day, end_day) FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/campaign_desc.csv' DELIMITER ',' CSV HEADER;

-- Import campaign table data
COPY campaign_table FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/campaign_table.csv' DELIMITER ',' CSV HEADER;

-- Import coupon data
COPY coupon FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/coupon.csv' DELIMITER ',' CSV HEADER;

-- Import coupon redemption data
COPY coupon_redemption FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/coupon_redempt.csv' DELIMITER ',' CSV HEADER;

-- Import transaction data (this might take some time due to the large file size)
COPY transaction_data FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/transaction_data.csv' DELIMITER ',' CSV HEADER;

-- Import causal data (this might take some time due to the large file size)
COPY causal_data FROM 'C:/Users/alexh/Desktop/The Complete Journey/archive/causal_data.csv' DELIMITER ',' CSV HEADER;

-- Analyze tables for query optimization
ANALYZE; 