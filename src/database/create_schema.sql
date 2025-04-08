-- Create database for Dunnhumby The Complete Journey dataset
CREATE DATABASE dunnhumby;

-- Connect to the database
\c dunnhumby

-- Create tables

-- Household demographics table
CREATE TABLE household_demographics (
    household_key INTEGER PRIMARY KEY,
    age_desc VARCHAR(10),
    marital_status_code CHAR(1),
    income_desc VARCHAR(20),
    homeowner_desc VARCHAR(20),
    hh_comp_desc VARCHAR(20),
    household_size_desc VARCHAR(10),
    kid_category_desc VARCHAR(20)
);

-- Product table
CREATE TABLE product (
    product_id INTEGER PRIMARY KEY,
    manufacturer INTEGER,
    department VARCHAR(50),
    brand VARCHAR(50),
    commodity_desc VARCHAR(100),
    sub_commodity_desc VARCHAR(100),
    curr_size_of_product VARCHAR(50)
);

-- Transaction data table
CREATE TABLE transaction_data (
    household_key INTEGER,
    basket_id BIGINT,
    day INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    sales_value NUMERIC(10, 2),
    store_id INTEGER,
    retail_disc NUMERIC(10, 2),
    trans_time INTEGER,
    week_no INTEGER,
    coupon_disc NUMERIC(10, 2),
    coupon_match_disc NUMERIC(10, 2),
    PRIMARY KEY (basket_id, household_key, product_id),
    FOREIGN KEY (household_key) REFERENCES household_demographics(household_key),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

-- Campaign description table
CREATE TABLE campaign_desc (
    campaign INTEGER PRIMARY KEY,
    description VARCHAR(50),
    start_day INTEGER,
    end_day INTEGER
);

-- Campaign table (household-campaign mapping)
CREATE TABLE campaign_table (
    household_key INTEGER,
    campaign INTEGER,
    description VARCHAR(50),
    PRIMARY KEY (household_key, campaign),
    FOREIGN KEY (household_key) REFERENCES household_demographics(household_key),
    FOREIGN KEY (campaign) REFERENCES campaign_desc(campaign)
);

-- Coupon table
CREATE TABLE coupon (
    coupon_upc VARCHAR(50),
    product_id INTEGER,
    campaign INTEGER,
    PRIMARY KEY (coupon_upc, product_id, campaign),
    FOREIGN KEY (product_id) REFERENCES product(product_id),
    FOREIGN KEY (campaign) REFERENCES campaign_desc(campaign)
);

-- Coupon redemption table
CREATE TABLE coupon_redemption (
    household_key INTEGER,
    day INTEGER,
    coupon_upc VARCHAR(50),
    campaign INTEGER,
    PRIMARY KEY (household_key, day, coupon_upc),
    FOREIGN KEY (household_key) REFERENCES household_demographics(household_key),
    FOREIGN KEY (coupon_upc, campaign) REFERENCES coupon(coupon_upc, campaign)
);

-- Causal data table (for promotions and displays)
CREATE TABLE causal_data (
    product_id INTEGER,
    store_id INTEGER,
    week_no INTEGER,
    display VARCHAR(10),
    mailer VARCHAR(10),
    PRIMARY KEY (product_id, store_id, week_no),
    FOREIGN KEY (product_id) REFERENCES product(product_id)
);

-- Create indexes for faster querying
CREATE INDEX idx_transaction_household ON transaction_data(household_key);
CREATE INDEX idx_transaction_product ON transaction_data(product_id);
CREATE INDEX idx_transaction_week ON transaction_data(week_no);
CREATE INDEX idx_transaction_day ON transaction_data(day);
CREATE INDEX idx_coupon_redemption_household ON coupon_redemption(household_key);
CREATE INDEX idx_coupon_redemption_campaign ON coupon_redemption(campaign);
CREATE INDEX idx_campaign_table_household ON campaign_table(household_key);
CREATE INDEX idx_campaign_table_campaign ON campaign_table(campaign); 