    USE magist;
    /* *************************************************************************************************************************************************
                                                             In relation to the products
       ************************************************************************************************************************************************* */
    -- Question: What categories of tech products does Magist have?
    
SELECT DISTINCT
    product_category_name_english AS categories_of_tech_products
FROM
    order_items
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
WHERE
    product_category_name_english LIKE ('%computer%')
        OR product_category_name_english LIKE ('%phon%')
        OR product_category_name_english LIKE ('%game%')
        OR product_category_name_english LIKE ('%appliance%')
        OR product_category_name_english LIKE ('%electronics%')
        OR product_category_name_english LIKE ('%auto%')
        OR product_category_name_english LIKE ('%air%')
        OR product_category_name_english LIKE ('%signaling%');
        -- ************************************************************************************************************************************************
           -- Question: No. of products of tech categories sold?
           
SELECT 
    COUNT(*) AS tech_products_sold,
    (SELECT 
            COUNT(*)
        FROM
            order_items) AS total_products_sold,
    (COUNT(*)) / (SELECT 
            COUNT(*)
        FROM
            order_items) * 100 AS percentage_over_total
FROM
    order_items
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
WHERE
    product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');
    -- **************************************************************************************************************************************************************
    -- Question: Average price of the products being sold?
    
SELECT 
    COUNT(*) AS tech_products_sold,
    ROUND(AVG(price)) AS avergae_price_of_sold_products
FROM
    order_items
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
WHERE
    product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');
    -- ***************************************************************************************************************************************************************
    	-- Question: Are expensive tech products popular? 
    
SELECT 
    ROUND(AVG(price))
FROM
    order_items;
SELECT 
    MAX(price)
FROM
    order_items;
    /*--------------------------------------------------------------------------------------------------------------------------------------------------
    AS maximum price of the prduct is '6,735' ans average price of the sold products is '121'. So I assume that the products whose value is equal to 
    or greater than 1000 are categorized as expensive products.
      -------------------------------------------------------------------------------------------------------------------------------------------------*/
    SELECT 
    COUNT(*) AS total_sold_tech_products,
    COUNT(CASE
        WHEN price >= 1000 THEN '1'
    END) AS expensive_tech_product,
    COUNT(CASE
        WHEN price >= 1000 THEN '1'
    END) / COUNT(*) * 100 AS '%age_of_expensive_over_total',
    COUNT(CASE
        WHEN price < 1000 THEN '1'
    END) AS cheap_tech_product,
    COUNT(CASE
        WHEN price < 1000 THEN '1'
    END) / COUNT(*) * 100 '%age_of_cheap_over_total'
FROM
    order_items
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
WHERE
    product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');
    
	  /* ***********************************************************************************************************************************************
                                                             In relation to the sellers
       ************************************************************************************************************************************************* */
       -- Question: How many months of data are included in the magist database?
       
SELECT 
    (TIMESTAMPDIFF(MONTH,
        MIN(order_purchase_timestamp),
        MAX(order_purchase_timestamp))) AS number_of_months
FROM
    orders;
    -- **************************************************************************************************************************************************
    
-- Question: How many sellers are there?? 

SELECT 
    COUNT(DISTINCT seller_id) AS total_sellers
FROM
    sellers;

-- Question: How many Tech sellers are there and what percentage of overall sellers are Tech sellers?

SELECT 
    COUNT(DISTINCT sellers.seller_id) AS no_of_tech_sellers,
    (SELECT 
            COUNT(DISTINCT seller_id)
        FROM
            sellers) AS total_sellers,
    COUNT(DISTINCT sellers.seller_id) / (SELECT 
            COUNT(DISTINCT seller_id)
        FROM
            sellers) * 100 AS tech_sellers_percentage
FROM
    sellers
        INNER JOIN
    order_items ON sellers.seller_id = order_items.seller_id
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
WHERE product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');
    -- **************************************************************************************************************************************************************
-- Question  total amount earned by all sellers and by tech sellers?
SELECT 
    ROUND((SELECT 
                    SUM(payment_value)
                FROM
                    order_payments)) AS amount_by_all_seller,
    ROUND(SUM(order_payments.payment_value)) AS amount_by_tech_seller
FROM
    sellers
        INNER JOIN
    order_items ON sellers.seller_id = order_items.seller_id
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
        INNER JOIN
    order_payments ON order_items.order_id = order_payments.order_id
WHERE product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');
    -- **************************************************************************************************************************************************************
    -- Question: average monthly income of all sellers and average monthly income of tech sellers?
SELECT 
    ROUND((SELECT 
                    SUM(payment_value)
                FROM
                    order_payments) / (SELECT 
                    (TIMESTAMPDIFF(MONTH,
                            MIN(order_purchase_timestamp),
                            MAX(order_purchase_timestamp)))
                FROM
                    orders)) AS avg_monthly_income_all_seller,
    ROUND(SUM(order_payments.payment_value) / (SELECT 
                    (TIMESTAMPDIFF(MONTH,
                            MIN(order_purchase_timestamp),
                            MAX(order_purchase_timestamp)))
                FROM
                    orders)) AS avg_monthly_income_tech_seller
FROM
    sellers
        INNER JOIN
    order_items ON sellers.seller_id = order_items.seller_id
        INNER JOIN
    products ON order_items.product_id = products.product_id
        INNER JOIN
    product_category_name_translation ON products.product_category_name = product_category_name_translation.product_category_name
        INNER JOIN
    order_payments ON order_items.order_id = order_payments.order_id
WHERE product_category_name_english IN ('consoles_games' , 'home_appliances','home_appliances_2','electronics','small_appliances','computers_accessories','pc_gamer',
    'computers','small_appliances_home_oven_and_coffee','auto','signaling_and_security','telephony','fixed_telephony','air_conditioning');

	  /* ***********************************************************************************************************************************************
                                                             In relation to the sellers
       ************************************************************************************************************************************************* */
       -- Question: What’s the average time between the order being placed and the product being delivered?
       
SELECT 
    ROUND(AVG(TIMESTAMPDIFF(DAY,
                order_purchase_timestamp,
                order_delivered_customer_date))) AS Average_time_in_days
FROM
    orders;
    -- **************************************************************************************************************************************************
    -- Question          Order_status
SELECT 
    order_status, COUNT(*) AS no_od_orders
FROM
    orders
GROUP BY order_status;

-- Question: How many orders are delivered on time vs orders delivered with a delay?

SELECT 
    COUNT(*) AS no_of_orders,
    CASE
        WHEN order_delivered_customer_date <= order_estimated_delivery_date THEN 'IN_TIME'
        WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 'LATE'
        ELSE 'No Info'
    END AS delivery_status
FROM
    orders
GROUP BY delivery_status;

-- ***************************************************************************************************************************************************
-- Question: Is there any pattern for delayed orders?
SELECT count(DISTINCT orders.order_id) As orders,
    CASE
        WHEN order_delivered_customer_date <= order_estimated_delivery_date THEN 'IN_TIME'
        WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 'LATE'
        ELSE 'No Info'
    END AS delivery_status
FROM
    orders
        LEFT JOIN
    order_items ON orders.order_id = order_items.order_id
    
    LEFT JOIN
    products ON order_items.product_id = products.product_id
    
    WHERE product_weight_g>=2500 OR product_height_cm>=90 
GROUP BY delivery_status;

-- Delivery time somehow depends on product height and weight..
    
