-- Add image column within properties. Consider to add img for rooms also?
ALTER TABLE properties ADD COLUMN image_url TEXT DEFAULT NULL;
-- run this view instead
CREATE OR REPLACE VIEW v_searchable_listings AS
SELECT 
    l.id,
    l.owner_user_id,
    l.property_id,
    l.room_id,
    l.price,
    l.deposit,
    l.available_from,
    l.status,
    l.created_at,
    u.full_name AS owner_name,
    p.address AS property_address,
    p.geo_lat,
    p.geo_lng,
    p.description, -- ✅ FIX: Pull description from Property (p), not Listing (l)
    p.image_url,
    r.room_name,
    -- 1. Unit Name Logic
    COALESCE(r.room_name, 'Whole Property') AS unit_name,
    
    -- 2. Listing Type Logic
    CASE WHEN l.room_id IS NULL THEN 'whole_house' ELSE 'room' END AS listing_type,
    
    -- 3. Area Logic (Room vs Whole House Sum)
    COALESCE(
        r.area_m2, 
        (SELECT SUM(area_m2) FROM rooms WHERE property_id = l.property_id)
    ) AS area_m2,

    -- 4. Capacity Logic (Room vs Whole House Sum)
    COALESCE(
        r.capacity, 
        (SELECT SUM(capacity) FROM rooms WHERE property_id = l.property_id)
    ) AS capacity,
    
    lf.features_json
FROM listings l
JOIN properties p ON l.property_id = p.id
LEFT JOIN users u ON l.owner_user_id = u.id
LEFT JOIN rooms r ON l.room_id = r.id
LEFT JOIN listing_features lf ON l.id = lf.listing_id;

-- 1. Drop the strict constraint that forbids having both IDs
ALTER TABLE listings DROP CHECK ck_listing_target;

-- 2. (Optional but recommended) Add a better constraint
-- This new rule just says: "You must have AT LEAST one of them."
ALTER TABLE listings ADD CONSTRAINT ck_listing_target 
CHECK (property_id IS NOT NULL OR room_id IS NOT NULL);
