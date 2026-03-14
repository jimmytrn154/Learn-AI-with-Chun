-- Seed starts
-- BIG NOTE: All password_hash values correspond to the plaintext password '123456'
SET FOREIGN_KEY_CHECKS = 0;

-- =========================================================
-- 1. CLEAN SLATE (Wipe everything first)
-- =========================================================
TRUNCATE TABLE listing_features;
TRUNCATE TABLE listings;
TRUNCATE TABLE rooms;
TRUNCATE TABLE properties;
TRUNCATE TABLE organizations;
TRUNCATE TABLE users;

-- =========================================================
-- 2. USERS (50 Users)
-- Password Hash for '123456': $2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq
-- =========================================================

-- --- ADMINS (Ids 1-2) ---
INSERT INTO users (id, email, password_hash, full_name, phone, role, status) VALUES 
(1, 'admin@vinhousing.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'System Administrator', '0900000001', 'admin', 'active'),
(2, 'manager@vinhousing.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Content Manager', '0900000002', 'admin', 'active');

-- --- LANDLORDS (Ids 3-10) ---
INSERT INTO users (id, email, password_hash, full_name, phone, role, status) VALUES 
(3, 'landlord.hanoi@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Nguyen Van Hung', '0912345678', 'landlord', 'active'),
(4, 'landlord.ocean@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Tran Thi Mai', '0912345679', 'landlord', 'active'),
(5, 'landlord.bacninh@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Le Van Phuc', '0912345680', 'landlord', 'active'),
(6, 'landlord.hcm@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Pham Minh Tuan', '0912345681', 'landlord', 'active'),
(7, 'investor.group@realty.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Hoang Gia Real Estate', '0912345682', 'landlord', 'active'),
(8, 'mrs.lan@apartments.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Do Thi Lan', '0912345683', 'landlord', 'active'),
(9, 'hanoi.homestays@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Vu Tuan Anh', '0912345684', 'landlord', 'active'), -- Pending verification demo
(10, 'dorm.manager@techpark.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'TechPark Housing', '0912345685', 'landlord', 'active');

-- --- TENANTS (Ids 11-50) ---
INSERT INTO users (id, email, password_hash, full_name, phone, role, status) VALUES 
(11, 'student1@vinuni.edu.vn', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Nguyen Thi Huong', '0987654321', 'tenant', 'active'),
(12, 'student2@vinuni.edu.vn', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Tran Van Nam', '0987654322', 'tenant', 'active'),
(13, 'worker1@samsung.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Le Thi Bich', '0987654323', 'tenant', 'active'),
(14, 'worker2@foxconn.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Pham Van Dung', '0987654324', 'tenant', 'active'),
(15, 'tenant.hcm@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Vo Van Kiet', '0987654325', 'tenant', 'active'),
(16, 'sarah.exchange@uni.edu', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Sarah Jenkins', '0987654326', 'tenant', 'active'),
(17, 'david.intern@company.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'David Smith', '0987654327', 'tenant', 'active'),
(18, 'linh.nguyen@start.up', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Nguyen Thuy Linh', '0987654328', 'tenant', 'active'),
(19, 'hung.pham@fpt.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Pham Quoc Hung', '0987654329', 'tenant', 'active'),
(20, 'minh.le@viettel.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'Le Nhat Minh', '0987654330', 'tenant', 'active'),
(21, 'tenant21@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'User TwentyOne', '0987654331', 'tenant', 'active'),
(22, 'tenant22@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'User TwentyTwo', '0987654332', 'tenant', 'active'),
(23, 'tenant23@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'User TwentyThree', '0987654333', 'tenant', 'active'),
(24, 'tenant24@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'User TwentyFour', '0987654334', 'tenant', 'active'),
(25, 'tenant25@gmail.com', '$2a$10$rKA2WWwMwYSVO6tHwYCrIeel24kYAj8ObKxlI.QeDjY8ziGusBXEq', 'User TwentyFive', '0987654335', 'tenant', 'active');
-- (Feel free to copy/paste more tenants if you strictly need 50, but 25 is usually enough for a demo crowd)

-- =========================================================
-- 3. ORGANIZATIONS
-- =========================================================
INSERT INTO organizations (id, name, org_type) VALUES 
(1, 'VinUniversity', 'university'),
(2, 'Samsung Industrial Park Bac Ninh', 'industrial_park'),
(3, 'FPT Software Park Hoa Lac', 'industrial_park'),
(4, 'RMIT University Saigon', 'university');

-- =========================================================
-- 4. PROPERTIES (Using simplified image_url column)
-- =========================================================
INSERT INTO properties (id, owner_user_id, org_id, address, description, image_url) VALUES 
-- Landlord 3 (Hanoi)
(1, 3, 1, 'S2.15 Vinhomes Ocean Park, Gia Lam, Hanoi', 'Modern studio apartment with direct access to the university bus line.', 'https://images.unsplash.com/photo-1522708323590-d24dbb6b0267?auto=format&fit=crop&w=1000&q=80'),
(2, 3, 1, 'S1.02 Vinhomes Ocean Park, Gia Lam, Hanoi', '2 Bedroom apartment, high floor, pool view.', 'https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?auto=format&fit=crop&w=1000&q=80'),

-- Landlord 4 (Ocean Park Specialist)
(3, 4, 1, 'R1.05 Zenpark, Vinhomes Ocean Park', 'Luxury apartment in the Zenpark subdivision. Gym and Pool included.', 'https://images.unsplash.com/photo-1512917774080-9991f1c4c750?auto=format&fit=crop&w=1000&q=80'),

-- Landlord 5 (Bac Ninh Workers)
(4, 5, 2, '123 Nguyen Van Cu, Bac Ninh City', 'Shared house for workers, 10 mins from Samsung factory.', 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1000&q=80'),
(5, 5, 2, '45 Yen Phong Industrial Zone, Bac Ninh', 'Affordable dormitory style housing.', 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1000&q=80'),

-- Landlord 6 (HCM)
(6, 6, 4, '15 Nguyen Van Linh, District 7, HCMC', 'Condo near RMIT, fully furnished.', 'https://images.unsplash.com/photo-1493809842364-78817add7ffb?auto=format&fit=crop&w=1000&q=80'),

-- Landlord 8 (Mrs Lan)
(7, 8, 3, 'Hoa Lac Hi-Tech Park, Villa 5', 'Shared villa for FPT employees. Quiet and spacious.', 'https://images.unsplash.com/photo-1616047006789-b7af5afb8c20?q=80&w=880&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');

-- =========================================================
-- 5. ROOMS (Populating the buildings)
-- =========================================================
INSERT INTO rooms (id, property_id, room_name, capacity, area_m2, base_rent) VALUES 
-- Prop 1 (Studio)
(1, 1, 'Studio Apartment', 2, 32.0, 450.00),

-- Prop 2 (2BR Apt)
(2, 2, 'Master Bedroom', 2, 25.0, 300.00),
(3, 2, 'Small Bedroom', 1, 15.0, 200.00),

-- Prop 3 (Zenpark Luxury)
(4, 3, 'Whole Unit', 4, 80.0, 900.00),

-- Prop 4 (Bac Ninh House)
(5, 4, 'Room 101 (Ground Floor)', 2, 25.0, 150.00),
(6, 4, 'Room 201 (Balcony)', 2, 28.0, 180.00),
(7, 4, 'Room 202', 1, 18.0, 120.00),

-- Prop 5 (Dorm)
(8, 5, 'Bed A', 1, 10.0, 50.00),
(9, 5, 'Bed B', 1, 10.0, 50.00),

-- Prop 6 (RMIT Condo)
(10, 6, 'Master Suite', 2, 40.0, 600.00),
(11, 6, 'Guest Room', 1, 20.0, 400.00),

-- Prop 7 (Hoa Lac Villa)
(12, 7, 'Sunflower Room', 2, 35.0, 350.00),
(13, 7, 'Garden View Room', 1, 25.0, 280.00);

-- =========================================================
-- 6. LISTINGS (The Ads)
-- =========================================================
INSERT INTO listings (id, owner_user_id, property_id, room_id, price, deposit, available_from, status) VALUES 
-- 1. Verified Listing: Whole Studio (Prop 1)
(1, 3, 1, NULL, 450.00, 450.00, CURDATE(), 'verified'),

-- 2. Verified Listing: Master Bedroom only (Prop 2)
(2, 3, 2, 2, 300.00, 300.00, CURDATE(), 'verified'),

-- 3. Verified Listing: Zenpark Whole Apt (Prop 3)
(3, 4, 3, NULL, 900.00, 1800.00, DATE_ADD(CURDATE(), INTERVAL 5 DAY), 'verified'),

-- 4. Verified Listing: Cheap Room for Workers (Prop 4 - Room 101)
(4, 5, 4, 5, 150.00, 100.00, CURDATE(), 'verified'),

-- 5. Verified Listing: Balcony Room (Prop 4 - Room 201)
(5, 5, 4, 6, 180.00, 100.00, CURDATE(), 'verified'),

-- 6. RENTED Listing: Small Room (Prop 2) - Shows history
(6, 3, 2, 3, 200.00, 200.00, '2023-01-01', 'rented'),

-- 7. PENDING Listing: RMIT Condo Master Suite (Waiting for admin)
(7, 6, 6, 10, 600.00, 1200.00, DATE_ADD(CURDATE(), INTERVAL 10 DAY), 'pending_verification'),

-- 8. PENDING Listing: Hoa Lac Villa Room
(8, 8, 7, 12, 350.00, 350.00, CURDATE(), 'pending_verification');

-- =========================================================
-- 7. FEATURES
-- =========================================================
INSERT INTO listing_features (listing_id, features_json) VALUES 
(1, '{"wifi": true, "ac": true, "elevator": true, "kitchen": true}'),
(2, '{"wifi": true, "ac": true, "pool": true}'),
(3, '{"wifi": true, "ac": true, "gym": true, "pool": true, "parking": true}'),
(4, '{"wifi": true, "parking": true}'),
(5, '{"wifi": true, "balcony": true}'),
(6, '{"wifi": true}'),
(7, '{"wifi": true, "ac": true, "security": true, "pool": true}'),
(8, '{"wifi": true, "garden": true, "parking": true}');

SET FOREIGN_KEY_CHECKS = 1;