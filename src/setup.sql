-- setup.sql
-- pgvectorの有効化
CREATE EXTENSION IF NOT EXISTS vector;

-- テーブルの作成
CREATE TABLE items ( id bigserial PRIMARY KEY, embedding vector(3) );

-- データ投入
INSERT INTO items (embedding) VALUES ('[1,2,3]'::vector), ('[4,5,6]'::vector), ('[7,8,9]'::vector), ('[0,1,2]'::vector), ('[3,4,5]'::vector), ('[6,7,8]'::vector), ('[9,0,1]'::vector), ('[2,3,4]'::vector), ('[5,6,7]'::vector), ('[8,9,0]'::vector);

-- 
CREATE TABLE imvecs ( id bigserial PRIMARY KEY, embedding vector(1280) );

-- 
CREATE TABLE images (
    image_id bigserial PRIMARY KEY,
    image_data bytea,
    file_name varchar(255),
    file_type varchar(50),
    upload_date timestamp DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1280),
    embedding_clip vector(768)
);

-- 
CREATE TABLE images2 (
    image_id bigserial PRIMARY KEY,
    image_data bytea,
    file_name varchar(255),
    file_type varchar(50),
    upload_date timestamp DEFAULT CURRENT_TIMESTAMP,
    embedding vector(1280)
);


-- L2距離
-- SELECT *, embedding <-> '[3,1,2]' AS l2_distance FROM items ORDER BY l2_distance;

-- 内積
-- SELECT *, (embedding <#> '[3,1,2]') * -1 AS inner_product FROM items ORDER BY inner_product;

-- コサイン類似度
-- SELECT *, 1 - (embedding <=> '[3,1,2]') AS cosine_similarity FROM items ORDER BY cosine_similarity;