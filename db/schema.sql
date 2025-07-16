-- schema.sql
CREATE TABLE IF NOT EXISTS queries (
    query_id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    dataset TEXT,
    source_id TEXT,
    reference TEXT,
    extraction_method TEXT,
    evaluation_metrics TEXT,
    generation_parameters JSONB
);

CREATE TABLE IF NOT EXISTS query_features (
    query_id INTEGER REFERENCES queries(query_id),
    task_type TEXT,
    semantic_cluster INTEGER,
    complexity_score FLOAT,
    PRIMARY KEY (query_id)
);

CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parameter_count BIGINT,
    model_type TEXT,
    quantization TEXT
);


CREATE TABLE IF NOT EXISTS inference_results (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(query_id),
    model_id TEXT NOT NULL,
    generated_output TEXT,
    accuracy FLOAT,
    energy_consumption FLOAT,
    latency FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



CREATE TABLE IF NOT EXISTS pregenerated_results (
    id SERIAL PRIMARY KEY,
    query_id INTEGER REFERENCES queries(query_id) ON DELETE CASCADE,
    model_id TEXT REFERENCES models(model_id) ON DELETE CASCADE,
    run_id INTEGER,
    generated_output TEXT,
    accuracy FLOAT,
    energy_consumption FLOAT,
    latency FLOAT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(query_id, model_id, run_id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_pregenerated_results_query_model 
ON pregenerated_results(query_id, model_id);

