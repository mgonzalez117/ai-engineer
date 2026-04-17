CREATE TABLE IF NOT EXISTS etl_metrics (
    id SERIAL PRIMARY KEY,
    pipeline_name TEXT NOT NULL,
    step TEXT NOT NULL,
    run_id TEXT NOT NULL,
    execution_date TIMESTAMP,
    nb_input INTEGER,
    nb_output INTEGER,
    nb_rejected INTEGER,
    duration_seconds DOUBLE PRECISION,
    rows_per_second DOUBLE PRECISION,
    success BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
