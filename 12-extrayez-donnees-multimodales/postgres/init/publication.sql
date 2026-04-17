CREATE TABLE IF NOT EXISTS publication (
    id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    fake BOOLEAN NOT NULL,

    content TEXT,
    image_url TEXT NOT NULL,
    document_url TEXT,
    video_url TEXT,
    author TEXT,

    PRIMARY KEY (source, id)
);
