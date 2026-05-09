CREATE EXTENSION IF NOT EXISTS vector;

SELECT 'CREATE DATABASE mlflow OWNER agewell'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

SELECT 'CREATE DATABASE prefect OWNER agewell'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'prefect')\gexec

SELECT 'CREATE DATABASE agentos OWNER agewell'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agentos')\gexec

\connect agentos
CREATE EXTENSION IF NOT EXISTS vector;

\connect agewell
CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION agewell;
