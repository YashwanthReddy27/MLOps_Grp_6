# DataPipeline

## Run Airflow with Docker Compose

This guide explains how to initialize and run Apache Airflow using the provided `docker.yaml` configuration file.

### Prerequisites
- **Docker** and **Docker Compose** installed on your local machine  
- The `docker.yaml` file must be located in the project’s root directory

### Initialization

Before starting the Airflow services, initialize the Airflow environment and database:

```
docker compose up airflow-init
```

This command sets up directories, initializes the metadata database, and prepares your environment for Airflow.

### Start Airflow Services

Once initialization is complete, start all Airflow containers (webserver, scheduler, worker, etc.):

```
docker compose  up
```

### Access the Airflow Web UI

After the containers start, open your browser and visit:

```
http://localhost:8080
```

You can log in using the credentials configured in your environment variables or the default Airflow credentials if provided.

### Stop and Clean Up

To stop the running containers:

```
docker compose down
```

To remove all associated volumes and networks as well:

```
docker compose down -v
```
