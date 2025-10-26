FROM apache/airflow:2.7.3-python3.10

USER airflow

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    kneed \
    requests \
    apache-airflow-providers-postgres \
    great-expectations==0.18.8 \
    sqlalchemy==1.4.48

RUN python -c "import great_expectations as ge; print('✅ Great Expectations version:', ge.__version__)"