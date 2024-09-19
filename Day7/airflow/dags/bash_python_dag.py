from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

# Default arguments for the DAG
default_args = {
    'owner': 'Duaa_Admin',
    'start_date': datetime(2023, 9, 19),
    'retries': 1,
}

# Define the DAG with a schedule interval to run every 5 minutes
dag = DAG(
    'bash_python_dag',
    default_args=default_args,
    description='A DAG with Bash and Python tasks running every 5 minutes',
    schedule_interval='*/5 * * * *',  # Cron expression for every 5 minutes
)

# Define the Bash task
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Running Bash task"',
    dag=dag,
)

# Define the Python task
def print_hello():
    print("Hello from Airflow")

python_task = PythonOperator(
    task_id='python_task',
    python_callable=print_hello,
    dag=dag,
)

# Set the task sequence
bash_task >> python_task  # Bash task runs first, then Python task
