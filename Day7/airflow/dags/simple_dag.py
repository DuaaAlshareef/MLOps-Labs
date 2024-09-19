from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define the default arguments for the DAG
default_args = {
    'owner': 'Duaa_Admin',
    'start_date': datetime(2024, 9, 19),  # Adjust start date to your needs
    'retries': 1
}

# Define the DAG
dag = DAG(
    'simple_airflow_dag',
    default_args=default_args,
    description='A simple DAG with two Python tasks',
    schedule_interval=None,  # Set to None for manual trigger
)

# Define the Python functions for the tasks
def print_starting_dag():
    print("Starting Airflow DAG")

def print_current_datetime():
    print(f"Current date and time: {datetime.now()}")

# Create the tasks
task_1 = PythonOperator(
    task_id='start_task',
    python_callable=print_starting_dag,
    dag=dag,
)

task_2 = PythonOperator(
    task_id='print_datetime',
    python_callable=print_current_datetime,
    dag=dag,
)

# Set the task sequence
task_1 >> task_2  # task_1 will run before task_2
