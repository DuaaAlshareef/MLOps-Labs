from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import pendulum

# Define default arguments for the DAG
default_args = {
    'owner': 'Duaa_Admin',
    'start_date': datetime(2023, 9, 19),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'xcom_dag',
    default_args=default_args,
    description='A DAG demonstrating XCom usage',
    schedule_interval=None,  # Manual trigger for now
)

# Task 1: Python task to push the current timestamp to XCom
def push_timestamp(ti):
    current_time = str(pendulum.now())  # Using Pendulum for timezone awareness
    ti.xcom_push(key='timestamp', value=current_time)
    print(f"Timestamp pushed to XCom: {current_time}")

push_task = PythonOperator(
    task_id='push_timestamp_task',
    python_callable=push_timestamp,
    dag=dag,
)

# Task 2: Bash task to pull the value from XCom and echo it
bash_task = BashOperator(
    task_id='bash_pull_task',
    bash_command='echo "Pulled timestamp from XCom: {{ ti.xcom_pull(key=\'timestamp\') }}"',
    dag=dag,
)

# Set task dependencies
push_task >> bash_task  # Bash task runs after the Python task
