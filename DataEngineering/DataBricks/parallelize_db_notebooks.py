from concurrent.futures import ThreadPoolExecutor
import time

# dims that databricks is in charge of
notebook_list = ['notebook1', 'notebook2']


# recieve step
step_ = dbutils.widgets.get('step')

# number of parallel notebooks
parallel_workers = 4


def run_notebook(path):
    try:
        # announce what we are doing
        print(f'Now working on Notebook: {path}')

        # track the time it takes:
        timestamp1 = time.time()

        # run the notebook
        dbutils.notebook.run(f"{path}", timeout_seconds=9700, arguments={"input-data": path, 'step': step_})

        # close time loop
        timestamp2 = time.time()
        time_Delta = (timestamp2 - timestamp1) / 60

        # report minutes
        print(f'This notebook: {path} took: {time_Delta:.{3}f} minutes')

    except Exception as e:
        print(f'Exception found: {e}')

with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
    executor.map(run_notebook, notebook_list)
print("Completed operations")

