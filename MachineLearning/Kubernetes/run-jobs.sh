#!/bin/bash 

# Function to check if a job is completed
check_job_status_and_cleanup() {
  job_name=$1
  namespace=$2 # Assumes default namespace if not specified
  while true; do
    # Check if the job is completed
    status=$(kubectl get jobs $job_name -n $namespace -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}')
    if [ "$status" == "True" ]; then
      echo "Job $job_name completed successfully."
      # Delete the job to clean up
      kubectl delete job $job_name -n $namespace
      echo "Job $job_name deleted."
      break
    else
      # Check if the job has failed
      failure=$(kubectl get jobs $job_name -n $namespace -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}')
      if [ "$failure" == "True" ]; then
        echo "Job $job_name failed."
        # Delete the job to clean up
        kubectl delete job $job_name -n $namespace
        echo "Job $job_name deleted."
        exit 1 # Exit with an error code to indicate failure
      fi
    fi
    echo "Waiting for job $job_name to complete..."
    sleep 30
  done
}

# Submit the first two jobs
kubectl apply -f /app/my_jobs/ray_stats_tune.yaml
kubectl apply -f /app/my_jobs/ray_nn_tune.yaml

# Wait for the first two jobs to complete and clean them up
check_job_status_and_cleanup "cpu-tuning-job" "default"
check_job_status_and_cleanup "gpu-tuning-job" "default"

# Submit the next two jobs
kubectl apply -f /app/my_jobs/ray_stats_pred.yaml
kubectl apply -f /app/my_jobs/ray_nn_pred.yaml

# Wait for the next two jobs to complete and clean them up
check_job_status_and_cleanup "cpu-predict-job" "default"
check_job_status_and_cleanup "gpu-predict-job" "default"
