#!/bin/bash

# Function to check and delete pods with ErrImagePull or ImagePullBackOff  
delete_stuck_pods() {  
  while true; do  
    for pod in $(kubectl get pods --no-headers | grep -E 'ErrImagePull|ImagePullBackOff' | awk '{print $1}'); do  
      echo "Deleting pod $pod stuck in image pull status..."  
      kubectl delete pod $pod  
    done  
    sleep 300 # Wait for 5 minutes before checking again  
  done  
}  

# Function to delete a job if it exists
delete_if_exists() {
  job_name=$1
  namespace=$2
  if kubectl get rayjobs $job_name -n $namespace --ignore-not-found | grep -q $job_name; then
    echo "Deleting existing rayjob $job_name..."
    if ! kubectl delete rayjobs $job_name -n $namespace; then
      echo "Failed to delete existing rayjob $job_name."
      exit 1
    else
      echo "Existing rayjob $job_name deleted successfully."
    fi
  fi
}

# Function to check and delete any competing jobs
check_and_delete_competing_jobs() {
  local -a competing_job_names=("$@")
  namespace=$1
  shift # Remove the first argument, which is namespace
  for job_name in "${competing_job_names[@]}"; do
    delete_if_exists $job_name $namespace
  done
}

# Function to submit a job
submit_job() {
  job_file=$1
  namespace=$2
  kubectl apply -f $job_file -n $namespace
  if [ $? -ne 0 ]; then
    echo "Failed to submit job from $job_file."
    exit 1
  fi
}

# Function to check if a rayjob is completed without cleaning it up
check_job_status_without_cleanup() {
  job_name=$1
  namespace=$2
  while true; do
    deployment_status=$(kubectl get rayjobs $job_name -n $namespace -o jsonpath='{.status.jobDeploymentStatus}' --ignore-not-found)
    job_status=$(kubectl get rayjobs $job_name -n $namespace -o jsonpath='{.status.jobStatus}' --ignore-not-found)

    if [[ "$deployment_status" == "Complete" ]] && [[ "$job_status" == "SUCCEEDED" ]]; then
      echo "Rayjob $job_name completed successfully."
      break
    elif [[ "$deployment_status" == "Complete" ]] && [[ "$job_status" == "FAILED" ]]; then
      echo "Rayjob $job_name failed."
      kubectl delete rayjobs $job_name -n $namespace
      echo "Rayjob $job_name deleted."
      exit 1
    else
      echo "Waiting for rayjob $job_name to complete..."
      sleep 30
    fi
  done
}

# Function to run a job and follow up with its dependent job without deleting the pods in between
run_job_and_keep_pods_alive_for_follow_up() {
  job_file=$1
  job_name=$2
  follow_up_file=$3
  follow_up_name=$4
  namespace=$5

  delete_if_exists $job_name $namespace
  submit_job $job_file $namespace
  check_job_status_without_cleanup $job_name $namespace
  
  echo "Starting the follow-up job immediately..."
  delete_if_exists $follow_up_name $namespace
  submit_job $follow_up_file $namespace
  check_job_status_without_cleanup $follow_up_name $namespace
}

# Main execution starts here

# Define the namespace and jobs to check
namespace="default"
competing_jobs=("cpu-tuning-job" "gpu-tuning-job")

# Delete any competing jobs in the 'default' namespace
check_and_delete_competing_jobs $namespace "${competing_jobs[@]}"

# Start the pod deletion check in the background  
delete_stuck_pods &  
PID_STUCK_PODS=$!

# Run the tuning jobs in parallel, keeping the pods alive between jobs
run_job_and_keep_pods_alive_for_follow_up /app/my_jobs/ray_stats_tune.yaml cpu-tuning-job /app/my_jobs/ray_stats_pred.yaml cpu-predict-job $namespace &
PID_CPU=$!
run_job_and_keep_pods_alive_for_follow_up /app/my_jobs/ray_nn_tune.yaml gpu-tuning-job /app/my_jobs/ray_nn_pred.yaml gpu-predict-job $namespace &
PID_GPU=$!

# Wait for the follow-up jobs to complete before exiting
wait $PID_CPU
wait $PID_GPU
wait $PID_STUCK_PODS  

echo "All jobs have completed."


kubectl delete jobs --all
kubectl delete rayjobs --all