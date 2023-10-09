for config in llm-configs/*.yaml; do
    # first, extract experiment name
    experiment_name=$(basename $config .yaml)
    
    # if the inference results already exist, skip
    if [ -f "llm-experiments/Succeeded-1xA100-80GB/$experiment_name/experiment.log" ]; then
        echo "Skipping $experiment_name because llm-experiments/Succeeded-1xA100-80GB/$experiment_name/experiment.log already exists"
        continue
        elif [ -f "llm-experiments/Failed-1xA100-80GB/$experiment_name/experiment.log" ]; then
        echo "Skipping $experiment_name because llm-experiments/Failed-1xA100-80GB/$experiment_name/experiment.log already exists"
        continue
        elif [ -f "llm-experiments/OOMed-1xA100-80GB/$experiment_name/experiment.log" ]; then
        echo "Skipping $experiment_name because llm-experiments/OOMed-1xA100-80GB/$experiment_name/experiment.log already exists"
        continue
        elif [ -f "llm-experiments/$experiment_name/inference_results.csv" ]; then
        echo "Skipping $experiment_name because llm-experiments/$experiment_name/inference_results.csv already exists"
    fi
    
    # check which GPU is free for running the experiment
    if [ "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv | wc -l)" -eq 1 ]; then
        cuda_device=0
        elif [ "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv | wc -l)" -eq 1 ]; then
        cuda_device=1
        elif [ "$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv | wc -l)" -eq 1 ]; then
        cuda_device=2
        elif [ "$(nvidia-smi -i 3 --query-compute-apps=pid --format=csv | wc -l)" -eq 1 ]; then
        cuda_device=3
    else
        echo "No free GPU found, skipping $experiment_name"
        continue
    fi
    
    # announce the experiment
    echo "Running $experiment_name on GPU $cuda_device"
    # create the directory if it doesn't exist
    mkdir -p llm-experiments/$experiment_name
    # run the experiment forcing the isolation
    CUDA_VISIBLE_DEVICES=$cuda_device optimum-benchmark --config-dir llm-configs --config-name $experiment_name | tee llm-experiments/$experiment_name/full.log
    
    # check if the inference results exist
    if [ -f "llm-experiments/$experiment_name/inference_results.csv" ]; then
        # if yes then move the results folder to the llm-experiments/Succeeded-1xA100-80GB/experiment_path
        echo "Moving llm-experiments/$experiment_name to llm-experiments/Succeeded-1xA100-80GB/$experiment_name"
        # create the directory if it doesn't exist
        mkdir -p llm-experiments/Succeeded-1xA100-80GB/
        mv llm-experiments/$experiment_name llm-experiments/Succeeded-1xA100-80GB/$experiment_name
        elif [ -f "llm-experiments/$experiment_name/experiment.log" ]; then
        # if "CUDA out of memory" is found in the log file, move the results folder to the llm-experiments/OOMed-1xA100-80GB/experiment_path
        if grep -q "CUDA out of memory" "llm-experiments/$experiment_name/experiment.log"; then
            echo "Moving llm-experiments/$experiment_name to llm-experiments/OOMed-1xA100-80GB/$experiment_name"
            # create the directory if it doesn't exist
            mkdir -p llm-experiments/OOMed-1xA100-80GB/
            mv llm-experiments/$experiment_name llm-experiments/OOMed-1xA100-80GB/$experiment_name
        else
            echo "Moving llm-experiments/$experiment_name to llm-experiments/Failed-1xA100-80GB/$experiment_name"
            # create the directory if it doesn't exist
            mkdir -p llm-experiments/Failed-1xA100-80GB/
            mv llm-experiments/$experiment_name llm-experiments/Failed-1xA100-80GB/$experiment_name
        fi
    fi
done
