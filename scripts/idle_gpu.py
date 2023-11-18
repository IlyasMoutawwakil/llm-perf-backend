import subprocess


def get_available_gpus():
    available_gpus = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    available_gpus = [int(gpu_id) for gpu_id in available_gpus if gpu_id != ""]

    return available_gpus


def get_pid_list(gpu_id: int):
    pid_list = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id={}".format(gpu_id),
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )
    pid_list = [int(pid) for pid in pid_list if pid != ""]

    return pid_list


def get_gpu_name(gpu_id: int):
    gpu_name = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id={}".format(gpu_id),
                "--query-gpu=gpu_name",
                "--format=csv,noheader",
            ]
        )
        .decode("utf-8")
        .strip()
    )

    return gpu_name


def get_idle_gpus():
    idle_gpus = []
    available_gpus = get_available_gpus()

    for gpu_id in available_gpus:
        pid_list = get_pid_list(gpu_id)
        gpu_name = get_gpu_name(gpu_id)

        if len(pid_list) == 0 and gpu_name != "NVIDIA DGX Display":
            idle_gpus.append(gpu_id)

    return idle_gpus


if __name__ == "__main__":
    idle_gpus = get_idle_gpus()
    assert len(idle_gpus) > 0, "No idle GPUs found!"
    print(idle_gpus[0])
