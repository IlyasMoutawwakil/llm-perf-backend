import os
import subprocess


def main():
    machine = os.uname().nodename
    out = subprocess.run(
        [
            "optimum-report",
            "gather",
            "--root-folders",
            f"dataset/{machine}",
            "--save-file",
            f"dataset/{machine}/perf-report.csv",
        ],
        capture_output=True,
    )

    if out.returncode == 0:
        print(">Report generation succeeded")
    else:
        raise Exception("Report generation failed")


if __name__ == "__main__":
    main()
