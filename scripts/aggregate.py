import os
import subprocess


MACHINE = os.environ.get("MACHINE", os.uname().nodename)


def main():
    subprocess.run(
        [
            "optimum-report",
            "gather",
            "--root-folders",
            f"dataset/{MACHINE}",
            "--save-file",
            f"dataset/{MACHINE}/perf-report.csv",
        ],
        capture_output=True,
    )


if __name__ == "__main__":
    main()
