import subprocess


def main():
    for seed in range(1, 11):
        subprocess.Popen([f'sbatch slurm_submit.wilkes3 {seed}'], shell=True)


if __name__ == '__main__':
    main()
