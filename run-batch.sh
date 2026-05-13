set -e

BATCH_SCRIPT=$(cat <<"EOF"
#!/usr/bin/env bash
#SBATCH --account C3SE2026-1-21 -p vera --time=0-12:00:00
#SBATCH -C ICELAKE|ZEN4
#SBATCH --array 0-39
#SBATCH --cpus-per-task 64

set -e

module load Rust/1.88.0-GCCcore-14.3.0

echo "Starting"
screen -d -m -S poisson_schottky bash -c "./target/release/examples/poisson_schottky sweep -N $SLURM_ARRAY_TASK_COUNT -n $SLURM_ARRAY_TASK_ID || read"
echo "Waiting"

while screen -wipe | grep -q "poisson_schottky" ; do
    sleep 1
done

echo "Done"
EOF
)

cargo build --release --example poisson_schottky
sbatch <(echo "$BATCH_SCRIPT")
watch -n 0.1 squeue -u loovj || true
squeue -u loovj
