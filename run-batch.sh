set -e

cargo build --release --example poisson_schottky

N_JOBS=40

BATCH_SCRIPT=$(cat <<"EOF"
#!/usr/bin/env bash
#SBATCH --account C3SE2026-1-21 -p vera --time=0-12:00:00
#SBATCH -C ICELAKE|ZEN4
#SBATCH --cpus-per-task 64

set -e

module load Rust/1.88.0-GCCcore-14.3.0

echo "Starting"
screen -d -m -S poisson_schottky bash -c "./target/release/examples/poisson_schottky $* sweep -N $SLURM_ARRAY_TASK_COUNT -n $SLURM_ARRAY_TASK_ID || read"
echo "Waiting"

while screen -wipe | grep -q "poisson_schottky" ; do
    sleep 1
done

echo "Done"
EOF
)

case "$1" in
    AB-0.44 )
        shift
        OPTIONS="--anode-area-um2 0.44 --nd-active-per-cm3 5.0e17 --length-active-nm 48 --nd-buffer-per-cm3 5.0e18 --length-buffer-nm 250 $*"
    ;;
    AB-3.2 )
        shift
        OPTIONS="--anode-area-um2 3.2 --nd-active-per-cm3 5.0e17 --length-active-nm 48 --nd-buffer-per-cm3 5.0e18 --length-buffer-nm 250 $*"
    ;;
    AB-3.2-large )
        shift
        OPTIONS="--anode-area-um2 3.2 --nd-active-per-cm3 5.0e17 --length-active-nm 48 --nd-buffer-per-cm3 5.0e19 --length-buffer-nm 2500 $*"
    ;;
    AI-0.8 )
        shift
        OPTIONS="--anode-area-um2 0.8 --nd-active-per-cm3 3.0e17 --length-active-nm 60 --nd-buffer-per-cm3 5.0e18 --length-buffer-nm 250 $*"
    ;;
    * )
        echo "* No diode kind detected as first argument. Passing options"
        OPTIONS="$*"
    ;;
esac

sbatch --array 0-$((N_JOBS-1)) <(echo "$BATCH_SCRIPT") "$OPTIONS"
watch -n 0.1 squeue -u loovj || true
squeue -u loovj
