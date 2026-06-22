# Cluster Training Setup

## Quick Start

1. Copy your SSH key into this directory:
   ```bash
   cp ~/.ssh/id_rsa cluster/
   chmod 600 cluster/id_rsa
   ```

2. Fill in `config.yaml` with your cluster details.

3. Test connection:
   ```bash
   ssh -i cluster/id_rsa -p 22 user@login.cluster.edu
   ```

4. Sync code and run:
   ```bash
   rsync -avz --exclude '.git' --exclude '.venv' --exclude 'data' ./ user@host:~/MLSS26_HACKATHON/
   ssh user@host "cd MLSS26_HACKATHON && apptainer exec --nv pytorch.sif python scripts/run_exp.py --epochs 50"
   ```

## Files

- `config.yaml` — cluster host, paths, container, scheduler settings
- `id_rsa` — SSH private key (gitignored, add manually)
- `*.sif` — Singularity container images (gitignored, add manually)

## Security

SSH keys and `.sif` files are gitignored (see `../.gitignore`). Add them manually after cloning.
