# CI Runner Setup Guide

This guide covers setting up a self-hosted CI runner for the llzk-to-shlo
project.

## Prerequisites

- Ubuntu 22.04+ (x86_64)
- clang-20 / clang++-20
- Bazel 7.4.1 (via bazelisk)
- sudo access

## 1. Install Nix

circom requires [Nix](https://nixos.org/) to build (LLVM 20 MLIR dev libraries
are managed via Nix flakes).

```bash
curl --proto '=https' --tlsv1.2 -sSf -L \
  https://install.determinate.systems/nix | sh -s -- install
```

### Optional: Store Nix artifacts on a separate disk

If the root filesystem has limited space, bind-mount `/nix` to a larger disk
**before** installing Nix:

```bash
# Prepare mount point (replace /data/$USER with your SSD path)
sudo mkdir -p /data/$USER/nix /nix
sudo mount --bind /data/$USER/nix /nix

# Persist across reboots
echo "/data/$USER/nix /nix none bind 0 0" | sudo tee -a /etc/fstab

# Then install Nix
curl --proto '=https' --tlsv1.2 -sSf -L \
  https://install.determinate.systems/nix | sh -s -- install
```

## 2. Build and install circom

circom is the Circom-to-LLZK compiler from
[project-llzk/circom](https://github.com/project-llzk/circom) (llzk branch). It
depends on LLVM 20 MLIR, which Nix provides.

```bash
git clone -b llzk https://github.com/project-llzk/circom.git /opt/circom-src
cd /opt/circom-src

# Build using Nix flake (downloads LLVM 20 automatically)
nix build

# Install to system PATH
sudo cp result/bin/circom /usr/local/bin/circom

# Verify
circom --version
```

### Updating circom

```bash
cd /opt/circom-src
git pull
nix build
sudo cp result/bin/circom /usr/local/bin/circom
```

## 3. Verify CI environment

```bash
# Required tools
clang-20 --version
bazel --version
circom --version

# Run the CI test suite
cd /path/to/llzk-to-shlo
bazel --bazelrc=.bazelrc.ci test --config=ci -- //... -//examples/...
```

## CI workflow

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs two jobs:

| Job                | Runner             | What it does                    |
| ------------------ | ------------------ | ------------------------------- |
| `build-and-test`   | `self-hosted, cpu` | `bazel test` with `--config=ci` |
| `pre-commit-style` | `self-hosted, cpu` | pre-commit on changed files     |

### What is excluded from CI

- `//examples/...` -- all targets depend on circom genrules that may fail on
  incomplete LLZK support
- Tests tagged `circom` -- require circom E2E pipeline
- Tests tagged `gpu` -- require stablehlo_runner + GPU hardware
