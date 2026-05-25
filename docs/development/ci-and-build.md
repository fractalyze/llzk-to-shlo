# CI and build

## Prerequisites

- LLVM/MLIR 20.x (`llvm-20-dev`, `libmlir-20-dev` on Debian/Ubuntu), or provided
  via the Nix flake (see below).
- Bazel 7.x (install via bazelisk).
- clang-20 / clang++-20.

Create `.bazelrc.user` in the project root with your local toolchain paths:

```bash
# .bazelrc.user — adjust paths to match your environment
build --action_env=CC=/usr/bin/clang-20
build --action_env=CXX=/usr/bin/clang++-20
build --action_env=LOCAL_CUDA_PATH=/usr/local/cuda-12.9
build --features=-header_modules --features=-module_maps
build --host_features=-header_modules --host_features=-module_maps
```

Then build and test:

```bash
bazel build //tools:llzk-to-shlo-opt
bazel test //...
```

## circom from the nix flake

circom is built via the Nix flake in the project repository. The circom ELF
binary is linked against a Nix-store glibc and resolves MLIR 20 shared libraries
from the flake's dev closure — NOT from the system LLVM package. This means the
nix dev shell is required for any use of the LLZK pipeline that involves circom:
without it, `circom` either fails to start (dynamic linker mismatch) or uses a
system MLIR version that mismatches the project's pinned dialect ABI.

Concretely: any `circom_to_llzk` Bazel genrule, any `//examples/...` target, and
any end-to-end test that compiles `.circom` source will fail to build or produce
garbage IR outside the nix dev shell. This is not a workaround — it is
load-bearing for any LLZK pipeline reproduction.

## Runner setup

Install Nix:

```bash
curl --proto '=https' --tlsv1.2 -sSf -L \
  https://install.determinate.systems/nix | sh -s -- install
```

If the root filesystem has limited space, bind-mount `/nix` to a larger disk
before installing:

```bash
sudo mkdir -p /data/$USER/nix /nix
sudo mount --bind /data/$USER/nix /nix
echo "/data/$USER/nix /nix none bind 0 0" | sudo tee -a /etc/fstab
# Then run the Nix installer above.
```

Build and install circom from the `llzk` branch of
[project-llzk/circom](https://github.com/project-llzk/circom):

```bash
git clone -b llzk https://github.com/project-llzk/circom.git /opt/circom-src
cd /opt/circom-src
nix build
sudo cp result/bin/circom /usr/local/bin/circom
circom --version
```

To update circom:
`git pull && nix build && sudo cp result/bin/circom /usr/local/bin/circom`.

Verify the full environment:

```bash
clang-20 --version
bazel --version
circom --version
bazel --bazelrc=.bazelrc.ci test --config=ci -- //... -//examples/...
```

### What is excluded from CI

- `//examples/...` — all targets depend on circom genrules that may fail on
  incomplete LLZK support.
- Tests tagged `circom` — require the circom E2E pipeline.
- Tests tagged `gpu` — require `stablehlo_runner` and GPU hardware.
- Tests tagged `manual` — full E2E regression; run locally with
  `bazel test //tests:batch_e2e_tests --test_tag_filters=manual`.

## Build mode and asserts

The repo default is `-c opt` (NDEBUG), which compiles out all `assert()` calls.
Pass invariants and internal consistency checks in the lowering passes rely on
`assert` for debug-time verification — they are silently absent in the default
build. When chasing a crash or verifier failure, always rebuild with `-c dbg`
before drawing any structural conclusion:

```bash
bazel build -c dbg //tools:llzk-to-shlo-opt
bazel test -c dbg //tests:lit_tests
```

The debug binary lives at `bazel-out/k8-dbg/bin/tools/llzk-to-shlo-opt`. Note
that the project occasionally ships ASan-instrumented dbg configs; if a
pre-existing dbg binary fails to start with an ASLR conflict message, force a
fresh build rather than chasing `setarch -R` workarounds.
