# AlphaFast1 import provenance

This branch captures the source tree used for the native NVIDIA reproduction on
the Inspire platform. It is intentionally based on the exact upstream commit
below, with the previously uncommitted AlphaFast1 implementation layered on top
as a normal Git commit.

## Source

- Imported: 2026-08-25
- Upstream: `https://github.com/RomeroLab/alphafast.git`
- Base commit: `4f3905e679b839e178d16249bad4e9c40b07332b`
- Import source: `/inspire/hdd/project/aisystem-and-infra/26010/alphafast1`
- Import target: `/inspire/hdd/project/aisystem-and-infra/26010/alphafast`
- Target branch: `import/alphafast1-4f3905e`
- Original tracked patch SHA-256: `9aa2457cbde792bc78db6047ea6e9ba9326f376d96640dd6e4018da38e240c76`
- Active launcher SHA-256: `3e6fb2a1309fef90df171ee1d5a03ee842efd3d8d41d7319fdbf0a98d8fb8653`

The import contains all 26 tracked changes from the source worktree, including
18 executable-mode changes, plus the local launch/recovery scripts, archived
launcher versions, package `__init__.py` shims, Python constant sources, and the
2PV7 example input. The imported file contents were compared byte-for-byte with
the source worktree before commit.

## Deliberately excluded generated state

The following were not added to Git:

- `.codex` and `.sii/` IDE/runtime state.
- `examples/output/` and `examples/output_multigpus/` generated predictions.
- Generated `ccd.pickle` and `chemical_component_sets.pickle` files, including
  the duplicate copies under `constants/constants/converters/`.

The generated artifacts used by the source worktree had these hashes:

- `src/alphafold3/constants/converters/ccd.pickle`:
  `8aaafefa9080f1f84abfcd1546d8883c86e002d167de242cc0b196557c1f8394`
- `src/alphafold3/constants/converters/chemical_component_sets.pickle`:
  `3fd6edb59838889ede3791e19bc3bec0908e5ae1bf8d25d10f9d0463ce346572`

These large generated artifacts remain outside Git and should be regenerated or
managed as versioned data rather than committed directly. Pre-existing untracked
files in the target worktree, including `resource_info.md`, were preserved and
were not included in this import.

## Reproduction evidence

The imported source version completed the `full` NVIDIA profile with three
cases, one seed (101), five samples per case, and validator exit code 0. The
runtime output and metadata were recorded under:

```text
/inspire/hdd/project/aisystem-and-infra/26010/af3_gpu_refactor_benchmark/reproduction/nvidia_native_full_seed1
/inspire/hdd/project/aisystem-and-infra/26010/af3_gpu_refactor_benchmark/reproduction/nvidia_native_full_seed1.run_meta
```
