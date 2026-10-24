# Artifact and source index

| File | Purpose | Retention |
|---|---|---|
| manifest.json | 99 original artifact paths, sizes, SHA256 and Release URLs; consumed by tools/restore_weights.py | Keep with this archive |
| FILES.tsv | Readable mapping of original paths to hashed Release asset names, including test/formal labels | Keep with this archive |
| source_manifest.json | SHA256 and size of each of the 951 copied server code/report files | Keep for integrity checks |
| upstream.json | Original RF-DETR remote, commit, branch and working-tree status | Keep for provenance |
| working_tree.patch | Git diff for tracked upstream files; new MARN files are present in rf-detr/src/rfdetr/marn/ | Keep; not a standalone installer |
| published_assets.json | Generated after all uploads: GitHub asset IDs, sizes and SHA256 digests | Keep for remote verification |

`.ckpt` files preserve full training state where produced by the experiment;
`.pth` files are stage-specific model exports. Read the corresponding loader
before use. Labels mark short test/probe files so they are not confused with
trained experimental checkpoints. No weights are removed from the server by
this backup. Restoring the entire archive downloads approximately 21.2 GiB.

Original project root: `/root/autodl-tmp/project_marn-detr`.
The one external official weight is mapped to `pretrained/rf-detr-small.pth`.
All other paths are relative to the original project root. Release names are
content hashes, so this index and manifest are needed to recover original names.
