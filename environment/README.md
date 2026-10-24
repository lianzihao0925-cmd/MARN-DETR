# Environment record

`installed_versions.txt` records installed distribution names and versions only.
It contains no credentials or direct package URLs. Recreate Python 3.11, install
the recorded CUDA 12.8 PyTorch and torchvision versions from PyTorch's official
index, then install this repository's `rf-detr/` in editable mode.

Experimental scripts record absolute AutoDL paths. When changing machines,
update the root path and caches deliberately; do not rerun completed jobs in
their old output directories. COCO images/annotations are obtained separately.
