"""Guarded, backed-up source edit. Run with the server's marn-detr Python."""
import ast
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

MARK = "MARN_C2_V1"


def one(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one source anchor in {label}, found {count}; no files changed")
    return source.replace(old, new, 1)


def in_method(source, class_name, method_name, old, new, label):
    """Edit only the named method; similar inference/export paths stay untouched."""
    tree = ast.parse(source, filename=label)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise RuntimeError(f"Expected one class {class_name} in {label}; no files changed")
    methods = [n for n in classes[0].body
               if isinstance(n, ast.FunctionDef) and n.name == method_name]
    if len(methods) != 1:
        raise RuntimeError(f"Expected one method {class_name}.{method_name}; no files changed")
    node = methods[0]
    lines = source.splitlines(keepends=True)
    start, end = node.lineno - 1, node.end_lineno
    section = "".join(lines[start:end])
    # A method at EOF may have no trailing newline. Include it only for matching,
    # then restore the original EOF convention after the edit.
    missing_newline = not section.endswith("\n")
    if missing_newline:
        section += "\n"
    section = one(section, old, new, f"{label}:{class_name}.{method_name}")
    if missing_newline:
        section = section.removesuffix("\n")
    result = "".join(lines[:start]) + section + "".join(lines[end:])
    ast.parse(result, filename=label)
    return result


def main():
    import rfdetr
    root = Path(rfdetr.__file__).resolve().parent
    bundle = Path(__file__).resolve().parent
    edits = {}
    paths = [root / rel for rel in (
        "models/transformer.py", "models/lwdetr.py", "models/criterion.py"
    )]
    originals = {p: p.read_text(encoding="utf-8") for p in paths}
    if any(MARK in s for s in originals.values()) or (root / "marn").exists():
        raise RuntimeError("C2 markers/package already exist; refuse to overwrite. Run the checks instead.")

    p = paths[0]
    source = originals[p]
    anchor = "                intermediate.append(self.norm(output))\n"
    insertion = '''
            # MARN_C2_V1: after native ref update + auxiliary capture; before the next layer.
            marn = getattr(self, "marn", None)
            if marn is not None and marn.enabled and layer_id == marn.config.layer:
                if not self.return_intermediate or self._export or self.enable_keypoint_processing:
                    raise RuntimeError("MARN C2 requires eager detection with intermediate outputs")
                output = marn(
                    output, intermediate[-1], hs_refpoints_unsigmoid[layer_id],
                    self.__dict__["_marn_heads"], self.bbox_reparam,
                )
'''
    edits[p] = in_method(source, "TransformerDecoder", "forward", anchor, anchor + insertion, str(p))

    p = paths[1]
    anchor = "        transformer_outputs = self.transformer(\n"
    insertion = '''        # MARN_C2_V1: transient head references, supplied by this model instance.
        if getattr(self.transformer.decoder, "marn", None) is not None:
            self.transformer.decoder.__dict__["_marn_heads"] = (self.class_embed, self.bbox_embed)

'''
    edits[p] = in_method(originals[p], "LWDETR", "forward", anchor, insertion + anchor, str(p))

    p = paths[2]
    anchor = "        if num_boxes is None:\n            num_boxes = self.num_boxes_for_targets(outputs, targets)\n"
    insertion = '''        # MARN_C2_V1: observe existing assignments without changing matcher/loss paths.
        observer = getattr(self, "marn_observer", None)
        if self.training and observer is not None:
            observer(matched_outputs, all_indices, layer_suffixes, targets)

'''
    edits[p] = in_method(originals[p], "SetCriterion", "forward", anchor, insertion + anchor, str(p))

    for p, source in edits.items():
        ast.parse(source, filename=str(p))
    for name in ("core.py", "integration.py"):
        ast.parse((bundle / name).read_text(encoding="utf-8"))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = root.parents[2] / "runs/source_backups" / ("marn_c2_" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    for p in paths:
        dest = backup / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    manifest = {str(p): {"before_sha256": hashlib.sha256(originals[p].encode()).hexdigest(),
                         "after_sha256": hashlib.sha256(edits[p].encode()).hexdigest()}
                for p in paths}
    (backup / "manifest.json").write_text(json.dumps(manifest, indent=2))
    package = root / "marn"
    try:
        package.mkdir()
        (package / "__init__.py").write_text('"""MARN-DETR experimental modules."""\n')
        for name in ("core.py", "integration.py"):
            shutil.copy2(bundle / name, package / name)
        for p, source in edits.items():
            temp = p.with_suffix(".marn_tmp")
            temp.write_text(source, encoding="utf-8")
            temp.replace(p)
    except BaseException:
        for p in paths:
            shutil.copy2(backup / p.relative_to(root), p)
        if package.exists():
            shutil.rmtree(package)
        raise
    print("C2_INSTALLED", root)
    print("SOURCE_BACKUP", backup)
    print("No training started. Next: python -u check_c2.py")


if __name__ == "__main__":
    main()
