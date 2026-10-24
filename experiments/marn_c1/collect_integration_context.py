"""Read-only source collection for official matching and optimizer-boundary wiring."""
import ast
from pathlib import Path


def main():
    import rfdetr
    root = Path(rfdetr.__file__).resolve().parent
    pieces = []
    specs = {
        "training/module_model.py": {
            "RFDETRModelModule": {"__init__", "training_step", "_shared_step", "forward",
                                  "optimizer_step", "on_train_batch_end", "on_before_optimizer_step"},
        },
        "models/criterion.py": {"SetCriterion": {"forward"}},
        "models/matcher.py": {"HungarianMatcher": {"forward"}},
    }
    for rel, classes in specs.items():
        p = root / rel
        source = p.read_text(encoding="utf-8")
        for cls in ast.walk(ast.parse(source)):
            if isinstance(cls, ast.ClassDef) and cls.name in classes:
                methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
                pieces.append(f"\n=== {p} {cls.name} methods ===\n" + ", ".join(n.name for n in methods))
                for n in methods:
                    if n.name in classes[cls.name]:
                        pieces.append(f"\n--- {cls.name}.{n.name}:{n.lineno} ---\n" + ast.get_source_segment(source, n))
    out = Path(__file__).with_name("integration_context.txt")
    out.write_text("\n".join(pieces), encoding="utf-8")
    print("INTEGRATION_CONTEXT_SAVED", out)


if __name__ == "__main__":
    main()
