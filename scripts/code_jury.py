"""Mandatory code validation before commit: syntax, import, forward, loss, backward.

Usage:
    python scripts/code_jury.py --task flu --env-dir env \\
        --train-py env/train.py --input-shape "(4, 5, 1)" \\
        --expected-output-shape "(4, 10)" \\
        --out experiments/loop-flu-YYMMDD-HHMM/iterations/iter-4-jury.json
"""

import argparse
import importlib
import inspect
import py_compile
import sys
from pathlib import Path

from pipeline_utils import ensure_parent_dir, parse_shape, utc_now_iso, write_json

COMMON_MODEL_NAMES = [
    "get_model", "build_model", "create_model", "Model",
    "Net", "GRUSeq2Seq", "CNN", "Classifier", "DiffusionForecaster",
    "LSTMSeq2Seq", "TinyTransformer", "TCNForecaster",
]


def check_syntax(train_py: str) -> dict:
    try:
        py_compile.compile(train_py, doraise=True)
        return {"status": "pass"}
    except py_compile.PyCompileError as e:
        return {"status": "fail", "error": str(e)}


def check_import(env_dir: str, train_py: str) -> dict:
    sys.path.insert(0, str(Path(env_dir).resolve()))
    module_name = Path(train_py).stem
    try:
        mod = importlib.import_module(module_name)
        return {"status": "pass", "module": module_name}
    except Exception as e:
        return {"status": "fail", "error": str(e)}


def discover_model(mod, model_factory: str | None) -> tuple:
    if model_factory:
        factory = getattr(mod, model_factory, None)
        if factory is None:
            return None, f"Specified --model-factory '{model_factory}' not found in module"
        if callable(factory):
            try:
                instance = factory()
                return instance, type(instance).__name__
            except Exception as e:
                return None, f"Factory '{model_factory}' raised: {e}"
        return None, f"'{model_factory}' is not callable"

    for name in COMMON_MODEL_NAMES:
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if inspect.isclass(obj):
            try:
                instance = obj()
                return instance, name
            except Exception:
                continue
        if callable(obj):
            try:
                instance = obj()
                return instance, name
            except Exception:
                continue
    return None, "No model found. Pass --model-factory with a callable function/class name."


def main():
    parser = argparse.ArgumentParser(description="Code Jury — mandatory pre-commit validation")
    parser.add_argument("--task", required=True, choices=["flu"])
    parser.add_argument("--env-dir", required=True)
    parser.add_argument("--train-py", required=True)
    parser.add_argument("--input-shape", required=True)
    parser.add_argument("--expected-output-shape", default=None)
    parser.add_argument("--model-factory", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    checks = {}
    errors = []
    overall_status = "pass"

    # 1. Syntax
    r = check_syntax(args.train_py)
    checks["syntax"] = r
    if r["status"] != "pass":
        errors.append(r["error"])

    # 2. Import
    if r["status"] == "pass":
        r2 = check_import(args.env_dir, args.train_py)
        checks["import"] = r2
        if r2["status"] != "pass":
            errors.append(r2["error"])
    else:
        checks["import"] = {"status": "skip", "reason": "syntax failed"}

    # 3. Instantiate
    mod = None
    if checks.get("import", {}).get("status") == "pass":
        module_name = Path(args.train_py).stem
        mod = importlib.import_module(module_name)
        model, class_name = discover_model(mod, args.model_factory)
        if model is None:
            checks["instantiate"] = {"status": "fail", "error": class_name}
            errors.append(class_name)
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.train()
            checks["instantiate"] = {"status": "pass", "model_class": class_name}
    else:
        checks["instantiate"] = {"status": "skip", "reason": "import failed"}
        model = None

    # 4. Forward
    if model is not None:
        try:
            input_shape = parse_shape(args.input_shape)
            x = torch.randn(*input_shape, device=device)
            if args.task == "flu":
                input_shape = parse_shape(args.input_shape)
                x_tau = torch.randn(input_shape[0], 10, 1, device=device)
                c = torch.randn(input_shape[0], 5, 1, device=device)
                tau = torch.randint(0, 30, (input_shape[0],), device=device)
                if hasattr(model, "denoiser"):
                    out = model.denoiser(x_tau, c, tau)
                elif hasattr(model, "forward"):
                    out = model(c)
                else:
                    out = model(c)
            else:
                out = model(x)
            out_shape = list(out.shape)
            check = {"status": "pass", "output_shape": out_shape}
            if args.expected_output_shape:
                expected = list(parse_shape(args.expected_output_shape))
                if out_shape != expected:
                    squeezed = [d for d in out_shape if d != 1]
                    if squeezed != expected and out_shape != [d for d in expected if d != 1]:
                        check["status"] = "fail"
                        check["error"] = f"Expected {expected}, got {out_shape}"
                        errors.append(check["error"])
            checks["forward"] = check
        except Exception as e:
            checks["forward"] = {"status": "fail", "error": str(e)}
            errors.append(str(e))
    else:
        checks["forward"] = {"status": "skip", "reason": "model not instantiated"}

    # 5. Loss + backward
    if checks.get("forward", {}).get("status") == "pass":
        try:
            if args.task == "medmnist":
                B = input_shape[0]
                target = torch.randint(0, 3, (B,), device=device)
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(out, target)
            else:
                target = torch.randn(*out.shape, device=device)
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(out, target)

            if not torch.isfinite(loss):
                checks["loss"] = {"status": "fail", "error": "loss is not finite"}
                errors.append("loss is not finite")
            else:
                checks["loss"] = {"status": "pass", "value": round(float(loss), 6)}

                loss.backward()
                params_with_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all().item())
                if params_with_grad == 0 and sum(1 for p in model.parameters() if p.requires_grad) > 0:
                    checks["backward"] = {"status": "fail", "error": "no trainable params have gradients"}
                    errors.append("backward produced no gradients")
                else:
                    checks["backward"] = {"status": "pass", "params_with_grad": params_with_grad}
        except Exception as e:
            checks["loss"] = {"status": "fail", "error": str(e)}
            errors.append(str(e))
    else:
        checks["loss"] = {"status": "skip", "reason": "forward failed"}
        checks["backward"] = {"status": "skip", "reason": "forward failed"}

    for c in checks.values():
        if c.get("status") == "fail":
            overall_status = "fail"

    report = {
        "task": args.task,
        "timestamp": utc_now_iso(),
        "status": overall_status,
        "checks": checks,
        "errors": errors,
    }

    write_json(args.out, report)
    print(f"Jury status: {overall_status.upper()}")
    for name, c in checks.items():
        print(f"  {name}: {c['status']}")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)

    if overall_status == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
