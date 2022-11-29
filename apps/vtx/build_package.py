import os
import tvm
import json
import glob
import shutil
import pathlib
import tempfile
import subprocess
from pathlib import Path


def detect_shared_library_ext():
    from sys import platform

    if platform == "linux" or platform == "linux2":
        return ".so"
    elif platform == "darwin":
        return ".dylib"
    elif platform == "win32":
        raise NotImplementedError()


def generate_wheel(
    model_name, module_path, tvm_dir, out_dir, ctx
):
    print(f"Packaging {model_name} ...")

    # Create output directory if needed.
    build_dir = Path(tempfile.mkdtemp())
    if not build_dir.exists():
        os.makedirs(build_dir)

    # Remove existing package if its already there.
    if (build_dir / model_name).exists():
        shutil.rmtree(build_dir / model_name)

    # Create cookiecutter json configuration.
    dirname = pathlib.Path(os.path.dirname(__file__))
    template_dir = dirname / "python_template"
    template = template_dir.absolute()
    config_path = template / "cookiecutter.json"
    if ctx == "cpu":
        ctx_mod = "tvm.cpu()"
    elif ctx == "cuda":
        if tvm.cuda().exist:
            ctx_mod = "tvm.cuda()"
        elif tvm.rocm().exist:
            ctx_mod = "tvm.rocm()"
        else:
            raise ValueError("No compatible GPU found.")
    else:
        raise NotImplementedError("Only cpu and gpu are currently supported.")

    config = {
        "module_name": model_name,
        "package_name": model_name,
        "ctx": ctx_mod,
    }
    with open(config_path, "w") as cc_config:
        json.dump(config, cc_config)

    # Run cookiecutter to generate a custom build directory.
    subprocess.run(["cookiecutter", "--no-input", template, "--output-dir", build_dir])

    # Copy tvm into template for full bundling.
    module_dir = build_dir / model_name
    if not (module_dir / "tvm").exists():
        shutil.copytree(tvm_dir / "python" / "tvm", module_dir / "tvm")

    # Make sure we include the tvm runtime library.
    ext = detect_shared_library_ext()
    shutil.copy(tvm_dir / "build" / f"libtvm_runtime{ext}", module_dir / "tvm")

    model_pkg = build_dir / model_name / model_name

    shutil.copy(module_path, model_pkg)

    # Build the wheel.
    subprocess.run(["poetry", "build"], cwd=(build_dir / model_name))

    # Move the wheel to the specified output path.
    wheel_path = glob.glob(str(build_dir / model_name / "dist" / "*.whl"))[0]
    shutil.copy(wheel_path, out_dir)
    shutil.rmtree(build_dir)


generate_wheel("vortex_fp16", Path("packaged.so"), Path("/home/ubuntu/relax"), Path("packaged_model"), "cuda")