import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"


def write_compat_pyconfig():
    """Rewrite a pyconfig.h that defaults Py_GIL_DISABLED=1 on a GIL build.

    Python.h uses #include \"pyconfig.h\", which prefers the header next to
    itself, so we remap that name with MSVC pragma include_alias + /FI.
    """
    import sysconfig

    src = Path(sysconfig.get_path("include")) / "pyconfig.h"
    if not src.exists():
        return None
    text = src.read_text(encoding="utf-8", errors="replace")
    needle_lf = (
        "#ifndef Py_GIL_DISABLED\n"
        "#define Py_GIL_DISABLED 1\n"
        "#endif"
    )
    needle_crlf = needle_lf.replace("\n", "\r\n")
    if needle_lf not in text and needle_crlf not in text:
        return None
    dest_dir = CSRC / "compat"
    dest_dir.mkdir(parents=True, exist_ok=True)
    patched = text.replace(needle_crlf, needle_crlf.replace("#define Py_GIL_DISABLED 1\r\n", "/* GIL build */\r\n"), 1)
    patched = patched.replace(needle_lf, needle_lf.replace("#define Py_GIL_DISABLED 1\n", "/* GIL build */\n"), 1)
    (dest_dir / "tvarant_pyconfig.h").write_text(patched, encoding="utf-8")
    (dest_dir / "tvarant_force_include.h").write_text(
        "#pragma once\n"
        '#pragma include_alias("pyconfig.h", "tvarant_pyconfig.h")\n'
        "#pragma include_alias(<pyconfig.h>, <tvarant_pyconfig.h>)\n",
        encoding="utf-8",
    )
    print("Patched pyconfig.h to keep the GIL-enabled Python ABI")
    return str(dest_dir)


def ensure_msvc_python_t_lib():
    """This CPython's pyconfig.h #defines Py_GIL_DISABLED and then
    `#pragma comment(lib, \"pythonXXXt.lib\")`. Regular installs only ship
    pythonXXX.lib, so provide a same-ABI stub name for the linker."""
    if sys.platform != "win32":
        return None
    import shutil

    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    src = Path(sys.base_prefix) / "libs" / f"python{ver}.lib"
    dest_dir = ROOT / "third_party" / "pylibs"
    dest = dest_dir / f"python{ver}t.lib"
    if not src.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
        shutil.copy2(src, dest)
    return str(dest_dir)


def find_local_winsdk():
    """MSVC on this machine has no system Windows SDK; use vendored NuGet headers/libs."""
    root = ROOT / "third_party" / "winsdk"
    inc_root = root / "sdk.cpp" / "c" / "Include" / "10.0.26100.0"
    lib_root = root / "sdk.x64" / "c"
    ucrt_inc = inc_root / "ucrt"
    um_inc = inc_root / "um"
    shared_inc = inc_root / "shared"
    ucrt_lib = lib_root / "ucrt" / "x64"
    um_lib = lib_root / "um" / "x64"
    if not ucrt_inc.exists() or not ucrt_lib.exists():
        return None
    return {
        "include": [str(ucrt_inc), str(um_inc), str(shared_inc)],
        "lib": [str(ucrt_lib), str(um_lib)],
    }


def inject_winsdk_env(sdk):
    include = os.pathsep.join(sdk["include"])
    lib = os.pathsep.join(sdk["lib"])
    os.environ["INCLUDE"] = include + os.pathsep + os.environ.get("INCLUDE", "")
    os.environ["LIB"] = lib + os.pathsep + os.environ.get("LIB", "")
    os.environ["LIBPATH"] = lib + os.pathsep + os.environ.get("LIBPATH", "")
    print("Using local Windows SDK headers from third_party/winsdk")


def find_opencl():
    """Return (include_dir, library_dir, libname) or None."""
    roots = []
    for key in ("OPENCL_ROOT", "CUDA_PATH", "INTELOCLSDKROOT"):
        val = os.environ.get(key)
        if val:
            roots.append(Path(val))
    roots.extend(
        [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"),
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"),
            Path(r"C:\Program Files (x86)\Intel\OpenCL SDK"),
            Path("/usr/local/cuda"),
            Path("/usr"),
        ]
    )
    for root in roots:
        inc = root / "include"
        if not (inc / "CL" / "cl.h").exists():
            continue
        for libdir in (root / "lib" / "x64", root / "lib64", root / "lib"):
            if libdir.exists():
                return str(inc), str(libdir), "OpenCL"
    if (Path("/usr/include/CL/cl.h")).exists():
        return "/usr/include", "/usr/lib", "OpenCL"
    return None


def main():
    sources = [
        "csrc/Module.cpp",
        "csrc/runtime/Runtime.cpp",
        "csrc/runtime/CpuSimRuntime.cpp",
        "csrc/runtime/OpenCLRuntime.cpp",
        "csrc/runtime/Allocator.cpp",
        "csrc/runtime/Guard.cpp",
        "csrc/runtime/Hooks.cpp",
        "csrc/runtime/Generator.cpp",
        "csrc/aten/Ops.cpp",
        "csrc/aten/ExtendedOps.cpp",
        "csrc/aten/LlmOps.cpp",
        "csrc/aten/CustomOps.cpp",
        "csrc/jit/Jit.cpp",
        "csrc/kernels/host/HostKernels.cpp",
    ]

    include_dirs = [str(CSRC), str(CSRC / "runtime"), str(CSRC / "kernels" / "host"), str(CSRC / "jit"), str(CSRC / "aten")]
    compat = write_compat_pyconfig()
    if compat:
        include_dirs.insert(0, compat)
    define_macros = []
    libraries = []
    library_dirs = []
    extra_compile_args = []
    extra_link_args = []

    sdk = find_local_winsdk()
    if sdk:
        inject_winsdk_env(sdk)
        include_dirs.extend(sdk["include"])
        library_dirs.extend(sdk["lib"])

    pylibs = ensure_msvc_python_t_lib()
    if pylibs:
        library_dirs.append(pylibs)

    if sys.platform == "win32":
        extra_compile_args = [
            "/std:c++17",
            "/EHsc",
            "/O2",
            "/permissive-",
            "/Zc:__cplusplus",
            "/DNOMINMAX",
            "/D_CRT_SECURE_NO_WARNINGS",
            "/wd4251",
            "/wd4275",
        ]
        extra_link_args = ["/MANIFEST:NO"]
        fi = CSRC / "compat" / "tvarant_force_include.h"
        if fi.exists():
            extra_compile_args.append(f"/FI{fi}")
    else:
        extra_compile_args = ["-std=c++17", "-fPIC", "-O3", "-Wno-unused-parameter"]

    force_opencl = os.environ.get("USE_OPENCL", "").lower() in {"1", "true", "on", "yes"}
    if force_opencl:
        ocl = find_opencl()
        if ocl is None:
            raise RuntimeError("USE_OPENCL=1 but OpenCL headers were not found")
        inc, libdir, libname = ocl
        include_dirs.append(inc)
        library_dirs.append(libdir)
        libraries.append(libname)
        define_macros.append(("USE_OPENCL", "1"))
        print(f"Building torch_tvarant with OpenCL ({inc})")
    else:
        print("Building torch_tvarant CPU simulator (set USE_OPENCL=1 to enable OpenCL)")

    ext = CppExtension(
        name="torch_tvarant._C",
        sources=sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    setup(
        name="torch-tvarant",
        version="0.1.0",
        description="PyTorch device backend for the Tvarant RISC-V SIMT GPGPU",
        packages=find_packages(include=["torch_tvarant", "torch_tvarant.*"]),
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
        zip_safe=False,
    )


main()
