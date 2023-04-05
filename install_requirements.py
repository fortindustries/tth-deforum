import platform
import subprocess


def pip_install_packages(packages,extra_index_url=None):
    for package in packages:
        try:
            print(f"..installing {package}")
            if extra_index_url is not None:
                running = subprocess.call(["pip", "install", "-q", package,  "--extra-index-url", extra_index_url], shell=False)
            else:
                running = subprocess.call(["pip", "install", "-q", package],shell=False)
        except Exception as e:
            print(f"failed to install {package}: {e}")
    return


def install_requirements():
    # Detect System
    os_system = platform.system()
    print(f"system detected: {os_system}")


    # Install pytorch
    torch = [
        "torch==2.0.0",
        "torchvision==0.15.1",
        "torchaudio==2.0.1"
    ]

    extra_index_url = "https://download.pytorch.org/whl/cu117" if os_system == 'Windows' else None
    pip_install_packages(torch,extra_index_url=extra_index_url)


    # List of common packages to install
    common = [
        "clean-fid",
        "colab-convert",
        "einops",
        "ftfy",
        "ipython",
        "ipywidgets",
        "jsonmerge",
        "jupyterlab",
        "jupyter_http_over_ws",
        "kornia",
        "matplotlib",
        "notebook",
        "numexpr",
        "omegaconf",
        "opencv-python",
        "pandas",
        "pytorch_lightning==1.7.7",
        "resize-right",
        "scikit-image==0.20.0",
        "scikit-learn",
        "timm",
        "torchdiffeq",
        "transformers==4.19.2",
        "safetensors",
        "albumentations",
        "more_itertools",
        "devtools",
        "validators",
        "numpngw",
        "open-clip-torch==2.13.0",
        "torchsde",
        "ninja",
    ]

    pip_install_packages(common)


    # Xformers install
    linux_xformers = [
        "triton==2.0.0",
        "xformers==0.0.16rc424",
    ]

    pip_install_packages(linux_xformers)


if __name__ == "__main__":
    install_requirements()