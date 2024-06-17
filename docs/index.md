# Welcome to QAttn's documentation!

Welcome to the QAttn documentation! QAttn (pronounced like katana) is a [python](https://docs.python.org/3) only framework with GPU kernels implemented in [Triton](https://triton-lang.org/) for quantized vision transformers. This framework implements integer and mixed-precision kernels for operations within vision transformers (currently matrix multiplication and attention) for static and dynamic quantization.

## Installation

To install the package, run

```bash
pip install qattn
```

or install from source to get the latest bleeding-edge source version.

```bash
pip install git+https://github.com/ibm/qattn.git
```

This package depends on Triton, requiring NVIDIA GPU (preferably Ampere or newer), and is tested only on Linux.


To install and modify source code, you can clone the repository locally and install it in editable mode.

```bash
git clone https://github.com/ibm/qattn.git
cd qattn
pip install -e .
```

## Usage

In the [Examples](examples) section, we present static and dynamic quantization usage samples using QAttn. QAttn is designed to be compatible with PyTorch FX-Quantization to replace dynamically models' graph floating-point modules with quantized ones. This comes with the downside of being unable to capture the control statements in the graph.

## Future direction

In the future, we will support the rest of the basic Vision Transformers operations (GELU, LayerNorm, Add, etc.) for fully quantized models. Next, we will move to the PyTorch 2.0 torchdynamo graph capture to enable integration with `torch.compile`.


## Contents
```{toctree}
:maxdepth: 2
:name: examples
examples
```

```{toctree}
:maxdepth: 2
:caption: QAttn Python API
qattn
```

## Acknowledgments


> The work is conducted within the project APROPOS. This project has received funding from the European Union’s Horizon 2020 (H2020) Marie Sklodowska-Curie Innovative Training Networks H2020-MSCA-ITN-2020 call, under the Grant Agreement no 956090. Project link: https://apropos-project.eu/
