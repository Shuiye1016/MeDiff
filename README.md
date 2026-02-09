# MeDiff
MeDiff: A Cascade Information Diffusion Prediction Framework via Meso-Scale Evolution

## Description
Our MeDiff is implemented mainly based on the following libraries (see the README file in source code folder for more details):

- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- DGL: https://www.dgl.ai/
- PytorchNLP: https://github.com/PetrochukM/PyTorch-NLP
- Networkx: https://networkx.org/

### File Tree

Our main project file structure and description:

```python
MeDiff
├─ README.md
├─ attention.py
├─ clustering.py
├─ config.py
├─ data_loader.py
├─ embeddings.py
├─ graph_utils.py
├─ model.py
├─ utils.py
└─ train.py
```

### Installation

- Create Conda Environment:

  This project requires Python 3.8 or higher. The deep learning backbone relies on PyTorch 1.12 or higher.

- Activate Conda Environment:

  After the installation is complete, activate the newly created Conda environment using the following command:

  ```bash
  conda activate <environment_name>
  ```
  
- Verify Environment Installation:

  You can verify whether the environment is correctly installed:

  ```bash
  conda list
  ```

## Usage

You can start training.

```bash
python train.py
```

### Experiment Settings

Please refer to Sec. 5.1 Experimental Setup of the paper and config.py for details on data and parameter settings.
