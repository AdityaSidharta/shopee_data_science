# Shopee Data Science Competition - ShopeeSense

This is the codebase for ShopeeSense submission for Shopee Data Science Competition. 
The competition requires the participants to extract attributes from the image and title description of
 a listed product. 
The product was from three categories - Mobile, Beauty, and Fashion.

For each of the categories, we built four algorithms which we ensemble to output the final prediction. 
The four algorithms are listed as follows:
1. Term Frequency - Inverse Document Frequency, Single Value Decomposition, and Gradient Boosted Decision Tree for product title.
2. Recurrent Neural Network with ASGD Weighted-Dropped Long Short Term Memory architecture for product title.
3. Convolution Neural Network with pretrained ResNet34 for product image.
4. Heuristic Algorithm based on the product title.

The architecture of the model is listed as follows:




## Getting Started

These instructions will get you a copy of the project up and running 
on your local machine for development and deployment purposes.

### Prerequisites

What things you need to install the software and how to install them.

### Managing Project Dependencies using Pyenv + Pipenv

We use [pipenv](https://docs.pipenv.org) for managing project dependencies
 and Python environments (i.e. virtual environments). All direct packages 
 dependencies (e.g. NumPy may be used in a User Defined Function), 
 as well as all the packages used during development (e.g. PySpark, 
 flake8 for code linting, IPython for interactive console sessions, etc.),
  are described in the `Pipfile`. Their **precise** downstream dependencies
   are described in `Pipfile.lock`.

### [OPTIONAL] Installing Pyenv
```bash
sudo apt install pyenv
```

Other installation instructions [here](https://github.com/pyenv/pyenv#installation).

#### [OPTIONAL] Automatically initialize pyenv when terminal loads
```
echo eval "$(pyenv init -)" >> ~/.bash_profile
source ~/.bash_profile
```

#### [OPTIONAL] Installing python

To see all available python versions

```
pyenv install --list
```

To install Python 3.7.2

```
pyenv install 3.7.2
```

To install Python 3.7.2 in this directory

```
cd sparkflow
pyenv install 3.7.2
pyenv local 3.7.2
```

### [OPTIONAL] Installing Pipenv

To get started with Pipenv, first of all download it - assuming that there is a global version of Python available 
on your system and on the PATH, then this can be achieved by running the following command,

```bash
pip install pipenv
```

Setting up your pipenv for the first time

```bash
pipenv --python 3.7.2
```

For more information, including advanced configuration options, see the [official pipenv documentation](https://docs.pipenv.org).

### Kaggle API

To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`). You can define a shell environment variable `KAGGLE_CONFIG_DIR` to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json` (on Windows it will be `%KAGGLE_CONFIG_DIR%\kaggle.json`).

For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command: 

`chmod 600 ~/.kaggle/kaggle.json`

You can also choose to export your Kaggle username and token to the environment:

```bash
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```
In addition, you can export any other configuration value that normally would be in
the `$HOME/.kaggle/kaggle.json` in the format 'KAGGLE_<VARIABLE>' (note uppercase).  
For example, if the file had the variable "proxy" you would export `KAGGLE_PROXY`
and it would be discovered by the client.


## Installation

Make sure that you're in the project's root directory (the same one in which the `Pipfile` resides), and then run,

```
make setup
```
This single command will ensure that pyenv and pipenv is installed within your computer, and 
it installs all of the package dependencies for the project.

```
make prepare-data
```
This will do the following:
1. Creates `/data` folder
2. Download all the data from Shopee Kaggle Dataset into `/data`, and unzip all of them
3. Renaming all files that ends with `_info_val_competition.csv` to `_info_test_competition.csv`
4. Splitting the training files as into Development dataset and Validation dataset, `_info_dev_competition.csv` and 
    `_info_val_competition.csv`
    
Development Dataset and Validation Dataset is a subset of the Training Dataset, where we will train the model on
Development Dataset, and Use the Validation Dataset to get the estimated Performance of the Model

```
bash bins/dl_large_files.sh
```

This script will try to download the Beauty, Fashion, and Mobile image files.

## Running 

### Project Structure

```
├── bins
│   ├── data.sh
│   ├── dl_large_files.sh
│   ├── fastai.sh
│   ├── img.sh
│   ├── lgb.sh
│   └── setup_linux.sh
├── config.sh
├── data
│   └── fono_api
├── __init__.py
├── LICENSE
├── Makefile
├── model
│   ├── common
│   │   ├── __init__.py
│   │   ├── split_data.py
│   │   └── topic.py
│   ├── heuristic
│   │   ├── fashion_library.json
│   │   ├── __init__.py
│   │   └── mobile
│   │       ├── enricher.py
│   │       ├── extractor.py
│   │       └── __init__.py
│   ├── image
│   │   ├── fastai
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   └── ml_model.py
│   │   ├── __init__.py
│   │   └── pytorch
│   │       ├── dataset.py
│   │       ├── __init__.py
│   │       ├── test.py
│   │       ├── train.py
│   │       └── validation.py
│   ├── __init__.py
│   ├── leak
│   │   ├── __init__.py
│   │   └── main.py
│   └── text
│       ├── common
│       │   ├── __init__.py
│       │   └── prediction.py
│       ├── fastai
│       │   ├── class_model.py
│       │   ├── __init__.py
│       │   ├── lm_model.py
│       │   └── main.py
│       ├── __init__.py
│       ├── lgb
│       │   ├── config.py
│       │   ├── eta_zoo.py
│       │   ├── __init__.py
│       │   ├── lgb_model.py
│       │   ├── main.py
│       │   └── tuning.py
│       └── utils
│           ├── common.py
│           └── __init__.py
├── notebooks
│   ├── adi
│   │   ├── combine_answer.ipynb
│   │   ├── concat_submission.ipynb
│   │   ├── fastai_clasifier.ipynb
│   │   ├── fastai_lm.ipynb
│   │   ├── ida_beauty_prediction.ipynb
│   │   ├── ida_fashion_prediction.ipynb
│   │   ├── image_fastai.ipynb
│   │   ├── image_fastai_pycharm.ipynb
│   │   ├── image_folder_management.ipynb
│   │   ├── __init__.py
│   │   ├── kyle_enricher.ipynb
│   │   ├── Kyle LM.ipynb
│   │   ├── kyle_prediction.ipynb
│   │   ├── kyle_validation.ipynb
│   │   ├── leak_answer.ipynb
│   │   ├── lgb_model.ipynb
│   │   └── submission.ipynb
│   ├── ida
│   │   └── fashion_heuristics_generate_review_heuristics_library.ipynb
│   ├── kyle
│   │   ├── first.py
│   │   └── __init__.py
│   └── placeholder
├── output
│   ├── logs
│   │   └── placeholder
│   ├── model_checkpoint
│   │   └── placeholder
│   ├── result
│   │   └── placeholder
│   └── result_metadata
│       └── placeholder
├── Pipfile
├── Pipfile.lock
├── README.md
├── try.png
└── utils
    ├── api_keys.py
    ├── common.py
    ├── envs.py
    ├── fonAPI.py
    ├── __init__.py
    ├── logger.py
    └── pytorch
        ├── callbacks
        │   ├── callback.py
        │   ├── __init__.py
        │   ├── loss.py
        │   └── optim.py
        ├── __init__.py
        └── utils
            ├── checkpoint.py
            ├── common.py
            ├── __init__.py
            └── lr_finder.py
```

All of the main algorithms is located under `model/` directory. Each of the main algorithms 
(`model/image/fastai`, `model/image/pytorch`, `model/text/lgb`, `model/text/fastai`) has a `main.py` 
which can be ran directly to obtain the final predictions. `config.sh` is needed to be run before running each of the main
python script to initialize the environment variables

Ensembling the predictions from each of the model can be done using the following jupyter notebook:
 [notebooks/adi/concat_submission.ipynb](notebooks/adi/concat_submission.ipynb)

## Built With

* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 
* [tqdm](https://github.com/tqdm/tqdm) - A fast, extensible progress bar for Python and CLI
* [pandas](https://pandas.pydata.org/pandas-docs/stable/) - powerful Python data analysis toolkit
* [seaborn](https://seaborn.pydata.org) - statistical data visualization
* [Pytorch](https://pytorch.org/) - Deep Learning Framework for Python
* [torchvision](https://pytorch.org/docs/stable/torchvision/index.html) - Image Toolkit for Pytorch
* [ipykernel](https://github.com/ipython/ipykernel) - IPython Kernel for Jupyter
* [kaggle](https://github.com/Kaggle/kaggle-api) - Official Kaggle API
* [scikit-learn](https://scikit-learn.org/stable/) - machine Learning in Python
* [scikit-image](https://scikit-image.org/) - image processing in python
* [category-encoders](http://contrib.scikit-learn.org/categorical-encoding/) - Categorical Encoding for Python
* [xgboost](https://xgboost.readthedocs.io/en/latest/) - optimized distributed gradient boosting library 
* [lightgbm](https://lightgbm.readthedocs.io/en/latest/) - gradient boosting framework that uses tree based learning algorithms
* [argparse](https://docs.python.org/3/library/argparse.html) - Parser for command-line options, arguments and sub-commands
* [fastai](https://github.com/fastai/fastai) - making neural nets uncool again

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Aditya Kelvianto Sidharta** [AdityaSidharta](https://github.com/AdityaSidharta)
* **Kyle Tan** [kyleissuper](https://github.com/kyleissuper)
* **Idawati Bustan** [idawatibustan](https://github.com/idawatibustan)
* **Nikolas Lee** [nykznykz](https://github.com/nykznykz)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

A huge thank you to [Shopee](https://shopee.sg/) Data Science team for organizing this hackathon.