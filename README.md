# 9517-group-project
## How to download the dataset
__Geting the kaggle KPI from Kaggle website__
[Kaggle Dataset Link](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)


mkdir -p ~/.kaggle

mv /path/to/kaggle.json ~/.kaggle/

chmod 600 ~/.kaggle/kaggle.json


__Using this command to download to dataset to your own laptop__

kaggle datasets download -d ankit1743/skyview-an-aerial-landscape-dataset

## Deep Learning


### Enviroment confugureation
contourpy==1.3.2

cycler==0.12.1

filelock==3.18.0

fonttools==4.57.0

fsspec==2025.3.2

Jinja2==3.1.6

joblib==1.4.2

kiwisolver==1.4.8

MarkupSafe==3.0.2

matplotlib==3.10.1

mpmath==1.3.0

munch==4.0.0

networkx==3.4.2

numpy==2.2.4

packaging==24.2

pandas==2.2.3

pillow==11.2.1

pretrainedmodels==0.7.4

pyparsing==3.2.3

python-dateutil==2.9.0.post0

pytz==2025.2

scikit-learn==1.6.1

scipy==1.15.2

seaborn==0.13.2

setuptools==78.1.0

six==1.17.0

sympy==1.13.1

threadpoolctl==3.6.0

torch==2.6.0

torchvision==0.21.0

tqdm==4.67.1

typing_extensions==4.13.2

tzdata==2025.2

### Overview

This project investigates advanced deep learning techniques for image classification.  
It compares the performance of the following convolutional neural network (CNN) architectures:

- **ResNet**: A residual learning framework that enables training of very deep networks.
- **SENet (Squeeze-and-Excitation Networks)**: Enhances channel interdependencies to improve representational power.
- **EfficientNet**: Scales depth, width, and resolution efficiently using compound scaling.

The models are evaluated on a custom dataset with metrics such as accuracy, confusion matrix, and Grad-CAM visualizations for model interpretability.

### Structure

-**The model name can be changed to the model you choose**
1.model.train
2.model.test
3.model.visualisation
4.misclassification result

### How to run




## Machine Learning part

The machine learning part focused on implementing and comparing various feature extraction techniques and classification models. The main code is included in the `machine_learning.ipynb` notebook, with additional result outputs available.

### Overview

The project explores the following techniques:
- **Feature Extraction**: Local Binary Patterns (LBP) and Scale-Invariant Feature Transform (SIFT)
- **Classification Models**: k-Nearest Neighbors (KNN) and Support Vector Machines (SVM)

### Structure

The repository is organized as follows:
- **`machine_learning.ipynb`**: The main Jupyter Notebook containing the implementation of the feature extraction methods and machine learning models.
- **Result Files**:
  - `LBP_KNN`: Results of using LBP for feature extraction combined with KNN for classification.
  - `LBP_SVM`: Results of using LBP for feature extraction combined with SVM for classification.
  - `SIFT_KNN`: Results of using SIFT for feature extraction combined with KNN for classification.
  - `SIFT_SVM`: Results of using SIFT for feature extraction combined with SVM for classification.

### Prerequisites

To run the code in this repository, you need the following:
- Python 3.8 or higher
- Jupyter Notebook
- Required Python libraries (listed in `requirements.txt` or detailed below):
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `opencv-python`

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/cyzg/9517-group-project.git
   cd 9517-group-project
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook machine_learning.ipynb
   ```

3. Follow the instructions in the notebook to execute each section and reproduce the results.

### Results

This part investigates the effectiveness of various combinations of feature extraction methods and classification algorithms. Comprehensive evaluation metrics and result visualizations are provided in the accompanying notebook.

## Authors

This project is developed by the contributors of the repository.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
