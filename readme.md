# Medical Classification Project

This Python project classifies patient data into four distinct classes using two different methodologies: `Semantic Routing` and `Modern_Bert`. Additionally, it provides tools for visualizing sentence embedding vectors using `t-SNE` and `U-MAP`. For interactive plots, users can set the argument `-i` to enable this feature.

## Prerequisites

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Files in the Project

* **`main.py`**: The main script to classify data or visualize sentence vectors
* **`Examples.xlsx`**: Contains a few example patient data for the classification models
* **`args.py`**: Handles command-line arguments
* **`modules/SemanticRoute.py`**: Implements the Semantic Routing method
* **`modules/ModernBert.py`**: Implements the ModernBert method
* **`utils/t_SNE.py`**: Script for generating t-SNE visualizations
* **`utils/UMAP.py`**: Script for generating UMAP visualizations
* **`utils/bert_finetuning.py`**: Script for finetuning ModernBert model

## Usage Instructions

### 1. Classify Data

You can classify patient data into one of four classes using `Semantic Routing` or `Modern_Bert`. Run the `main.py` file with the following arguments:

#### Semantic Routing

```bash
python main.py --semantic
```

* By default, the dataset is set to `Classification_v1.json` and the embedding model is set to `sentence-transformers/all-MiniLM-L6-v2`

* To use a custom dataset or embedding model, add the arguments `--dataset <path_to_dataset>` or `--embedding <embedding_model>`

#### ModernBert

```bash
python main.py --bert
```

* Before using ModernBert, download the model weights from [this link](https://drive.google.com/file/d/1EFjGK_Y7ByFaq-zvWfeRUnryQJirSQ-W/view?usp=drive_link) and place them in the `Modernbert_model` folder.
* Both methods launch an interactive Gradio application where you can input text and receive classification results.

### 2. Visualize Sentence Vectors

#### t-SNE Visualization

```bash
python main.py --tsne -i
```

* To visualize a different dataset, add the argument `--dataset <path_to_dataset>`
* **`-i` or `--interactive`**: Optional flag to enable interactive plotting.

#### UMAP Visualization

```bash
python main.py --umap -i
```

* To visualize a different dataset, add the argument `--dataset <path_to_dataset>`
* **`-i` or `--interactive`**: Optional flag to enable interactive plotting.

### 3. Example Usage

#### Classify Data Using Semantic Routing

```bash
python main.py --semantic
```

#### Generate t-SNE Plot (Interactive)

```bash
python main.py --tsne --dataset Classification_v1.json -i
```

## Project Structure

```plaintext
.
├── main.py                  # Main entry point
├── args.py                  # Argument parser
├── Examples.xlsx            # Example dataset
├── modules/                 # Contains Semantic Routing and ModernBert implementations
│   ├── SemanticRoute.py
│   └── ModernBert.py
├── utils/                   # Utility scripts for visualizations and finetuning
│   ├── t_SNE.py
│   ├── UMAP.py
│   └── bert_finetuning.py
├── Datasets/                # Contains classification datasets
│   ├── Classification_v1.json
│   └── Classification_bert.json
├── Modernbert_model/       # Contains ModernBert model and weights (download from the link before use)
└── Visual_plots/            # Contains t-SNE and UMAP plots
```

## Notes

* Make sure to put your custom datasets inside the `Datasets` folder to work on it.
* Currently it supports only `huggingface embedding models`. It can further be extended to OpenAI embedding models or any other third-party provider.
* For ModernBert, download the model weights from [this link](#) and put them in the `Modeernbert_model` folder before running the classification.
