# HeartGPT
Interpretable Pre-Trained Transformers for Heart Time-Series Data: 
[Link to the paper](https://www.arxiv.org/abs/2407.20775)

In this repo, there are trained model files, both for the pre-trained models ECGPT and PPGPT, and the models fine-tuned for arrythmias and beat detection. There are also GUIs designed for generation, interpretability, and for implementation of the fine-tuned models.

![GPT_comparison](figures/Comparison_slide_cropped.png)

In this work,  we apply the generative pre-trained transformer (GPT) framework to periodic heart time-series data to create two pre-trained general purpose cardiac models, namely PPG-PT and ECG-PT. The models are capable of being fine-tuned for many different cardiac related tasks such as screening for arrythmias. A big enphasis of this work is on showing that the pre-trained models are fully interpretable, and that this interpretability carries over to fine-tuning tasks. The pre-trained transformers are interpretable in the following ways -

**Aggregate attention maps** show that the model focuses on similar points in previous cardiac cycles in order to make predictions and gradually broadens its attention in deeper layers:
![Aggregate Attention](figures/aggregate_attention_editw.png)

Tokens with the same value (the time-series equivalent of a **homonym**) that occur at different distinct points in the ECG and PPG cycle form separate clusters in high dimensional space, based on their position in the cardiac cycle, as they are updated with the context of other tokens via the transformer blocks.
![Homonyms](figures/homonyms_vector_similarityw.png)

**Individual attention heads respond to specific physiologically relevent features**, such as the dicrotic notch in PPG and the P-wave in ECG.
![individual_heads](figures/SA_individual_editw.png)


This work was inspired by a [tutorial](https://github.com/karpathy/ng-video-lecture) created by Andrej Karpathy.

# Base code and model files:
**Heart_PT_generate.py** is a python script which loads in example contexts of either ECG or PPG, and uses the appropriate model to generate future time steps.

The pre-trained pytorch model files are in zip folders (**ECGPT_560k_iters** and **PPGPT_500k_iters**).

# Fine-tuning

An example adapted fine-tuning model definition is provided in **"Heart_PT_finetune.py"**, along with how to freeze different layers. 

![Finetune_fig](figures/fine_tuning_diagramw.png)
