# Connection between simplicity bias and small dataset training
Mini-project for CS-439 OptML

## Abstract

Simplicity bias is an already widely studied phenomenon in the field of deep neural network, which is often thought to be directly related to the excellent generalization performance of neural networks. However, this feature also makes the model perform poorly on Out Of Distribution(OOD) data. We note that the field of small dataset training is also affected by this problem. So based on previous work, we try to reduce the level of the simplicity bias by using different optimization methods and applying the same optimization methods in small dataset training to explore the connection between them. We also explore the effect of the landscape of the achieved local minima on the generalization ability of the model through the spectral of the Hessian matrix at the local minima.

## Project structure

`data/:` dataset for MNIST, twitter sentiment, embedding of tweets

`experiment_results/:` results for all the experiments. (The part of results for simplicity bias is in the notebook "Synthetic_data_exp.ipynb")

`figures/:` all the figures we used in the report. (The part of images for simplicity bias is in the notebook "Synthetic_data_exp.ipynb")

`scripts/:` some python scripts that being called by the experiments related to simplicity bias.

`Synthetic_data_exp.ipynb:` experiments on simplicity bias.

`sb_in_nlp.ipynb:` experiments of simplicity bias in natural language processing.

`test_bigData.ipynb:` experiments on MNIST dataset with Adam and SGD (Appendix B).

`test_smalldata.ipynb:` experiments on the small dataset and landscape detection, all errors are keyboard interrupt with results saved in the `experiment_results/`.



