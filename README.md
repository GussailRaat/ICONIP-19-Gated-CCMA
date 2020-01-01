## Multi-task Learning for Multi-modal Emotion Recognition and Sentiment Analysis
Code for the paper [Multi-task Gated Contextual Cross-Modal Attention Framework for Sentiment and Emotion Analysis](https://link.springer.com/chapter/10.1007/978-3-030-36808-1_72) (ICONIP 2019)

For the evaluation of our proposed multi-task gated CCMA framerwork, we use benchmark multi-modal dataset i.e, MOSEI which has both sentiment and emotion classes.

### Dataset

* You can download datasets from [here](https://drive.google.com/open?id=1s10Bvmb7mInYof_Aui9y8q29dKmxYiB1).

* Download the dataset from given link and set the path in the code accordingly make two folder (i) results and (ii) weights.

-------------------------------------------------------
### For MOSEI Dataset:
For trimodal-->>  python trimodal_gated_multitask.py  

-------------------------------------------------------

### Emotion Results Extractor

Follow these steps to extract the threshold based resluls for emotion:

* Open the text file i.e., multiTask_emotion_results_extractor.txt
* Copy and paste on the terminal

#### Example: for trimodal
##### For preference F1 score:

If the result file name is emotion_trimodal_True:80_True:10.txt then run the following command 

* cat emotion_trimodal_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 6,6  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

So based on threshold, desired output will be stored in Emotion-Multi-task.txt (preference is F1-score)

##### For preference W-Acc:

If the result file name is emotion_trimodal_True:80_True:10.txt then run the following command 

* cat emotion_trimodal_True:80_True:10.txt | grep "mmmu" | grep "average" | grep -P "Threshold:" | sort -k 7,7  | tail -1 | cut -d$'\t' -f'5,6' >> Emotion-Multi-task.txt

So based on threshold, desired output will be stored in Emotion-Multi-task.txt (preference is W-Acc)

-------------------------------------------------------

### --versions--

python: 2.7

keras: 2.2.2

tensorflow: 1.9.0
