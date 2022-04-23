# Description
  Generate the math expression LATEX sequence according to the handwritten math expression image.
# How to run
  ```
    git clone https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git
    pip install -q transformers
    pip install -q datasets jiwer
    pip install sentencepiece
  ```
# Train
 for Ubuntu you can use screen and run train2.py<br>
 for Jupyter you can run train.ipynb
 
# Evaluate
  >On CHROME 2014 dataset CER = 0.365<br>
  >On CHROME 2016 dataset CER = 0.353<br>
  >On CHROME 2014 dataset Accuracy = 0.158<br>
  >On CHROME 2014 dataset Accuracy = 0.143<br>
# Improve
  On CROHME 2014 test dataset the Accuracy is worst below image's model.
  ![image](https://user-images.githubusercontent.com/56353753/160466308-0fbc4d84-f3e0-4f6a-957e-d42fd32d59fd.png)
  
  You can try to use the same processor model as the model,the effect of different models is very poor.
  


Thanks @NielsRogge's Notebook so much.It's very helpful.<br>
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
