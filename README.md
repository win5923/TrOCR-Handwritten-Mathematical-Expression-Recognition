# How to run
  ```
    git clone https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git
    pip install -q transformers
    pip install -q datasets jiwer
  ```
# Train
 for shell you can use screen and run train2.py<br>
 for Jupyter you can run train.ipynb
 
# Evaluate
  >On CHROME 2014 dataset CER = 0.365<br>
  >On CHROME 2016 dataset CER = 0.353<br>
  >On CHROME 2014 dataset Accuracy = 0.158<br>
  >On CHROME 2014 dataset Accuracy = 0.143<br>

Thanks @NielsRogge's Notebook so much.It's very helpful.
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
