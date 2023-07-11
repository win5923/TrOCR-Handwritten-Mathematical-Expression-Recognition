# Description
  Generate the math expression LATEX sequence according to the handwritten math expression image.
# How to run
  ```
    git clone https://github.com/win5923/TrOCR-Handwritten-Mathematical-Expression-Recognition.git
    pip install transformers
    pip install datasets jiwer
    pip install sentencepiece
  ```
# Train
 for Ubuntu you can use screen and run train2.py<br>
 for Jupyter you can run train.ipynb
   ```
    python train2.py
    python train.ipynb
  ```
# Inference
  use predict.py or test.py to inference on new images.
# Evaluate
  >On CHROME 2016 dataset CER = 0.193<br>
  >On CHROME 2016 dataset Accuracy = 0.306<br>

# Improve
  On CROHME 2016 test dataset the Accuracy is worst below image's model.
  ![image](https://user-images.githubusercontent.com/56353753/172812273-075e46aa-cb7d-4c2c-9436-3661c202dc39.png)
  
  


Thanks @NielsRogge's Notebook so much.It's very helpful.<br>
https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
