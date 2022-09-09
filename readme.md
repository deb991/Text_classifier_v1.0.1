<h1>Text Classifier</h1>

Text Classifier for analyze text from given input, whether it is a scanned document(Mostly Imge or some other format), or a native PDf file. 

<h3>Special activity & Commands are as follows. </h3>

** To get process/ execution flow: \

dagit -f .\models\lda_model\__init__.py

**ex: dagit -f <script location\>**

** To get each function to be considered as an op method. So we can visualize execution flow. 

```@op
    def func_name(args, .....):
        desired operation 
        ....\
        ....
            ......``
        return something
```

** To make visualize execution flow, 

```@job
   def func_name():
       #Direct call function name
       <op_method_name>(<if any other op_method required>)
       
       Voila, if everything set properly, execution flow will be visible. 
       
```

To materialize previous step need to follow. 

<h3>OCR Integration</h3>
* As of now Hard coded Pytessaract path into the script. 


``ocr.pytesseract.tesseract_cmd = \
    'C:\\Tesseract-OCR\\tesseract.exe'``

* Added features -- OCR Analyzer :
    * Added ``cv2.imshow("Thresold_Image", threshed)``
      First param is "Window Name", 
      Second one is the "processed data". 
    * Usually CV2 mostly accepts Numpy array Data. So better we pass it, without use "try" block. 
    * OCR Analyzer only for test purpose 

<br>Next Changes target: 
  1. Pass data from OCR to model & analyze data. 
  2. Add NB along with LDA model. </br>

* To test on LDA model, 


  ``JText-classifier_main/models/lda_model/__init__.py``

