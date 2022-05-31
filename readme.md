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

To materialize previous step need to followed. 

