Group Members:
  - Indrajeet Patwardhan
  - Vince Bjazevic
  - Francisco Ocegueda
  - Viraat Udar
  - Joel Anil John


Programs Included:

  - model_tools.py: Helper code to initialize the MLP and CNN models as well as perform training and testing
    (Adapted from L06_ANN_digit.py).

  - model_dispatcher.py: Dispatches random MLP and CNN models for modeling based on user submitted command. 
    The program is run with:

      python ./model_dispatcher.py {Verbosity (0-2)} {Number of random samples} {File Name}
    
    There are two predefined reference models inside of the dispatcher. To do a quick run, enter:

      python ./model_dispatcher.py 0 0 my_file
    
    After each succesful run, the modeling results are written to the specified filename.
    
  - resnet_trees.py: A ResNet-50 model trainer and classifier for identifying tree classes. This program reads
    from the Images folder and treeNames.txt in order to work properly. There are no special commands to run
    this code.