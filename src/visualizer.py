import matplotlib.pyplot as plt

def drawAccuracyComparisons(title : str, categories: list, accuracy : list, path : str):
    """
        The  method draw an accuracy comparison Bar diagram;
        Input : List of accuracy of each different strategies
        Output : Draw a Bar diagram Or Errors

        Input arguments (
            title:string, 
            categories: List of string (X-axis), 
            accuracy : List of float (Y-axis),
            path : path to save/store the diagram 
            )
    """
    if len(categories) != len(accuracy):
        raise ValueError("The size of categories and accuracy are not equal")
    if len(accuracy) == 0 :
        raise ValueError ("Empty input")
    
    # Diagram type is bar
    plt.bar(categories, accuracy)
    plt.xlabel('categories')
    plt.ylabel('accuracy')
    plt.title(title)


    plt.savefig(path + 'bar_graph.png')
    # You can use plt.show() to show the new diagram and do not need to save it
    # plt.show() 
    plt.close()
    print("Diagram saved to " + path + 'bar_graph.png')