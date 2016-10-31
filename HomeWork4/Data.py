

class Data:
    """
    def __init__(self, architecture, reg_coeff, decay, momentum, accuracy, timeTaken):
        self.architecture=architecture
        self.reg_coeff=reg_coeff
        self.decay=decay
        self.momentum=momentum
        self.accuracy=accuracy
        self.timeTaken=timeTaken
    """
    def __init__(self, architecture, accuracy, timeTaken):
        self.Architecture=architecture
        self.Accuracy=str(round(accuracy*100, 2)) + "%"
        self.TimeTaken=timeTaken

