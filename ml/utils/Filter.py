#import pyeda as pyeda
import formulate

class Filter():
    """
    Filter data based on user passed arguments.
    The input from the user is a logical expression as a 'string'
    Translate 'string' to boolean logic via:
        i)   eval()
        ii)  pyeda()
    """

    def __init__(self, FilterString):
        super(Filter, self).__init__()
        # Intialise data members
        self.FilterString = FilterString
        self.FilterList = self.FilterLogic = []
        self.variables = set()

        # Construct data members based on input to filter class instance
        self.TokeniseFilterRequest() # Create a list of logic
        #self.ConvertLogic() # Convert to a list of boolean expressions

    def ConvertLogic(self):

        # Loop through list of expressions
        for expr in self.FilterList:
            # Convert using pydea
            #self.FilterLogic.append( pyeda.expr.expr(expr).to_dnf() )
            self.FilterLogic.append( eval(expr) )


    def TokeniseFilterRequest(self):

        # Split the list into a list of elements defined by the delimiter
        # ','
        self.FilterList = self.FilterString.split(',')
        print("<Filter.py>::   Filtering will be applied using:")
        print("<Filter.py>::      {}".format(self.FilterList))

        # storing variables of the expression
        for filter in self.FilterList:
            # taking set union here
            self.variables |= formulate.from_numexpr(filter).variables
