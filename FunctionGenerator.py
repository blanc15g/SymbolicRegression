'''
Current Goal of this project is to perform single variable symbolic regression
Method is to loop through all possible function trees of a reasonably small size
and return the simplest/best approximation
'''

__author__ = 'guy'






import numpy as np
import copy
from scipy import optimize as opt
import random
import time

nodeTypes = []

class Node():
    '''A parent class used to represent all the nodes in the function tree
    It's not actually needed, but just a reference to what functions should always be available'''

    COMPLEXITY = 0  #a measure of node complexity, aim to minimize this with good fit
                    #these are arbitrary, based on what makes the "most sense"

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''
        raise Exception("__init__ not defined for {0}".format(type(self)))
        self.parent = parent

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''
        raise Exception("__repr__ not defined for {0}".format(type(self)))
        return ""


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''
        raise Exception("evolve not defined for {0}".format(type(self)))
        return []

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''

        raise Exception("create_basic not defined for child of {0}".format(type(parent)))
        return []

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        raise Exception("preprocess not defined for {0}".format(type(self)))

    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''
        raise Exception("evaluate not defined for {0}".format(type(self)))
        return 0

    def functionComplexity(self):
        '''
        Returns a measure of the complexity for the function starting at this node
        '''
        if hasattr(self, 'children'):
            childComplexities = 0
            for child in self.children:
                childComplexities += child.functionComplexity()
            return self.COMPLEXITY * (len(self.children) - 1) + childComplexities

        if hasattr(self, 'child'):
            return self.COMPLEXITY + self.child.functionComplexity()

        return self.COMPLEXITY

    def __lt__(self, other):
        '''
        makes nodes sortable based on complexity
        '''
        return self.functionComplexity() < other.functionComplexity()

    def numCoef(self):
        return self.preprocess(0)









class Coef(Node):
    '''
    Class representing a coefficient node in a function tree
    '''

    COMPLEXITY = 1
    #low because coefficients are already penalized in adjusted R^2 calculation

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''
        self.parent = parent
        self.coefNum = -1   #don't yet know which coefficient this is

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        if (self.coefNum == -1):
            return "coef"   #don't yet know which coefficient

        return chr(ord('a') + self.coefNum)  #label first coefficient a, second b, etc.


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        return []   #coefficients don't evolve

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node

                In this case return empty list because we do not want to add coef nodes in this way
        '''

        return []

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        self.coefNum = numCoefsNumbered

        return numCoefsNumbered + 1


    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation

        :return:    The value evaluated
        '''

        if (self.coefNum == -1):
            raise Exception("Preprocessing needed before evaluating")

        #print(coefficients[self.coefNum])
        return np.array([coefficients[self.coefNum]] * len(x))

nodeTypes.append(Coef)



class Var(Node):

    COMPLEXITY = 3
    #Adding the variable is about middle of the road complexity

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''

        self.parent = parent

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        return "x"


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''
        possibleEvolutions = []

        for nodeT in nodeTypes:
            possibleEvolutions += nodeT.create_basic(self.parent)

        return possibleEvolutions

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node

                In this case return empty list because we do not want to add Var nodes in this way
        '''

        return []

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        return numCoefsNumbered

    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''

        return x

nodeTypes.append(Var)


class Add(Node):
    '''
    Class representing a node that adds its children together
    '''

    COMPLEXITY = 1  #addition is pretty simple

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''
        self.parent = parent
        self.children = []  #this will be set later

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''


        return '(' + ' + '.join(map(str,self.children)) + ')'


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        possibleEvolutions = []

        containsVar = False
        for child in self.children:
            if type(child) == Var:
                containsVar = True

        if not containsVar:
            #add a new Var node as one evolution
            newAdd = copy.deepcopy(self)
            newAdd.children.append(Var(self))
            possibleEvolutions.append(newAdd)

        #add all recursive evolutions
        for i, child in enumerate(self.children):
            for evolved in child.evolve():
                newAdd = copy.deepcopy(self)
                newAdd.children[i] = evolved
                possibleEvolutions.append(newAdd)
        return possibleEvolutions

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''

        if type(parent) == Add:     #we don't want to have two nested Adds for redundancy purposes
            return []

        ans = Add(parent)           #just one possible basic building block
        ans.children = [Coef(ans), Var(ans)]
        return [ans]

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        for child in self.children:
            numCoefsNumbered = child.preprocess(numCoefsNumbered)   #update as we go

        return numCoefsNumbered



    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''

        return sum(child.evaluate(x, coefficients) for child in self.children)


nodeTypes.append(Add)


class Mult(Node):
    '''
    Node representing multiplication
    '''

    COMPLEXITY  = 2


    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''

        self.parent = parent
        self.children = []

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        return '(' + ' * '.join(map(str,self.children)) + ')'


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        possibleEvolutions = []

        containsVar = False
        for child in self.children:
            if type(child) == Var:
                containsVar = True

        if not containsVar:
            #add a new Var as one evolution
            newAdd = copy.deepcopy(self)
            newAdd.children.append(Var(self))
            possibleEvolutions.append(newAdd)

        #add all recursive evolutions
        for i, child in enumerate(self.children):
            for evolved in child.evolve():
                newAdd = copy.deepcopy(self)
                newAdd.children[i] = evolved
                possibleEvolutions.append(newAdd)

        return possibleEvolutions

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''

        if type(parent) == Mult:    #don't want nested Mult
            return []

        ans = Mult(parent)
        ans.children = [Coef(ans), Var(ans)]    #only one basic building block
        return [ans]

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        for child in self.children:
            numCoefsNumbered = child.preprocess(numCoefsNumbered)   #update as we go

        return numCoefsNumbered

    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''

        product = 1   #initialize product to 1

        #print(coefficients)

        for child in self.children:
            #print(child.evaluate(x, coefficients))
            product *= child.evaluate(x, coefficients)
        return product

nodeTypes.append(Mult)


class Power(Node):
    '''
    Node representing the exponentation of its children
    '''

    COMPLEXITY = 5

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''

        self.parent = parent
        self.children = []
        self.evolutionCounter = 0

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        return ' ** '.join(map(str,self.children))


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        possibleEvolutions = []

        self.evolutionCounter += 1
        if self.evolutionCounter % 2 == 0:
            #add version with extra var on the end of the power chain
            #only do this every other evolution because power chains are uncommon
            varAdd = copy.deepcopy(self)
            varAdd.children.append(Var(self))
            possibleEvolutions.append(varAdd)

        if (type(self.children[-1]) != Coef):   #check whether we should add extra coef to end
            coefAdd = copy.deepcopy(self)
            coefAdd.children.append(Coef(self))
            possibleEvolutions.append(coefAdd)

        #add all recursive evolutions
        for i, child in enumerate(self.children):
            for evolved in child.evolve():
                newAdd = copy.deepcopy(self)
                newAdd.children[i] = evolved
                possibleEvolutions.append(newAdd)

        return possibleEvolutions

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''

        if type(parent) == Power or type(parent) == Log:    #don't make a power node here
            return []

        answers = [Power(parent),Power(parent), Power(parent)]  #three basic building blocks
        answers[0].children = [Coef(answers[0]), Var(answers[0])]
        answers[1].children = [Var(answers[1]), Var(answers[1])]
        answers[2].children = [Var(answers[2]), Coef(answers[2])]
        return answers

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        for child in self.children:
            numCoefsNumbered = child.preprocess(numCoefsNumbered)   #update as we go

        return numCoefsNumbered

    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''


        powerChain = 1   #initialize power

        for child in reversed(self.children):
            powerChain = child.evaluate(x, coefficients) ** powerChain

        return powerChain

nodeTypes.append(Power)


class Invert(Node):
    '''A node that evaluates to the recipricol of it's child node'''

    COMPLEXITY = 2

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''

        self.parent = parent

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        return '(1/('+ str(self.child)+'))'


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        possibleEvolutions = []

        for evolved in self.child.evolve():     #just pass on evolution to the child
            newAdd = copy.deepcopy(self)
            newAdd.child = evolved
            possibleEvolutions.append(newAdd)

        return possibleEvolutions


    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''

        if type(parent) == Invert:  #don't want nested invert nodes
            return []

        ans = Invert(parent)
        ans.child = Var(ans)
        return [ans]

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        return self.child.preprocess(numCoefsNumbered)


    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''

        return 1/self.child.evaluate(x, coefficients)

nodeTypes.append(Invert)



class Log(Node):
    '''
    Node that returns the natural log of its child
    '''

    COMPLEXITY = 5

    def __init__(self, parent):
        '''
        :param parent:  Parent node of this node
        '''

        self.parent = parent

    def __repr__(self):
        '''
        :return: A string representation of the node
        '''

        return "np.log(" + str(self.child) + ')'


    def evolve(self):
        '''
        :return: A list of all the nodes this could evolve into
        '''

        possibleEvolutions = []

        for evolved in self.child.evolve():
            if  (type(evolved) != Mult and
                type(evolved) != Power and
                type(evolved) != Invert):

                newAdd = copy.deepcopy(self)
                newAdd.child = evolved
                possibleEvolutions.append(newAdd)
        return possibleEvolutions

    def create_basic(parent):
        '''
        Used for evolutionary purposes

        :return: A list of basic building blocks of this node
                Can also return empty if we do not want to evolve building blocks of this node
        '''


        ans = Log(parent)
        ans.child = Var(ans)
        return [ans]

    def preprocess(self, numCoefsNumbered):
        '''
        Numbers coefficient nodes and returns number of coefficients in the tree

        :param numCoefsNumbered: The number of coefficients that have already been numbered

        :return: The number of coefficients numbered after function ends
        '''

        return self.child.preprocess(numCoefsNumbered)

    def evaluate(self, x, coefficients):
        '''
        Evaluates the function formed by considering the tree starting at self

        :param coefficients:    Coefficients to use for evaluating purposes
        :param x:   The independent variable of the evaluation
        :return:    The value evaluated
        '''

        return np.log(self.child.evaluate(x, coefficients))

nodeTypes.append(Log)





def nextGen(prevGen):
    answer = []
    for tree in prevGen:
        answer += tree.evolve()
    return answer

firstGen = [Var(None), Coef(None)]
g = nextGen(firstGen)
g = nextGen(g)



def fitY(functionTree, xData, yData):
    '''
    Fits a function to the data and returns what y values would be given by that function

    :param functionTree: A node that is at the root of the function to be evaluated
    :param xData: Independent variable data
    :param yData: Dependent variable data

    :return: The closest the function can get to yData
    '''

    numCoef = functionTree.preprocess(0)
    if numCoef == 0:
        return functionTree.evaluate(x, [])

    def yFromCoef(x, *coef):
        #print(coef)
        return functionTree.evaluate(x, coef)

    initialGuess = np.array([1] * numCoef)

    coefficients = opt.curve_fit(yFromCoef, xData, yData, initialGuess)[0]
    return yFromCoef(x, *coefficients)


def nodeToFunctionString(node):
    '''
    Converts a function tree to the code for it's Python string

    :param node: top node of the function tree
    :return: string that can be put into eval() to output the corresponding function object
    '''
    numCoef = node.preprocess(0)
    stringRep = 'lambda x'
    for i in range(numCoef):
        stringRep += ', ' + chr(ord('a') + i)
    stringRep += ': '

    #now right part of the lambda
    stringRep += str(node)
    return stringRep

def fitYEvalVersion(function, xData, yData, numCoef):
    '''
    Same as fitY but takes in a function found using nodeToFunctionString

    :param function: Python function object
    :param xData: Independent variable data
    :param yData: Dependent variable data

    :return: The closest the function can get to yData
    '''

    if (numCoef == 0):
        return function(xData)

    guess = [1] * numCoef
    coefficients = opt.curve_fit(function, xData, yData, guess)[0]
    return function(xData, *coefficients)



x = np.array([ 1.025,  1.075,  1.125,  1.175,  1.225,  1.275,  1.325,  1.375,
       1.425,  1.475,  1.525,  1.575,  1.625,  1.675,  1.725,  1.775,
       1.825,  1.875,  1.925,  1.975])

y = np.array([-38.94870755, -39.04621238, -39.12120312, -39.17805992,
      -39.22037884, -39.25116963, -39.27271959, -39.28706432,
      -39.29572699, -39.29996363, -39.30075747, -39.29882897,
      -39.29480065, -39.28913329, -39.28217057, -39.2742044 ,
      -39.26543558, -39.25631536, -39.24702911, -39.23775099])


def serializedBasedOnData(functionTree, x, y):
    '''
    Attempts to give each functionTree a unique signature that will signify whether two produce identical results
    To do this, we fit the function to a given set of data and use the fitted data as a signature for the function
    identical functions should produce identical results

    :param functionTree: Function to serialize
    :param x: xData to use
    :param y: yData to use

    :return: A tuple that is hopefully a unique signature for the function
    '''

    rounded = [round(point, 5) for point in fitY(functionTree, x, y)]
    return tuple(rounded)

def serializedBasedOnDataString(function, x, y, numCoef):
    '''
    Same as above but using string evalaution method

    :param function: Function to serialize
    :param x: xData to use
    :param y: yData to use

    :return: A tuple that is hopefully a unique signature for the function
    '''

    rounded = [round(point, 5) for point in fitYEvalVersion(function, x, y, numCoef)]
    return tuple(rounded)

def findUniqueFunctions(depth, xData, yData):
    '''
    Attempts to find a list of all unique functions that are up to depth evolutions away from just a single Var node

    :param depth: Depth to search for

    :param xData: List of np arrays containing various x distributions
    :param yData: List of np arrays containing the corresponding y distributions

    :return: List of nodes to unique functions
    '''

    models = []
    serializedUsed = set()  # we will attempt to serialize functions into this set
                            #based on their fitted data
                            #to prevent using the same function twice

    currentGen = [Var(None), Coef(None)]
    models += currentGen

    numOverlap = 0        #these two variables aren't needed, but good as status chcekcs
    numFailed = 0



    for i in range(depth):


        nextCurrentGen = []     #stores the next generation
        for model in nextGen(currentGen):
            serialized = ()
            for dataSet in range(len(xData)):   #assumes xData and yData have same length
                try:
                    serialized += serializedBasedOnData(model, xData[dataSet], yData[dataSet])
                except:
                    pass

            if serialized == ():    #failed to fit any of the data
                numFailed += 1

            else:
                if serialized in serializedUsed:
                    numOverlap += 1
                else:
                    nextCurrentGen.append(model)
                    models.append(model)
                    serializedUsed.add(serialized)

        currentGen = nextCurrentGen
        print(i, len(currentGen), numOverlap, numFailed)  #status check

    return models


def findUniqueFunctionsStringVersion(depth, xData, yData, output = False):

    if output:
        outputFile = open('models.txt', 'w')
    '''
    Same as above but uses string method of evaluation and returns strings. Generally a bit faster

    :param depth: Depth to search for

    :param xData: List of np arrays containing various x distributions
    :param yData: List of np arrays containing the corresponding y distributions

    :return: List of strings representing each unique function
    '''

    models = []
    serializedUsed = set()  # we will attempt to serialize functions into this set
                            #based on their fitted data
                            #to prevent using the same function twice

    currentGen = [Var(None), Coef(None)]
    for model in currentGen:
        models.append(nodeToFunctionString(model))
        if output:
            outputFile.write(nodeToFunctionString(model) + '\n')

    numOverlap = 0        #these two variables aren't needed, but good as status checks
    numFailed = 0


    for i in range(depth):
        nextCurrentGen = []     #stores the next generation
        for model in nextGen(currentGen):
            serialized = ()
            functionString = nodeToFunctionString(model)
            numCoef = model.preprocess(0)
            function = eval(functionString)
            for dataSet in range(len(xData)):   #assumes xData and yData have same length
                try:
                    serialized += serializedBasedOnDataString(function, xData[dataSet], yData[dataSet], numCoef)
                except:
                    pass

            if serialized == ():    #failed to fit any of the data
                numFailed += 1

            else:
                if serialized in serializedUsed:
                    numOverlap += 1
                else:
                    nextCurrentGen.append(model)
                    models.append(functionString)
                    serializedUsed.add(serialized)

                    if output:
                        outputFile.write(functionString + '\n')

        currentGen = nextCurrentGen
        print(i, len(currentGen), numOverlap, numFailed)  #status check

    if output:
        outputFile.close()
    return models


DELIMTER = '\t\t'#:-:\t\t'
def modelsWithComplexity(depth, xData, yData, output = False):
    #if output:
    #    outputFile = open('modelsTable.txt', 'w')
    '''
    Same as above but gives 3 pieces of information for each model:
        1) complexity
        2) num coefficients
        3) model itself

    :param depth: Depth to search for

    :param xData: List of np arrays containing various x distributions
    :param yData: List of np arrays containing the corresponding y distributions

    :return: List of strings representing each unique function
    '''

    models = []
    serializedUsed = set()  # we will attempt to serialize functions into this set
                            #based on their fitted data
                            #to prevent using the same function twice

    currentGen = [Var(None), Coef(None)]
    for model in currentGen:
        models.append((model.functionComplexity(), model.numCoef(), nodeToFunctionString(model)))
        #if output:
        #    outputFile.write(nodeToFunctionString(model) + '\n')

    numOverlap = 0        #these two variables aren't needed, but good as status checks
    numFailed = 0


    for i in range(depth):
        nextCurrentGen = []     #stores the next generation
        for model in nextGen(currentGen):
            serialized = ()
            functionString = nodeToFunctionString(model)
            numCoef = model.preprocess(0)
            function = eval(functionString)
            for dataSet in range(len(xData)):   #assumes xData and yData have same length
                try:
                    serialized += serializedBasedOnDataString(function, xData[dataSet], yData[dataSet], numCoef)
                except:
                    pass

            if serialized == ():    #failed to fit any of the data
                numFailed += 1

            else:
                if serialized in serializedUsed:
                    numOverlap += 1
                else:
                    nextCurrentGen.append(model)
                    models.append((model.functionComplexity(), model.numCoef(), functionString))
                    serializedUsed.add(serialized)

                    #if output:
                    #    outputFile.write(functionString + '\n')

        currentGen = nextCurrentGen
        print(i, len(currentGen), numOverlap, numFailed)  #status check

    models.sort()
    if output:
        with open('modelsTable.txt', 'w') as outputFile:
            for model in models:
                outputFile.write(DELIMTER.join(list(map(str,model))) + '\n')


    return models


#Sample data from another research project
x1 = np.array([ 1.025,  1.075,  1.125,  1.175,  1.225,  1.275,  1.325,  1.375,
       1.425,  1.475,  1.525,  1.575,  1.625,  1.675,  1.725,  1.775,
       1.825,  1.875,  1.925,  1.975])

y1 = np.array([-38.94870755, -39.04621238, -39.12120312, -39.17805992,
      -39.22037884, -39.25116963, -39.27271959, -39.28706432,
      -39.29572699, -39.29996363, -39.30075747, -39.29882897,
      -39.29480065, -39.28913329, -39.28217057, -39.2742044 ,
      -39.26543558, -39.25631536, -39.24702911, -39.23775099])



#Linear sample data with noise
x2 = np.array(list(range(1, 20)))
y2 = []
for xVal in x2:
    y2.append(xVal*2 + 3 + random.random())

y2 = np.array(y2)

#Logistic sample data with noise
x3 = np.array(list(range(1,20)))
y3 = []
for xVal in x3:
    y3.append(5/(1+2**(4*xVal + 3) + random.random()))
y3 = np.array(y3)


xData = [x1, x2, x3]
yData = [y1, y2, y3]
