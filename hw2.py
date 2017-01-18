import collections, csv, numpy, random, math
import matplotlib.pyplot as plt
from sys import argv

# ------------------------------------------------------------------------
# Expression data

class ExpressionProfile:
    """A sample id, label (for supervised), and profile (list of expression values)."""
    def __init__(self, id, label, values):
        self.id = id
        self.label = label
        self.values = values

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, pos):
        return self.values[pos]
        
    def __str__(self):
        return self.id + (':'+self.label if self.label is not None else '')

    def __repr__(self):
        return str(self)

    @staticmethod
    def read(filename, delimiter=','):
        """Read instances from the file, in delimited (default: comma-separated) format. 
        If the first column has a ':' in it, the name is the part before and the label is the part after; 
        else the name is the whole thing and the label is None.
        Args:
          filename (string): path to file
          delimeter (string): separates columns in file
        Returns:
          list of ExpressionProfile
        Examples:
          >>> simple = ExpressionProfile.read('tests/simple.csv')
        """
        reader = csv.reader(open(filename,'r'), delimiter=delimiter)
        profiles = []
        for row in reader:
            if row[0].find(':')<0: (id,label) = (row[0],None)
            else: (id,label) = row[0].split(':')
            values = numpy.array([float(v) for v in row[1:]])
            profiles.append(ExpressionProfile(id, label, values))
        return profiles

def plot_profiles(profiles):
    """Plot the profiles.
    Uses a symmetric blue-red colormap (blue for negative, red for positive)
    Args:
      profiles (list of ExpressionProfile): what to plot
    Examples:
      >>> plot_profiles(simple)
    """
    cmap = plt.cm.RdBu_r
    cmap.set_bad('k')
    matrix = numpy.vstack([profile.values for profile in reversed(profiles)])
    mmax = max(numpy.nanmax(matrix), abs(numpy.nanmin(matrix)))
    plt.pcolormesh(numpy.ma.masked_invalid(matrix), cmap=cmap, vmin=-mmax, vmax=mmax)
    plt.xlim(0, len(profiles[0]))
    plt.ylim(0, len(profiles))
    plt.xticks([])
    plt.yticks([])
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.colorbar()
    plt.show()

# ------------------------------------------------------------------------
# Unsupervised

class ExpressionCluster:
    """A binary tree with ExpressionProfiles at the leaves.
    For convenience, both inner nodes and leaves use the same class,
    but leaves have a profile field and inner nodes have left and right children.
    """

    def __init__(self, cluster_size):
        """Initialize the node. Customize this as you wish.
        Args:
          cluster_size: total number of leaves in this subtree
        """
        self.cluster_size = cluster_size

    @staticmethod
    def make_leaf(profile):
        """Make a leaf ExpressionCluster
        Args:
          profile (ExpressionProfile): what's at the leaf
        Returns:
          ExpressionCluster
        """
        t = ExpressionCluster(1)
        t.profile = profile
        return t
    
    @staticmethod
    def make_inner(left, right):
        """Make an inner ExpressionCluster
        Args:
          left, right (ExpressionCluster): children
        Returns:
          ExpressionCluster
        """
        t = ExpressionCluster(left.cluster_size + right.cluster_size)
        t.left = left; t.right = right
        return t
    
    def __str__(self):
        if self.cluster_size==1: return str(self.profile)
        return '('+str(self.left)+', '+str(self.right)+')'

    def pprint(self):
        """My best stab at an ASCII pretty-print of the cluster.
        Inspired by http://stackoverflow.com/questions/4965335/how-to-print-binary-tree-diagram
        """
        if self.cluster_size==1: 
            print str(self.profile)
        else:
            self.right.pprint_helper(True, ' ')
            print '-*'
            self.left.pprint_helper(False, ' ')

    def pprint_helper(self, isRight, indent):
        """Helper method for pprint; don't call directly.
        Args:
          isRight (boolean): is this a right child
          indent (string): to print before the node
        """
        angle = '/' if isRight else '\\'
        if self.cluster_size==1:
            print indent+' '+angle+'-'+str(self.profile)
        else:
            self.right.pprint_helper(True, indent + ('   ' if isRight else ' | '))
            print indent+' '+angle+'-*'
            self.left.pprint_helper(False, indent + (' | ' if isRight else '   '))

    def ordered_profiles(self):
        """The profiles in the leaves of the cluster tree, in order from left to right.
        This can be useful for generating the profile plot laid out to correspond to the tree.
        Returns:
          list of ExpressionProfile
        Examples:
          >>> plot_profiles(hierarchical(simple).ordered_profiles())
        """
        if self.cluster_size==1: return [self.profile]
        l = self.left.ordered_profiles()
        l.extend(self.right.ordered_profiles())
        return l


def hierDisCal(v1, v2, type, pool):
    hierDis = 0;
    sum = 0;
    listOfDis = []
    for x in v1:
        for y in v2:
            data1 = pool[x].profile
            data2 = pool[y].profile

            dis = calEucDis(data1, data2)

            if type == 'average':
                sum += dis
            else:
                listOfDis.append(dis)


    if type=='average':
        hierDis = sum/float(len(v1)*len(v2))
    elif type=='min':
        hierDis = min(listOfDis)
    elif type=='max':
        hierDis = max(listOfDis)

    return hierDis
    pass

def calEucDis(v1, v2):
    sum = 0;
    for x in range(0, len(v1)):
        sum += math.pow(v1[x] - v2[x], 2)

    return math.sqrt(sum)


def hierarchical(profiles, linkage='average'):
    """Hierarchically cluster the profiles.
    Args:
      profiles (list of ExpressionProfile): what to cluster
      linkage (string): how to evaluate between-cluster distances; choices = 'average', 'min', 'max'
    Returns:
      ExpressionCluster
    Examples:
      >>> hierarchical(simple).pprint()
    """
    if linkage=='stub': return hierarchical_stub(profiles)
    # TODO: your code here
    # Step 1: build leaf nodes
    leafNodes = []
    clusters = []
    for x in range(0, len(profiles)):
        leaf = ExpressionCluster.make_leaf(profiles[x])
        leafNodes.append(leaf)
        clusters.append({"node": leaf, "indices": [x]})


    # Step 2: build distance matrix
    numInClusters = 0;
    while(numInClusters<len(leafNodes)):
        dMatrix = []
        minVal = numpy.inf
        minPair = [0, 0]
        for i in range(0, len(clusters)):
            dMatrix.append([])
            for j in range(0, len(clusters)):
                if i!=j:
                    dis = hierDisCal(clusters[i]["indices"], clusters[j]["indices"], linkage, leafNodes)
                    if minVal>dis:
                        minVal = dis
                        minPair[0] = i
                        minPair[1] = j
                    dMatrix[i].append(dis)
                else:
                    dMatrix[i].append(0)

        newInnerNode = ExpressionCluster.make_inner(clusters[minPair[0]]["node"], clusters[minPair[1]]["node"])
        clusters[minPair[0]]["indices"] = clusters[minPair[0]]["indices"] + clusters[minPair[1]]["indices"]
        clusters[minPair[0]]["node"] = newInnerNode
        # for i in clusters[minPair[1]]:
        #     clusters[minPair[0]].append(i)
        #
        del clusters[minPair[1]]
        # # leafNodes[minPair[0]] = ExpressionCluster.make_inner(leafNodes[minPair[0]], leafNodes[minPair[1]])
        # # del leafNodes[minPair[1]]
        # ExpressionCluster.make_inner(leafNodes[minPair[0]], leafNodes[minPair[1]])
        size = 0;
        if len(clusters) == 1:
            size = clusters[0]["node"].cluster_size
        else:
            size = clusters[minPair[0]]["node"].cluster_size
        numInClusters = max(numInClusters, size)

    print 1
    return clusters[0]["node"]


def hierarchical_stub(profiles):
    """This is a stub for hierarchical to illustrate the input/output relationship.
    Sorts by label and just builds a binary tree by repeatedly splitting the data in half.
    Examples:
      >>> plot_profiles(hierarchical_stub(simple).ordered_profiles())
    """
    if len(profiles)==1: return ExpressionCluster.make_leaf(profiles[0])
    mid = int(len(profiles)/2)
    sorted_ps = sorted(profiles, key=lambda p: p.label)
    return ExpressionCluster.make_inner(hierarchical_stub(sorted_ps[:mid]), hierarchical_stub(sorted_ps[mid:]))

# ------------------------------------------------------------------------
# Supervised

def neighbors(instance, others, k):
    """The k nearest neighbors to instance among others.
    Args:
      instance (ExpressionProfile): who needs a neighbor
      others (list of ExpressionProfile): its possible neighbors
      k (int): how many to choose
    Returns:
      list of ExpressionProfile
    """
    # TODO: your code here
    pass

def knn(train, test, k):
    """Assign a class label to each test instance, based on its k-nearest neighbors in the training set.
    Args:
      train (list of ExpressionProfile): possible neighbors whose labels will be used
      test (list of ExpressionProfile): instances whose labels will be inferred from neighbors
      k (int): how many train neighbors to use, in order to label each test instance with the most popular label among neighbors
    Returns:
      list of string: labels for test instances, in same order
    """
    # TODO: your code here
    pass

def xval(profiles, k, nfold=5, nrep=10):
    """Cross-validate knn.
    Repeat:
    * split profiles into folds randomly
    * for each fold, use the other folds as training, assign test labels to the left-out fold, and see which are right
    Average the number of correct labels for each label, over the repetitions.
    Args:
      profiles (list of ExpressionProfile)
      k (int): for knn
      nfold (int): the number of roughly-equally-sized sets to split the profiles into
      nrep (int): the number of times to repeat the split, with random partitioning each time
      nfeatures (int or None): how many features to use, if not None
    Returns:
      dictionary of label : average # correct for that label, over the folds and repetitions
    Examples:
      >>> xval(simple, 1, 5, 10)
      {'A': 4.0, 'B': 4.0, 'C': 4.0}
      >>> xval(simple, 3, 5, 10)
      {'A': 2.2, 'B': 4.0, 'C': 4.0}   # your mileage may vary due to randomness
    """
    # TODO: your code here
    pass

def relabel(profiles):
    """A new set of profiles with labels shuffled up among the originals.
    Args:
       profiles (list of ExpressionProfile)
    Returns:
       list of ExpressionProfile: new instances with shuffled labels
    Examples:
       >>> relabel(simple)
       [0:A, 1:B, 2:C, 3:B, 4:C, 5:B, 6:A, 7:C, 8:C, 9:B, 10:A, 11:A]   # will vary due to randomness
    """
    # TODO: your code here
    pass

# ------------------------------------------------------------------------
# simplistic command-line driver
# call as one of these
#   hier <filename> <linkage>
#   knn <filename> <k> <nfold> <nrep>
#   perm <filename> <k> <nfold> <nrep>

# Command-line driver -- just some hard-coded test cases -- add your own if you want
profiles = ExpressionProfile.read("tests/simple.csv")
hierarchical(profiles, 'average').pprint()
'''if __name__ == '__main__':
    command = argv[1]
    profiles = ExpressionProfile.read(argv[2])
    if command=='hier':
        hierarchical(profiles, argv[3]).pprint()
    elif command=='knn':
        print xval(profiles, int(argv[3]), int(argv[4]), int(argv[5]))
    elif command=='perm':
        print xval(relabel(profiles), int(argv[3]), int(argv[4]), int(argv[5]))
    else:
        print('unknown command')'''
