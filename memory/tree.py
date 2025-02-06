"""
Reference: https://github.com/Howuhh/prioritized_experience_replay.git
"""

class SumTree:
    """
    A binary tree data structure where the value of each node is the sum of the values of its children.
    It is used to store and efficiently retrieve cumulative sums.
    
    Parameters
    ----------
    size : int
        The size of the data array.
        
    Attributes
    ----------
    nodes : list of float
        The list of nodes in the tree, where each node is the sum of its children.
    data : list
        The list of data items corresponding to the leaf nodes.
    size : int
        The size of the data array.
    count : int
        The current count of added elements.
    real_size : int
        The actual number of elements in the tree.
    """
    
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size
        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]
        
    def update(self, data_idx, value):
        """
        Update the value at a specific index and propagate the change up the tree.
        
        Parameters
        ----------
        data_idx : int
            The index of the data to be updated.
        value : float
            The new value to be set at the specified index.
            
        Notes
        -----
        This method updates the value at the given index and adjusts the parent nodes
        to reflect the change in the tree structure.
        """
        
        idx = data_idx + self.size - 1
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2
        
    def add(self, value, data):
        """
        Add a value and associated data to the tree.
        
        Parameters
        ----------
        value : any
            The value to be added.
        data : any
            The data associated with the value.
            
        Notes
        -----
        This method updates the internal data structure with the given value and data,
        increments the count, and adjusts the real size of the tree.
        """
        
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right, = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]
            
        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]
    
    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
