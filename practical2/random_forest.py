import numpy as np
 
class DecisionNode:
    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 column: int = None,
                 value: float = None,
                 false_branch  = None,
                 true_branch = None,
                 is_leaf: bool = False):
        """
        Building block of the decision tree.

        :param data: numpy 2d array data can for example be
         np.array([[1, 2], [2, 6], [1, 7]])
         where [1, 2], [2, 6], and [1, 7] represent each data point
        :param labels: numpy 1d array
         labels indicate which class each point belongs to
        :param column: the index of feature by which data is splitted
        :param value: column's splitting value
        :param true_branch(false_branch): child decision node
        true_branch(false_branch) is DecisionNode instance that contains data
        that satisfies(doesn't satisfy) the filter condition.
        :param is_leaf: is true when node has no child

        """
        self.data = data
        self.labels = labels
        self.column = column
        self.value = value
        self.false_branch = false_branch
        self.true_branch = true_branch
        self.is_leaf = is_leaf


class DecisionTree:

    def __init__(self,
                 max_tree_depth=4,
                 criterion="gini",
                 task="classification",
                 discrete_columns = [],
                 threshold=0.55):
        self.tree = None
        self.max_depth = max_tree_depth
        self.task = task
        self.discrete_columns = discrete_columns
        self.threshold = threshold

        if criterion == "entropy":
            self.criterion = self._entropy
        elif criterion == "square_loss":
            self.criterion = self._square_loss
        elif criterion == "gini":
            self.criterion = self._gini
        else:
            raise RuntimeError(f"Unknown criterion: '{criterion}'")

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        """
        Gini criterion for classification tasks.

        """
        gini = 0
        for elem in np.unique(labels):
            curr_elem = labels[labels == elem]
            gini += (curr_elem.size / labels.size)**2
        return 1 - gini
        
    
    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        """
        Entropy criterion for classification tasks.

        """
        entropy = 0
        for elem in np.unique(labels):
            curr_elem = labels[labels == elem]
            p_elem = curr_elem.size / labels.size
            entropy += p_elem * np.log2(p_elem)
            
        return (-1) * entropy
        
    @staticmethod
    def _square_loss(labels: np.ndarray) -> float:
        """
        Square loss criterion for regression tasks.

        """
        return np.var(labels)
        
    def _iterate(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 current_depth=0) -> DecisionNode:
        """
        This method creates the whole decision tree, by recursively iterating
         through nodes.
        It returns the first node (DecisionNode object) of the decision tree,
         with it's child nodes, and child nodes' children, ect.
        """

        if len(labels) == 1:
            return DecisionNode(data, labels, is_leaf = True)
            
        impurity = self.criterion(labels)
        best_column, best_value = None, None
        
        gain = impurity
        right_node_index = []
        left_node_index = []
                
        for column, column_values in enumerate(data.T):
            if column in self.discrete_columns:
                for_arr = np.unique(column_values)
            else:
                try:
                    for_arr = np.arange(min(column_values), 
                                         max(column_values), 
                                         (max(column_values) - min(column_values)) / 40)  
                except:
                    continue
            for split_value in for_arr:
                right = column_values >= split_value
                index_right = np.where(right)[0]
                satisfy_condition_right = np.take(labels, index_right)

                left = column_values < split_value
                index_left = np.where(left)[0]
                satisfy_condition_left = np.take(labels, index_left)

                impurity_right = self.criterion(satisfy_condition_right)
                impurity_left = self.criterion(satisfy_condition_left)

                nm_right = satisfy_condition_right.size / column_values.size
                nm_left = satisfy_condition_left.size / column_values.size

                curr_gain = nm_left * impurity_left + nm_right * impurity_right 
                
                if(curr_gain < gain and curr_gain != 0):
                    gain = curr_gain
                    best_column = column
                    best_value = split_value
                    right_node_index = index_right
                    left_node_index = index_left
        
        if best_column is None or current_depth == self.max_depth:
            return DecisionNode(data, labels, is_leaf=True)
        else:
            left_node_data = data[left_node_index]
            left_node_labels = labels[left_node_index]
            
            if(left_node_labels.size == 0):
                return DecisionNode(data, labels, is_leaf=True)
            
            left_node = self._iterate(left_node_data, left_node_labels, current_depth+1)
            
            right_node_data = data[right_node_index]
            right_node_labels = labels[right_node_index]
            
            if(right_node_labels.size == 0):
                return DecisionNode(data, labels, is_leaf=True)
           
            right_node = self._iterate(right_node_data, right_node_labels, current_depth+1)
              
            return DecisionNode(data, labels, best_column, best_value, left_node, right_node)
        
    def fit(self, data: np.ndarray, labels: np.ndarray):
        self.tree = self._iterate(data, labels)

    def predict(self, point: np.ndarray) -> float or int:
        """
        This method iterates nodes starting with the first node i. e.
        self.tree. Returns predicted label of a given point (example [2.5, 6],
        where 2.5 and 6 are points features).

        """
        node = self.tree

        while True:
            if node.is_leaf:
                if self.task == "classification":
                    counts = np.bincount(node.labels)
                    if(len(counts) == 1):
                        return node.labels[0]
                    else:
                        if(counts[1]/len(node.labels) >= self.threshold):
                            return 1
                        return 0
                else:
                    return np.mean(node.labels)
            
            if point[node.column] >= node.value:
                node = node.true_branch
            else:
                node = node.false_branch



class RandomForestClassifier(object):
    def __init__(self,
                 n_esimators = 20,
                 max_depth = 12,
                 thetta = 80,
                 betta = 70,
                 task = 'classification',
                 criteria = 'gini',
                 discrete_columns = [1, 2, 3, 8, 7, 9, 10, 11, 5],
                 threshold=0.55):
        
        self.n_esimators = n_esimators
        self.max_depth = max_depth
        self.thetta = thetta
        self.betta = betta
        self.task = task
        self.criteria = criteria
        self.discrete_columns = discrete_columns
        self.estimators = []
        self.threshold=threshold
        
        for i in range(n_esimators):
            tree = DecisionTree(max_depth, criteria, task, discrete_columns)
            self.estimators.append(tree)
            
            
    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        :param data: array of features for each point
        :param labels: array of labels for each point
        """
        
        self.estimators_columns = []
        for i in range(self.n_esimators):
            row_size = int((self.thetta / 100) * len(data))
            row_indexes = np.random.choice(len(data), size=row_size, replace=False)
            column_size = int((self.betta / 100) * data.shape[1])
            column_indexes = np.random.choice(data.shape[1], size=column_size, replace=False)
            new_data = (data[row_indexes])[:,column_indexes]
            new_labels = labels[row_indexes]
            self.estimators_columns.append(column_indexes)
            self.estimators[i].fit(new_data, new_labels)
            

    def predict(self, data: np.ndarray) -> np.ndarray:
        predicts = np.ndarray(len(data))
        
        for point_index in range(len(data)):
            curr_predict = []
            i = 0 
            for esimator in self.estimators:
                curr_predict.append(esimator.predict(data[point_index,self.estimators_columns[i]]))
                i += 1                    
            if self.task == 'classification':
                counts = np.bincount(curr_predict)
                if(len(counts)==1):
                    predicts[point_index] = curr_predict[0]
                else:
                    if(counts[1]/len(curr_predict)>=self.threshold):
                        predicts[point_index] = 1
                    else:
                        predicts[point_index] = 0
            else:
                predicts[point_index] = (np.mean(curr_predict))    
        return predicts


def f1_score(y_true: np.ndarray, y_predicted: np.ndarray):
    """
    only 0 and 1 should be accepted labels and 1 is the positive class
    """
    
    tp =0 
    fn = 0
    fp = 0
    
    for elem_true, elem_predict in zip(y_true, y_predicted):
        if elem_true == elem_predict:
            if elem_true == 1:
                tp +=1
        else:
            if elem_true == 1:
                fn += 1
            else:
                fp += 1
    
    if(tp + fp == 0):
        return 0
    else:
        precision = tp / (tp + fp)
            
    if(tp + fn == 0):
        return 0
    else:
        recall = tp /(tp + fn)
    
    return 2 * ((precision * recall) / (precision + recall))

def data_preprocess(data: np.ndarray) -> np.ndarray:
    new_data = pd.DataFrame(data,columns=['age','foundation_type','roof_type','position','loc_id',
                                         'num_floors','area','height','num_families','ownership_type',
                                         'configuration','surface_condition'])
    not_numeric_columns = ['foundation_type', 'roof_type', 'position', 
                           'ownership_type', 'configuration', 'surface_condition']
    
    dictionaries = [{'w': 0, 'r': 1, 'i': 2, 'h': 3, 'u': 4},
                    {'q': 0, 'n': 1, 'x': 2},
                    {'s': 0, 'j': 1, 't': 2, 'o': 3},
                    {'v': 0, 'a': 1, 'r': 2, 'w': 3},
                    {'d': 0, 'q': 1, 'o': 2, 'c': 3, 'u': 4, 'f': 5, 'n': 6, 's': 7, 'a': 8, 'm': 9},
                    {'t': 0, 'n': 1, 'o': 2}]
    
    for i in range(len(not_numeric_columns)):
        new_data[not_numeric_columns[i]] = new_data[not_numeric_columns[i]].map(dictionaries[i])
    return np.array(new_data)
