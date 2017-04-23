import codecs
import numpy as np

class SentimentCorpus:
    
    def __init__(self, train_per=0.8, dev_per=0, test_per=0.2):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        X, y, feat_dict, feat_counts = build_dicts()
        self.nr_instances = y.shape[0]
        self.nr_features = X.shape[1]
        self.X = X
        self.y = y
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts
        
        train_y, dev_y, test_y, train_X, dev_X, test_X = split_train_dev_test(self.X, self.y, train_per, dev_per, test_per)
        self.train_X = train_X
        self.train_y = train_y
        self.dev_X = dev_X
        self.dev_y = dev_y
        self.test_X = test_X
        self.test_y = test_y

def split_train_dev_test(X, y, train_per, dev_per, test_per):
    if (train_per + dev_per + test_per) > 1:
        print "train/dev/test splits should sum to one"
        return
    dim = y.shape[0]
    split1 = int(dim * train_per)
    
    if dev_per == 0:
        train_y, test_y = np.vsplit(y, [split1])
        dev_y = np.array([])
        train_X = X[0:split1,:]
        test_X = X[split1:,:]
        dev_X = np.array([])
    else:
        split2 = int(dim*(train_per+dev_per))
        train_y,dev_y,test_y = np.vsplit(y,(split1,split2))
        train_X = X[0:split1,:]
        dev_X = X[split1:split2,:]
        test_X = X[split2:,:]
        
    return train_y,dev_y,test_y,train_X,dev_X,test_X

def build_dicts():
    '''
    builds feature dictionaries
    ''' 
    feat_counts = {}

    # build feature dictionary with counts
    nr_pos = 0
    with codecs.open("positive.review", 'r', 'utf8') as pos_file:
        for line in pos_file:
            nr_pos += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
    
    nr_neg = 0
    with codecs.open("negative.review", 'r', 'utf8') as neg_file:
        for line in neg_file:
            nr_neg += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    # remove all features that occur less than 5 (threshold) times
    to_remove = []
    for key, value in feat_counts.iteritems():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]

    # map feature to index
    feat_dict = {}
    i = 0
    for key in feat_counts.keys():
        feat_dict[key] = i
        i += 1

    nr_feat = len(feat_counts) 
    nr_instances = nr_pos + nr_neg
    X = np.zeros((nr_instances, nr_feat), dtype=float)
    y = np.vstack((np.zeros([nr_pos,1], dtype=int), np.ones([nr_neg,1], dtype=int)))
    
    with codecs.open("positive.review", 'r', 'utf8') as pos_file:
        nr_pos = 0
        for line in pos_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_pos,feat_dict[name]] = int(counts)
            nr_pos += 1
        
    with codecs.open("negative.review", 'r', 'utf8') as neg_file:
        nr_neg = 0
        for line in neg_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_pos+nr_neg,feat_dict[name]] = int(counts)
            nr_neg += 1
    
    # shuffle the order, mix positive and negative examples
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]
    
    return X, y, feat_dict, feat_counts











