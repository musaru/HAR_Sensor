import numpy as np
import os
import argparse
import scipy.io as scio
import sys 
import logging
from scipy import stats
from collections import Counter
import random
import json
from sklearn.model_selection import train_test_split
import logging

#from DataSplitting import create_user_data,creat_json, random_creat_json

dataset_dir='F:/PostDoctoral_Research/HAR/Sensor/Dataset/USC-HAD/USC-HAD'
stored_path='F:/PostDoctoral_Research/HAR/Sensor/Dataset/USC-HAD/Processed_Data'
import os
dir_path = os.getcwd()
print(dir_path)

stdv=2
imbalance=False
sample_size=256
min_samp=20
k_shots=20


WINDOW_SIZE=512 
OVERLAP_RATE=0.0
NUM_USERS = 14
NUM_LABELS = 12
def create_user_data( user_id ,num_users, reshaped_segments , labels , stdv  ):
    X = []
    Y = []

    for i in range( num_users ):
        index = np.where( user_id == i )

        ############################# Pathological distribution #########################
        user_labels =  np.unique(labels[index[0]])
        print(f'the re labels for {i} user is  ' , user_labels )

        num_labels = len( np.unique(labels[index[0]]) )
        num_classes =  num_labels - stdv
        activity_sample = np.random.choice(np.unique(labels[index[0]]), max(1,num_classes),  replace=False)
        print(f'activity_sample for {i} user  is ' ,  activity_sample)
        
        activity_index = np.empty(0,dtype=int)
        for elem in activity_sample:
            activity_index = np.append( activity_index ,  np.where( labels[index[0]] == elem )[0] )
        X.append(reshaped_segments[index[0]][activity_index]   )
        Y.append(labels[index[0]][activity_index] )
        
        print(f'now the labels for {i} user is ' , np.unique( labels[index[0]][activity_index]  ))
    return X,Y

def create_user_data_for_one_idx( user_id ,num_users, reshaped_segments , labels  , stdv = 0 ):
    X = []
    Y = []

    for i in range( num_users ):
        index = np.where(user_id == i + 1 )

    #############################Pathological distribution#########################    
        num_labels = len(np.unique(labels[index[0]]))
        num_classes =  num_labels - stdv
        activity_sample = np.random.choice(np.unique(labels[index[0]]), max(2,num_classes),  replace=False)        # print(shared)
        print( 'activity_sample is ' ,  activity_sample)
        activity_index = np.empty(0,dtype=int)
        for elem in activity_sample:
            activity_index = np.append( activity_index ,  np.where( labels[index[0]] == elem )[0] )
        X.append(reshaped_segments[index[0]][activity_index]   )
        Y.append(labels[index[0]][activity_index] -1)
        print('the  re labels is ' , np.unique(labels[index[0]] -1 ) )
        print('now the labels is ' , np.unique( labels[index[0]][activity_index] -1 ))
    return X,Y



def CountLabels( uname , y_train ,logging, mode = 'train'):
    labels = {}
    labels_num = {}
    count_y = Counter( y_train )
    for idx in count_y:
        labels_idxs = np.where( y_train == idx)[0]
        labels[idx] = [ count_y[idx] , labels_idxs ]#当前标签的总数，对应标签的索引
        labels_num[str(idx)] = count_y[idx]
        logging.info(f'the {mode} user {uname} contains the label {idx} samples is {count_y[idx]}')
    return labels,labels_num
def slide_window(array, w_s, stride):
    '''
    滑窗处理
    array: ---
    w_s: 窗口大小
    stride： 滑动步长
    '''
    x = []
    times = (array.shape[0] - w_s) // stride + 1
    i=0
    for i in range(times):
        x.append(array[stride*i: stride*i+w_s]) 
    #最后一个保留处理 
    if stride*i+w_s < array.shape[0]-1:
        x.append(array[-w_s:])
    return x
def shot_data( X , y , label_idx ,logging, k_shots ,stdv, split_radio=0.7 ):
    trainx , trainy , testx , testy = [],[],[],[]

    for idx in label_idx:
        len_this_label = int(label_idx[idx][0] * split_radio)
        train_size = len_this_label if len_this_label < k_shots else  k_shots
        if train_size != 0:
            if stdv != 0 and train_size == k_shots:
                train_size = np.random.randint( max( 1, train_size - stdv ) ,  train_size + stdv )        

            test_size= max( int(train_size*(1-split_radio) / split_radio ) , 1 )
            shffle_idxs = label_idx[idx][1]
            assert len(shffle_idxs) == label_idx[idx][0] 
            random.shuffle( shffle_idxs )
            if len(shffle_idxs) != 1:
            # logging.info(f'==>label {idx} of  k_shots is {k_shots} ')
                if trainx == []:
                    # print('!!!!!')
                    trainx = X[ shffle_idxs[:train_size] ]
                    trainy = y[ shffle_idxs[:train_size] ]
                    testx  = X[ shffle_idxs[train_size:train_size+test_size] ]
                    testy  = y[ shffle_idxs[train_size:train_size+test_size] ]
                else:
                    trainx = np.concatenate( (trainx ,X[ shffle_idxs[:train_size] ] ) )
                    trainy = np.concatenate( (trainy ,y[ shffle_idxs[:train_size] ] ) )
                    testx  = np.concatenate( (testx  ,X[ shffle_idxs[train_size:train_size+test_size] ] ) )
                    testy  = np.concatenate( (testy  ,y[ shffle_idxs[train_size:train_size+test_size] ] ) )
            else:
                pass
            logging.info(f'==>train label {idx} of length is {train_size} ')
        else:
            logging.info(f'==>label {idx} is only one sample so discarded')
    return trainx , trainy , testx , testy
def ShotDataForAllSamples( X , y , label_idx ,logging, k_shots ,stdv, split_radio=0.7 ):
    trainx , trainy , testx , testy = [],[],[],[]

    for idx in label_idx:
        len_this_label = label_idx[idx][0] if label_idx[idx][0] <= k_shots else  k_shots
        if len_this_label != 1:
            if stdv != 0:
                # print(max( 1, len_this_label - stdv + 1) ,  min( len_this_label + stdv - 1 , label_idx[idx][0] ))
                len_this_label = np.random.randint( max( 1, len_this_label - stdv + 1) ,  min( len_this_label + stdv - 1 , label_idx[idx][0] ) )        
            train_size = max( int( len_this_label * split_radio ) , 1 )
            test_size  = len_this_label - train_size 
            shffle_idxs = label_idx[idx][1]
            assert len(shffle_idxs) == label_idx[idx][0] 
            random.shuffle( shffle_idxs )
            if len(shffle_idxs) != 1:
            # logging.info(f'==>label {idx} of  k_shots is {k_shots} ')
                if trainx == []:
                    # print('!!!!!')
                    trainx = X[ shffle_idxs[:train_size] ]
                    trainy = y[ shffle_idxs[:train_size] ]
                    testx  = X[ shffle_idxs[train_size:train_size+test_size] ]
                    testy  = y[ shffle_idxs[train_size:train_size+test_size] ]
                else:
                    trainx = np.concatenate( (trainx ,X[ shffle_idxs[:train_size] ] ) )
                    trainy = np.concatenate( (trainy ,y[ shffle_idxs[:train_size] ] ) )
                    testx  = np.concatenate( (testx  ,X[ shffle_idxs[train_size:train_size+test_size] ] ) )
                    testy  = np.concatenate( (testy  ,y[ shffle_idxs[train_size:train_size+test_size] ] ) )
            else:
                pass
            logging.info(f'==>label {idx} of length is {len_this_label} ')
        else:
            logging.info(f'==>label {idx} is only one sample so discarded')
    return trainx , trainy , testx , testy


def creat_json( X,y,train_path , test_path ,k_shots ,stdv , logging_path, num_users=10  ):
    data_logging = {}
    num_samples  = {'train':[],'test':[]}
    logging.basicConfig(filename=os.path.join( os.path.dirname(os.path.dirname(train_path)),\
        f'dataInfo_{str(num_users)}u_{str(stdv)}p.log'), level=logging.DEBUG , filemode='w') 
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt='%m-%d %H:%M')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    
    logger.addHandler(sHandler)
    for i in range( num_users ):
        # print(f'==>original k_shots is{k_shots}')

        uname = i #'f_{0:05d}'.format(i)
        # X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.5 , stratify=y[i] ,random_state= 42 )   
        user_labels, user_label_num= CountLabels( uname , y[i],logging=logger,mode = 'all')
        X_train , y_train ,  X_test , y_test= shot_data( X[i] , y[i] , user_labels , logger , k_shots , stdv = stdv )
        user_train_labels, user_train_label_num= CountLabels( uname , y_train,logging=logger)
        user_test_labels,_ = CountLabels(uname , y_test , mode='test',logging=logger )        
        data_logging[str(uname)] = user_train_label_num
        count_y = Counter( y_train )
        # for idx in count_y:
        #     print(f'the user{uname} contains the label {idx} samples is {count_y[idx]}')
        print('************************************************************')
        train_data = {'x': X_train, 'y': y_train}
        num_samples['train'].append(len(y_train))
        with open(train_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=train_data,  allow_pickle=True)
        test_data  = {'x': X_test, 'y': y_test}
        num_samples['test'].append(len(y_test))
        with open(test_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data, allow_pickle=True)
        
    # print(data_logging)    
    logger.info(f"train {num_samples['train']}")
    logger.info(f"test {num_samples['test']}")
    logger.info(f"Num_samples:{num_samples['train'] + num_samples['test']}")
    logger.info(f"Total_samples:{sum(num_samples['train'] + num_samples['test'])} ,Train_samples:{sum(num_samples['train'])} , Test_samples:{sum(num_samples['test'])}" )
    with open(logging_path,'w') as outfile:
        json.dump(data_logging, outfile )    


def delete_sparse_class(x,y, min_samples = 3 ):
    count_y = Counter( y )
    # print(count_y)
    labels_idxs = np.arange(len(y))
    delete_labels_idxs = None
    for idx in count_y:
        if count_y[idx]  < min_samples:
            if delete_labels_idxs is not None:
                delete_labels_idxs = np.concatenate([delete_labels_idxs,np.where( y == idx)[0]])
            else:
                delete_labels_idxs = np.where( y == idx)[0]
            print(f'the class {idx} of original length is {count_y[idx]}, deleting the class {idx}')
    print(f'deleting idxs of the classes is {delete_labels_idxs}')
    if delete_labels_idxs is not None:
        labels_idxs = np.delete(labels_idxs,delete_labels_idxs)
    return x[labels_idxs],y[labels_idxs]


def random_creat_json( X,y, train_path , test_path , logging_path, sample_size = 280 , num_users=10 , min_sample = 10  ):
    '''
    imbalance distribution . pelease generating balance distribution by use 'creat_json' function with alpha = 0.
    '''
    data_logging = {}
    num_samples  = {'train':[],'test':[]}
    logging.basicConfig(filename=os.path.join( os.path.dirname(os.path.dirname(train_path)),\
        f'dataInfo_{str(num_users)}u_{str(sample_size)}l.log'), level=logging.DEBUG , filemode='w') 
    logger = logging.getLogger()
    formatter = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt='%m-%d %H:%M')

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    
    logger.addHandler(sHandler)
    for i in range( num_users ):
        uname = i #'f_{0:05d}'.format(i)
        imbalance_sample_size   = np.random.randint(min_sample,sample_size)
        shuffle_idxs            = np.arange(len(X[i]))
        np.random.shuffle( shuffle_idxs )
        
        imbalance_sample_idxs   = shuffle_idxs[:imbalance_sample_size] if len(X[i]) > imbalance_sample_size else np.arange(len(X[i]))
        # print(len(X[i]),imbalance_sample_idxs)
        random_x                = X[i][ imbalance_sample_idxs ]
        random_y                = y[i][ imbalance_sample_idxs ]
        # print(len(random_x),len(random_y))
        random_x , random_y     = delete_sparse_class(random_x , random_y)
        X_train, X_test, y_train, y_test            = train_test_split( random_x , random_y , test_size=0.2 , stratify=random_y ,random_state= 42 )   
        user_train_labels,  user_train_label_num    = CountLabels( uname , y_train,logging=logger)
        user_test_labels ,  _                       = CountLabels(uname , y_test , mode='test',logging=logger )        
        data_logging[str(uname)]    = user_train_label_num
        count_y = Counter( y_train )
        # for idx in count_y:
        #     print(f'the user{uname} contains the label {idx} samples is {count_y[idx]}')
        print('************************************************************')
        train_data = {'x': X_train, 'y': y_train}
        num_samples['train'].append(len(y_train))
        with open(train_path + str(i) + '.npz', 'wb') as f:
             np.savez_compressed(f, data=train_data, allow_pickle=True)
            
        test_data  = {'x': X_test, 'y': y_test}
        num_samples['test'].append(len(y_test))
        with open(test_path + str(i) + '.npz', 'wb') as f:
            np.savez_compressed(f, data=test_data, allow_pickle=True)
        
    logger.info(f"train {num_samples['train']}")
    logger.info(f"test {num_samples['test']}")
    logger.info(f"Num_samples:{num_samples['train'] + num_samples['test']}")
    logger.info(f"Total_samples:{sum(num_samples['train'] + num_samples['test'])} ,Train_samples:{sum(num_samples['train'])} , Test_samples:{sum(num_samples['test'])}" )
    with open(logging_path,'w') as outfile:
        json.dump(data_logging, outfile )   
def merge_data(path, w_s, stride):
    '''
    所有数据按类别进行合并
    path: 原始 USC_HAD 数据路径
    w_s： 指定滑窗大小
    stride： 指定步长
    '''
    result = [] # 12类，按索引放置每一类数据
    '''对每一个数据进行滑窗处理，将滑窗后的数据按类别叠加合并放入result对应位置'''
    subject_list = os.listdir(path)
    subject_list = [ subject for subject in subject_list if subject.find('Subject')!=-1 ]
    os.chdir(path)
    user_id =[]
    labels = []
    print(subject_list)
    for i , subject in enumerate( subject_list):
        print(i)
        if not os.path.isdir(subject):
            continue
        print('======================================================\n         current Subject sequence: 【%s】\n'%(subject))
        mat_list = os.listdir(subject)
        os.chdir(subject)

        for mat in mat_list:
            category = int(mat[1:-6])-1 #获取类别
            content = scio.loadmat(mat)['sensor_readings']
            

            x = slide_window(content, w_s, stride)
            # print(x.)
            result.extend(x)
            user_id.extend([i]*len(x))
            labels.extend([category]*len(x))

        os.chdir('../')
    os.chdir('../')
    print(len(user_id) ,len(result), len(labels))
    return np.array(user_id,dtype=int) , np.expand_dims(np.array(result),axis=1 ), np.array( labels , dtype=int)


def process_data(dataset_dir):
    user_id , reshaped_segments ,  labels = merge_data(
                                    path = dataset_dir, 
                                    w_s = WINDOW_SIZE, 
                                    stride = int(WINDOW_SIZE*(1-OVERLAP_RATE))  )
    return user_id , reshaped_segments, labels


#def main():
parser = argparse.ArgumentParser()
parser.add_argument("--n_class", type=int, default=NUM_LABELS, help="number of classification labels")
parser.add_argument("--min_sample", type=int, default=20, help="Min number of samples per user.")
parser.add_argument("--n_ways", type=int, default=NUM_LABELS, help="n ways")
parser.add_argument("--k_shots", type=int, default=20, help="k shot for train samples")
parser.add_argument("--stdv", type=int, default=2, help="noise for choosing k shots and deleting n ways")
parser.add_argument("--n_user", type=int, default=NUM_USERS,
                    help="number of local clients, should be muitiple of 10.")
parser.add_argument("--dataset_dir", type=str, default='F:/PostDoctoral_Research/HAR/Sensor/Dataset/USC-HAD/USC-HAD')
parser.add_argument('--imbalance',action='store_true',default=True)
parser.add_argument("--sample_size", type=int, default=256, help="Min number of samples per user.")
args = parser.parse_args()
print()
print("Number of classes: {}".format(args.n_class))
print("stdv for noisy: {}".format(args.stdv))
print("n_ways: {}".format(args.n_ways))
print("k_shots: {}".format(args.k_shots))
#dir_path = os.path.dirname(os.path.realpath(__file__))
#dir_path = os.path.dirname(dir_path)
#stored_path=stored_path
#print(dir_path)
if args.imbalance:
    # assert args.stdv == 0 , print('do not delete some classes')
    train_path      =  os.path.join( stored_path,f'uschad/{args.n_user}u_{args.sample_size}l_data/train/')
    test_path       =  os.path.join( stored_path,f'uschad/{args.n_user}u_{args.sample_size}l_data/test/')
    logging_path    =  os.path.join( stored_path,'uschad/VisualDataDistribution',f'{args.n_user}u_{args.sample_size}l_logging.json')
else:
    train_path      =  os.path.join( stored_path,f'uschad/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/train/')
    test_path       =  os.path.join(stored_path,f'uschad/{args.k_shots}k_{args.n_user}b_{args.stdv}p_data/test/')
    logging_path    =  os.path.join( stored_path,'uschad/VisualDataDistribution',f'{args.k_shots}k_{args.n_user}b_{args.stdv}p_logging.json')

dir_path = os.path.dirname(train_path)
for path in (train_path,test_path,logging_path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

user_id , segments, labels = process_data(dataset_dir=args.dataset_dir)
print('useid num is ', len(user_id), 'segments num is ' , len(segments))

user_data_x,user_data_y = create_user_data( user_id ,NUM_USERS, segments , labels  ,stdv=args.stdv )



if args.imbalance:
    random_creat_json(user_data_x , user_data_y , train_path , test_path, logging_path=logging_path,sample_size=args.sample_size,num_users=NUM_USERS,min_sample=args.min_sample)
else:
    creat_json( user_data_x , user_data_y , train_path , test_path, k_shots=args.k_shots , stdv=args.stdv, logging_path=logging_path,num_users=NUM_USERS) 
print("Finish Generating Samples")

#if __name__ == '__main__':


import numpy as np
#npz_dir = "F:/PostDoctoral_Research/HAR/Sensor/Dataset/USC-HAD/Processed_Data/uschad/20k_14b_2p_data/test"
npz_dir = "F:/PostDoctoral_Research/HAR/Sensor/Dataset/USC-HAD/Processed_Data/uschad/14u_256l_data/train"
last_folder_name = os.path.basename(npz_dir)
print(last_folder_name)
X_list = []
y_list = []

# Loop through each .npz file and load the data
for i in range(14):  # Assuming you have 14 files named from 0.npz to 13.npz
    with np.load(npz_dir + '/' + str(i) + '.npz', allow_pickle=True) as data:
        train_data = data['data'].item()  # .item() is used to get the dictionary
        X_train_loaded = train_data['x']
        y_train_loaded = train_data['y']
        
        # Append the loaded data and labels to the lists
        X_list.append(X_train_loaded)
        y_list.append(y_train_loaded)

# Concatenate the data and labels along the first axis
X_merged = np.concatenate(X_list, axis=0)
y_merged = np.concatenate(y_list, axis=0)

# Verifying the merged data dimensions
print("X_merged shape:", X_merged.shape)  # Should print: (112*14, 1, 512, 6)
print("y_merged shape:", y_merged.shape)  # Should print: (112*14,)

# Save the merged data to a new .npz file
output_path = npz_dir + '/'+last_folder_name+'_merged_data.npz'
with open(output_path, 'wb') as f:
    np.savez_compressed(f, data={'x': X_merged, 'y': y_merged}, allow_pickle=True)

print("Merged data saved to", output_path)