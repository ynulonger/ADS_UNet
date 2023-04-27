import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metric
from skimage import feature as skft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def LBP_extractor(split):
    radius = 2 # hypermeter
    n_point = radius * 8 # # hypermeter
    features = 18
    dataset = np.empty([0,features])
    labels = []
    writer = pd.ExcelWriter(f'data.xlsx')
    max_unique = 0
    class_list = ['blanket1', 'blanket2', 'canvas1', 'ceiling1', 'ceiling2', 'cushion1', 'floor1', 
                'floor2', 'grass1', 'lentils1', 'linseeds1', 'oatmeal1', 'pearlsugar1', 'rice1', 
                'rice2', 'rug1', 'sand1', 'scarf1', 'scarf2', 'screen1', 'seat1', 
                'seat2', 'sesameseeds1', 'stone1', 'stone2', 'stone3', 'stoneslab1', 'wall1']
    for i, c in zip([idx for idx in range(28)], class_list):
        image_list = glob.glob('/ssd/kylberg/texture-imgs/'+c+'*')
        image_list.sort()
        if split=='train':
            image_list = image_list[:100]
        elif split=='test':
            image_list = image_list[100:]
        for image_name in image_list:
            image = cv2.imread(image_name, flags=0)
            image = cv2.resize(image, (512,512))
            # image = (image-127)/127.0
            lbp = skft.local_binary_pattern(image, n_point, radius, 'uniform') # LBP
            # print('class:',i, 'length:',len(np.unique(lbp)), 'lbp:',np.unique(lbp))
            if max_unique<len(np.unique(lbp)):
                max_unique = len(np.unique(lbp))
            # max_bins = int(lbp.max() + 1)  # 256 histogram
            max_bins = features # histogram
            hist, bins = np.histogram(lbp, bins=max_bins, range=(0, max_bins))
            hist = np.reshape(np.array(hist),[1,-1])/262144
            dataset = np.concatenate([dataset, hist], axis=0)
            labels.append(i)
    Labels = pd.DataFrame(np.reshape(np.array(labels),[-1,1]))
    Dataset = pd.DataFrame(dataset)
    writer = pd.ExcelWriter(f'dataset.xlsx')
    Dataset.to_excel(writer, 'hist', float_format='%.8f')
    Labels.to_excel(writer, 'labels', float_format='%.8f')
    writer.save()
    writer.close()
    print('max_unique:', max_unique)
    return dataset, labels

def get_confusion_matrix(cm, model):
    classes=28
    labels = [str(i) for i in range(classes)]
    acc = np.sum(cm*np.eye(classes))/np.sum(cm)
    print('acc',acc)

    xlocations = np.array(range(len(labels)))
    plt.figure(figsize=(18,19))
    plt.xticks(xlocations, labels,fontsize=12)
    plt.yticks(xlocations, labels,fontsize=12)
    plt.title(model+', Accuracy: '+'%.3f' %(acc*100)+'%', family='fantasy', fontsize=15)
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.tight_layout()
    cm = cm/np.sum(cm,axis=1)
    x, y = np.meshgrid([i for i in range(len(labels))],[i for i in range(len(labels))])
    for x,y in zip(x.flatten(), y.flatten()):
        prob = cm[x,y]
        if x==y:
            plt.text(x,y,"%00.1f" %(prob*100,), color='green', fontsize=12, va='center', ha='center')
    plt.imshow(cm, cmap=plt.cm.jet)
    plt.colorbar(fraction=0.045)
    plt.savefig(f'{model}_cm.png')


train_data, train_label = LBP_extractor('train')
test_data, test_label = LBP_extractor('test')

# train_data, train_label = LBP_extractor('train')
# test_data, test_label = LBP_extractor('test')

print(f'train: {train_data.shape[0]}, test: {test_data.shape[0]}')
model= RandomForestClassifier(random_state = 50,n_estimators = 100) # random forest
model.fit(train_data, train_label)
predictions = model.predict(test_data)
cm = metric.confusion_matrix(test_label, predictions)
get_confusion_matrix(cm, 'Random_Forest')
acc = metric.accuracy_score(test_label, predictions)
# AP = metric.average_precision_score(test_label, predictions)
# print(f'acc:{acc}, AP:{acc}')
print(f'acc:{acc}')


