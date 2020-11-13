import argparse
from bs4 import BeautifulSoup
import os
import pickle
import time
import numpy as np

def intersections(real_boxes, pred_boxes):
    cont = 0
    for box in real_boxes:
        for cur_box in pred_boxes:
            if (box[0] > cur_box[0] and box[0] < cur_box[2]) or (box[2] > cur_box[0] and box[2] < cur_box[2]):
                if (box[1] > cur_box[1] and box[1] < cur_box[3]) or (box[3] > cur_box[1] and box[3] < cur_box[3]):
                    cont += 1
                    continue
                if (cur_box[1] > box[1] and cur_box[1] < box[3]) or (cur_box[3] > box[1] and cur_box[3] < box[3]):
                    cont += 1
                    continue
            if (cur_box[0] > box[0] and cur_box[0] < box[0]) or (cur_box[2] > box[0] and cur_box[2] < box[2]):
                if (box[1] > cur_box[1] and box[1] < cur_box[3]) or (box[3] > cur_box[1] and box[3] < cur_box[3]):
                    cont += 1
                    continue
                if (cur_box[1] > box[1] and cur_box[1] < box[3]) or (cur_box[3] > box[1] and cur_box[3] < box[3]):
                    cont += 1
                    continue
    return cont

parser = argparse.ArgumentParser()
parser.add_argument('--annotations')
parser.add_argument('--predictions')
parser.add_argument('--conf')
parser.add_argument('--results')
opt = parser.parse_args()
print(opt)
path_annotations = opt.annotations
path_predictions = opt.predictions
prob = opt.conf

total_boxes = 0
boxes = {}
for file in os.listdir(path_annotations):
    if file[0] == '.':
        continue
    if file not in boxes:
        boxes[file] = []
    tags_file = open(os.path.sep.join([path_annotations, file]), 'r')
    tags_file = BeautifulSoup(tags_file.read())
    for i in tags_file.find_all('object'):
        aux = (int(i.xmin.text), int(i.ymin.text),
                int(i.xmax.text), int(i.ymax.text))
        total_boxes += 1
        boxes[file].append(aux)

with open(path_predictions, 'rb') as f:
    preds = pickle.load(f)

aux_preds = {}
for file in preds:
    aux_preds[file.split('/')[-1].split('.')[0]] = preds[file]

preds = aux_preds


false_neg = 0
true_pos = 0
false_pos = 0
true_neg = 0
intersect = 0

for file in boxes:
    if len(boxes[file]) == 0:
        if file.split('.')[0] in preds:
            false_pos += 1
        else:
            true_neg += 1
    else:
        if file.split('.')[0] not in preds:
            false_neg += 1
        else:
            true_pos += 1
            intersect += intersections(preds[file.split('.')[0]], boxes[file])






# path_annotations = './testing/testing/annotations/'
# path_pos = './testing/testing/arma/'
# path_neg = './testing/testing/no_arma/'
# pos_preds_file = './resultados/prob_001/preds_001_arma.pickle'
# neg_preds_file = './resultados/prob_001/preds_001_noarma.pickle'
# prob = 0.01
#
# with open(pos_preds_file, 'rb') as f:
#     pos_preds = pickle.load(f)
#
# with open(neg_preds_file, 'rb') as f:
#     neg_preds = pickle.load(f)
#
#
# total_boxes = 0
# boxes = {}
# for file in os.listdir(path_annotations):
#     if file[0] == '.':
#         continue
#     if file not in boxes:
#         boxes[file] = []
#     tags_file = open(os.path.sep.join([path_annotations, file]), 'r')
#     tags_file = BeautifulSoup(tags_file.read())
#     for i in tags_file.find_all('object'):
#         aux = (int(i.xmin.text), int(i.ymin.text),
#                 int(i.xmax.text), int(i.ymax.text))
#         total_boxes += 1
#         boxes[file].append(aux)
#
# false_neg = 0
# true_pos = 0
# false_pos = 0
# true_neg = 0
# intersect = 0
#
# for file in os.listdir(path_pos):
#     if file in pos_preds:
#         true_pos += 1
#     else:
#         false_neg += 1
#     if file in pos_preds and (file.split('.')[0]+'.xml') in boxes:
#         intersect += intersections(pos_preds[file], boxes[file.split('.')[0]+'.xml'])
#
#
# for file in os.listdir(path_neg):
#     if file in neg_preds:
#         false_pos += 1
#     else:
#         true_neg += 1
#
#
#

try:
    precision = true_pos/(true_pos+false_pos)
except:
    precision = float('nan')
recal = true_pos/(false_neg+true_pos)


print_str = f'''


    Probabilidad:{prob}

                 Pos    Neg
    Pos_detect   {true_pos}     {false_pos}
    Neg_detect   {false_neg}     {true_neg}


    Estos datos son solo de la clase positiva:
        Intersecciones entre cajas: {intersect}
        Total de cajas reales: {total_boxes}
        Total de cajas detectadas: {true_pos+false_pos}



Accuracy: {(true_pos+true_neg)/(true_pos+false_pos+true_neg+false_neg)}
F1-score: {(2*precision*recal)/(precision+recal)}
        '''

print(print_str)
with open(opt.results, 'w') as f:
    f.write(print_str)
