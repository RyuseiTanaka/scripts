import json
import random
from itertools import permutations

bdd_file = open("./datasets/bdd100k/labels_coco/bdd100k_labels_images_train_coco.json","r") 
bdd100k_list = json.load(bdd_file)  

num_all = len(bdd100k_list['images'])

outname = "bdd100k_supervision.txt"
fout = open(outname, "w")
SupPercent = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]

all_idx_list ={}
for per in SupPercent:
    num_label = int(per/100.0 * num_all)
    id_list ={} 
    for seed in range(10):
        lr = random.sample(range(num_all), num_all)
        id_list[seed] = lr[:num_label] 
    all_idx_list[per] = id_list
print(all_idx_list)
json.dump(all_idx_list,fout)