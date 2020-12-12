import csv
dict = {}
with open('map1.txt','r') as file:
    lines = file.readlines()
    for line in lines:
        key = line.split("\t")[0]
        value = line.strip().split("\t")[1]
        dict[key] = value

dict1  ={}
with open("mapping.txt",'r') as file:
    lines = file.readlines()
    for line in lines:
        key = line.split("\t")[0]
        value = line.strip().split("\t")[1]
        dict1[key] = value
dict2 = {}
with open('/home/jingshuai/桌面/input/landmark-recognition-2020/train.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    print('a')
    for row in spamreader:
        filename = row[0].split(',')[0]
        label = row[0].split(',')[1]
        # print(filename+"\t"+label)
        dict2[filename] = label
        # dict2[1]
dict3 = {}
with open("map2.txt",'w') as file:
    for key in dict.keys():
        # print(key)
        file.write(dict.get(key)+"\t"+dict1[key]+"\n")
        dict3[dict.get(key)] = dict1[key]
        # print(dict.get(key)+"\t"+dict1[key])
final = {}
with open("final.txt",'w') as file:
    for key in dict2.keys():
        if key in dict3:
            final[dict2[key]] = dict3[key]
            # file.write(dict2[key]+"\t"+dict3[key]+"\n")

    for key in final.keys():
        file.write(key+"\t"+final[key]+"\n")
dict4 = {}
with open('/home/jingshuai/桌面/train_label_to_category.csv') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        landmark_id = row[0].split(',')[0]
        if(landmark_id.find("landmark")==-1):
            label = row[0].split(',')[1].split(':')[2]
            dict4[landmark_id] = label

for key in final.keys():
    actual_category = dict4[key]
    print(final[key]+"\t"+actual_category)

with open('finalmap.txt','w') as file:
    for key in final.keys():
        actual_category = dict4[key]
        file.write(final[key] + "\t" + actual_category+"\n")