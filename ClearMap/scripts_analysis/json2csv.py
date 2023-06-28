import csv
import json


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):

    # create a dictionary
    data = {}

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:

            # Assuming a column named 'No' to
            # be the primary key
            key = rows['id']
            data[key] = rows

    # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))



def get_children(rows, ordersR, key,IDs2rm):
    children=[]
    children_key=[]
    print(key)
    # print(rows)
    ids_children=[]
    for child_rows in ordersR:
        # if key in IDs2rm:
            # print('already there')
        if child_rows['id'] not in IDs2rm:
            if child_rows['parent_structure_id']==key:
                print('key',child_rows['id'])
                children_key.append(child_rows['id'])
                IDs2rm.append(child_rows['id'])
                children.append(get_children(child_rows,ordersR,child_rows['id'],IDs2rm)[0])
        rows['children']=children
    # print('len(IDs2rm)',len(IDs2rm))
    # if len(children_key)>0:
    print(key, len(children), 'children_keys',children_key)
    if key in children_key:
        print('key in children key')
        raise RuntimeError('key in children key')
    return rows,IDs2rm

def test_json(csvFilePath, jsonFilePath):

    # create a dictionary
    data = []
    IDs2rm=[]
    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8') as csvf:
        # csvReader = csv.DictReader(csvf)
        ordersR = list(csv.DictReader(csvf))

        # Convert each row into a dictionary
        # and add it to data
        for i in range(len(ordersR)):
            ordersR[i]['id']=int(ordersR[i]['id'])
            ordersR[i]['level']=int(ordersR[i]['level'])
            ordersR[i]['parent_structure_id']=int(ordersR[i]['parent_structure_id'])
            ordersR[i]['graph_order']=int(ordersR[i]['graph_order'])
            ordersR[i]['atlas_id']=int(ordersR[i]['atlas_id'])

        for rows in ordersR:

            # Assuming a column named 'No' to
            # be the primary key
            key = rows['id']

            # if key in IDs2rm:
                # print('already there')

            if key not in IDs2rm:
                print(key)
                children=[]
                IDs2rm.append(key)
                ch,IDs2rm=get_children(rows, ordersR, rows['id'],IDs2rm)
                # children.append(ch)
                ids_children=[]
                # for child_rows in ordersR:
                #     if child_rows['parent_structure_id']==key:
                #         #     children.append(child_rows)
                #         #     ids_children.append(child_rows['id'])
                #         #     IDs2rm.append(child_rows['id'])
                #         ch=get_children(child_rows, ordersR, rows['id'],IDs2rm)
                #         children.append(ch)

                rows=ch
                # rows['children']=children
                # print(rows)
                # data[key] = rows
                data.append(rows)


    print(len(data), len(ordersR))
    # Open a json writer, and use the json.dumps()
    # function to dump data
    # with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    #     jsonf.write(json.dumps(data, indent=1))
    return data

# Driver Code

# Decide the two file paths according to your
# computer system
workdir='/home/sophie.skriabine/Documents/'

csvFilePath = workdir+'region_ids_ADMBA.csv'
jsonFilePath = workdir+'region_ids_test_ADMBA.json'

# Call the make_json function
data=test_json(csvFilePath, jsonFilePath)

# def date_handler(obj):
#     if hasattr(obj, 'isoformat'):
#         return obj.isoformat()
#     else:
#         json.JSONEncoder.default(self,obj)
with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
    jsonf.write(json.dumps(data, indent=1))



make_json(csvFilePath, jsonFilePath)