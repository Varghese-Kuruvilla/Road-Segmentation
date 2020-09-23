#Script to convert CVAT annotation into json format
import xml.etree.ElementTree as ET 
import re
import json
import os


def xml_to_json(xml_file_path,img_file_path):
    
        """ 
        Parses XML annotation file(in cvat for images-1.1 format) and generates output json file 
        :type xml_file_path: Absolute path of the XML annotation file
        :param xml_file_path:
    
        :type img_file_path: Absolute path of the image file
        :param img_file_path:
    
        :raises:
    
        :rtype:
        """'''
        #NOTE: The function is written to work only with polygon type annotations
    '''
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    flag = True #Flag to create dictionary
    for child in root:
        if(child.tag == 'image'):
            imgheight = child.attrib['height']
            imgwidth = child.attrib['width']
            if(flag == True):
                #Create dictionary
                json_data = {
                    "imgHeight":imgheight,
                    "imgWidth":imgwidth,
                    "objects":[]
                }
                flag = False
            img_id = child.attrib['id']
            for grandchild in child: #Grandchild of ROOT
                if(grandchild.tag == 'polygon'):
                    id_name = grandchild.attrib['label']
                    poly_points = grandchild.attrib['points']
                    poly_points = re.split('\;|\,',poly_points)
                    poly_ls = []
                    for i in range(0,len(poly_points),2):
                        poly_ls.append([float(poly_points[i]),float(poly_points[i+1])])

                    poly_json = {                        
                            "date":"NaN",
                            "deleted":0,
                            "draw": "NaN",
                            "id": int(img_id),
                            "label": str(id_name),
                            "polygon": poly_ls,
                            "user":"iit",
                            "verified":0
                        }
                    json_data["objects"].append(poly_json)
    # print("json_data:",json_data)
    #Write JSON data to a file

    # json_data = json.dumps(json_data)
    #NOTE: Check the output filename of the json file
    json_file_name = os.path.basename(img_file_path)
    json_file_name = json_file_name.split('_gtFine_')[0]
    with open(json_file_name + '_gtFine_polygons.json','w') as fp:
        json.dump(json_data,fp,indent=1,sort_keys=True)

    
            










if __name__ == '__main__':
    xml_file_path = ''
    img_file_path = ''
    xml_to_json(xml_file_path,img_file_path)