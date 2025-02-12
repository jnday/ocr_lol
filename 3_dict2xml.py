import os
import json
from xml.etree.ElementTree import Element, tostring
from xml.dom.minidom import parseString

def dict_to_xml(dict_obj, root_name='root'):

    def _build_xml(parent, data):
        if isinstance(data, dict):
            for key, value in data.items():
                child = Element(str(key))
                parent.append(child)
                _build_xml(child, value)
        elif isinstance(data, (list, tuple)):
            for item in data:
                child = Element('item')
                parent.append(child)
                _build_xml(child, item)
        else:
            parent.text = str(data)
    
    root = Element(root_name)
    _build_xml(root, dict_obj)
    
    # Convert to string with pretty printing
    rough_string = tostring(root, 'utf-8')
    parsed = parseString(rough_string)
    return parsed.toprettyxml(indent="  ")



def main():
    input_dir = 'step_2/'
    output_dir = 'step_3/'
    os.makedirs(output_dir, exist_ok=True)

    with os.scandir(input_dir) as iter:
        for entry in iter:
            if entry.is_file() and entry.name.endswith('.json'): 
                filename = entry.name.removesuffix('.json')
                xml_path = output_dir + filename + '.xml'
                
                # load .json data file
                with open(entry.path) as file:
                    data = json.load(file)

                xml_output = dict_to_xml(data, 'data')
                
                with open(xml_path, 'w') as f:
                    f.write(xml_output)


if __name__ == '__main__':
    main()

