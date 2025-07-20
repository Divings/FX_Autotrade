import xml.etree.ElementTree as ET

def load_config_from_xml(xml_path: str) -> dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    config = {}
    for table in root.findall(".//table[@name='bot_config']"):
        key = table.find("./column[@name='key']").text
        value = table.find("./column[@name='value']").text

        # 自動で型変換
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass

        config[key] = value

    return config