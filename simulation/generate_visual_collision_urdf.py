"""Generate an experimental URDF whose collision meshes match its visuals."""

import argparse
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET


def generate(source: Path, output: Path) -> int:
    tree = ET.parse(source)
    root = tree.getroot()
    converted = 0
    for link in root.findall("link"):
        collisions = link.findall("collision")
        if not collisions:
            continue
        visual = link.find("visual")
        geometry = None if visual is None else visual.find("geometry")
        mesh = None if geometry is None else geometry.find("mesh")
        if visual is None or mesh is None:
            raise ValueError(f"{link.get('name')} has collision geometry but no visual mesh")
        insertion = min(list(link).index(collision) for collision in collisions)
        for collision in collisions:
            link.remove(collision)
        collision = ET.Element("collision", {"name": "visual_mesh_collision"})
        origin = visual.find("origin")
        if origin is not None:
            collision.append(deepcopy(origin))
        collision_geometry = ET.SubElement(collision, "geometry")
        collision_geometry.append(deepcopy(mesh))
        link.insert(insertion, collision)
        converted += 1
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    count = generate(args.source, args.output)
    print(f"generated {args.output}: {count} visual-mesh collisions")


if __name__ == "__main__":
    main()
