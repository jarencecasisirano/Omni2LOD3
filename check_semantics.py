import json

with open('outputs/00_json_wall_merged/NIMBB-demi_merged.json', encoding='utf-8') as f:
    cm = json.load(f)

for obj_id, obj in cm['CityObjects'].items():
    print("Object:", obj_id, "type:", obj.get("type"))
    for gi, geom in enumerate(obj.get('geometry', [])):
        sem = geom.get('semantics', {})
        surfaces = sem.get('surfaces', [])
        vals_flat = []
        def flatten(x):
            if isinstance(x, list):
                for i in x:
                    flatten(i)
            elif x is not None:
                vals_flat.append(x)
        flatten(sem.get('values', []))
        type_counts = {}
        for idx in vals_flat:
            if idx is not None and idx < len(surfaces):
                t = surfaces[idx].get('type', '?')
            else:
                t = "INVALID(" + str(idx) + ")"
            type_counts[t] = type_counts.get(t, 0) + 1
        print("  Geom", gi, "(" + geom["type"] + "):", len(surfaces), "surface defs,", len(vals_flat), "polygons")
        for t, c in type_counts.items():
            print("    " + t + ":", c, "polygons")
        print("  Surface defs:", [s.get('type') for s in surfaces])
