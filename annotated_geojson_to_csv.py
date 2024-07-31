import geopandas as gpd
import argparse
import os
import ast

def get_args():
    parser = argparse.ArgumentParser(description='Convert annotated GeoJSON to CSV')
    parser.add_argument('--input', dest="input",  type=str, help='Path to input GeoJSON file')
    parser.add_argument('--output', dest="output", type=str, help='Path to output CSV file')
    
    arg = parser.parse_args()
    arg.input = os.path.abspath(arg.input)
    return arg

def check_inputs_paths(args):
    assert os.path.isfile(args.input), "Input file does not exist"
    assert args.input.endswith(".geojson"), "Input file must be a .geojson file"
    assert args.output.endswith(".csv"), "Output file must be a .csv file"

def process_geodataframe(input_path):
    gdf = gpd.read_file(input_path)
    # remove cells not annotated
    gdf = gdf[gdf['classification'].notnull()]
    # extract annotation to new column
    gdf['cell_label'] = gdf['classification'].apply(lambda x: ast.literal_eval(x).get('name'))
    # clean up
    gdf['CellID'] = gdf['name']
    gdf.drop(columns=['id','objectType','geometry', 'name', 'classification'], inplace=True)
    return gdf

def save_csv(gdf, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gdf.to_csv(output_path, index=False)

def main():
    args = get_args()
    check_inputs_paths(args)
    save_csv(process_geodataframe(args.input), args.output)
    print(f"Success")

if __name__ == "__main__":
    main()