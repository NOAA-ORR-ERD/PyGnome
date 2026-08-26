from dataclasses import dataclass, field
import json
import logging
import tempfile
import zipfile
from pathlib import Path
import geopandas as gpd
import os


# ------------------------------------------------------------------
# Some internal data structures for erma_data_package tests
# ------------------------------------------------------------------
@dataclass
class ShapefileData:
    filename: str
    gdf: gpd.GeoDataFrame
    srid: int = None
    geometry_types: list = field(default_factory=list)


@dataclass
class LayerJSONData:
    filename: str
    shapefile: ShapefileData
    raw_data: dict = field(default_factory=dict)
    mapfile_layer: dict = field(default_factory=dict)

    @property
    def srid(self):
        """Extract srid from mapfile_layer -> shapefile object."""
        sf = self.mapfile_layer.get("shapefile", {})
        if isinstance(sf, dict) and "srid" in sf:
            return sf.get("srid")
        return None
    # Should add more properties for convenience


@dataclass
class PackageData:
    zip_path: Path
    layer_jsons: dict[str, LayerJSONData] = field(default_factory=dict)
    shapefiles: dict[str, ShapefileData] = field(default_factory=dict)


# ------------------------------------------------------------------
# Parsing Logic
# ------------------------------------------------------------------
def _parse_layer_json(zf, json_filename):
    # Load the base json file
    with zf.open(json_filename) as f:
        data = json.load(f)
    # Grab the mapfile_layer if its there
    mapfile_layer = data.get("mapfile_layer", [])
    if isinstance(mapfile_layer, dict):
        mapfile_layer = mapfile_layer
    # Make our LayerJSONData wrapper.  Start with no shapefile and parse
    # and add it later.
    return LayerJSONData(
        filename=json_filename, raw_data=data, mapfile_layer=mapfile_layer, shapefile=None
    )


def _parse_shapefile(shp_path):
    gdf = gpd.read_file(shp_path)
    crs = gdf.crs
    srid = crs.to_epsg() if crs else None
    geom_types = list(set(gdf.geometry.geom_type.dropna().unique()))
    return ShapefileData(
        filename=shp_path.name,
        gdf=gdf,
        srid=srid,
        geometry_types=geom_types,
    )


def parse_zip_to_package_data(zip_path: Path | str, extract_dir: Path | str) -> PackageData:
    """Core parser that takes any package zip file path and parses it into PackageData."""
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)

    if not zip_path.exists():
        raise FileNotFoundError(f"Package zip file does not exist: {zip_path}")
    # Little container to hold the package data parsed out
    pkg = PackageData(zip_path=zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        # Parse JSON files in layers/
        layer_json_files = [
            name for name in namelist if name.startswith("layers/") and name.endswith(".json")
        ]
        # Create the LayerJSONData structures
        for jf in layer_json_files:
            pkg.layer_jsons[jf] = _parse_layer_json(zf, jf)

        # Extract nested shapefiles in source_files/
        source_zip_files = [
            name for name in namelist if name.startswith("source_files/") and name.endswith(".zip")
        ]
        # Look through the shapefile zips and create ShapefileData structures
        for inner_zip in source_zip_files:
            extracted_inner = zf.extract(inner_zip, path=extract_dir)
            inner_subfolder = extract_dir / Path(inner_zip).stem

            with zipfile.ZipFile(extracted_inner, "r") as inner_zf:
                inner_zf.extractall(inner_subfolder)

            for shp_file in inner_subfolder.rglob("*.shp"):
                pkg.shapefiles[shp_file.name] = _parse_shapefile(shp_file)
        # Now loop through the layers and link up thier shapefiles if they exist
        # Note multiple layers can point to the same shapefile, so be aware
        for json_name, json_value in pkg.layer_jsons.items():
            # Try to match the shapefile associated with the layer we are looking at
            if json_value.mapfile_layer and json_value.mapfile_layer['shapefile']:
                for shape_name, shape_val in pkg.shapefiles.items():
                    if Path(shape_val.filename).stem == Path(json_value.mapfile_layer['shapefile']['file']).stem:
                        json_value.shapefile = shape_val

    return pkg


def parse_package(package_path):
    with tempfile.TemporaryDirectory() as extract_dir:
        return parse_zip_to_package_data(package_path, extract_dir)
