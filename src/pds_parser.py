from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import numpy as np

@dataclass(frozen=True)
class CornerCoordinates:
    """Contains corner coordinates of the observation."""
    upper_left_lat: float
    upper_left_lon: float
    upper_right_lat: float
    upper_right_lon: float
    lower_left_lat: float
    lower_left_lon: float
    lower_right_lat: float
    lower_right_lon: float
    
    @property
    def lat_range(self) -> Tuple[float, float]:
        """(min_lat, max_lat)"""
        lats = [self.upper_left_lat, self.upper_right_lat, self.lower_left_lat, self.lower_right_lat]
        return min(lats), max(lats)
    
    @property
    def lon_range(self) -> Tuple[float, float]:
        """(min_lon, max_lon)"""
        lons = [self.upper_left_lon, self.upper_right_lon, self.lower_left_lon, self.lower_right_lon]
        return min(lons), max(lons)

@dataclass(frozen=True)
class BandInfo:
    """Information for a single spectral band."""
    band_number: int  # 1-indexed as in XML
    center_wavelength_nm: float
    band_width_nm: float

@dataclass(frozen=True)
class ArrayDescriptor:
    """Describes the binary array layout."""
    ndim: int                    # 2 or 3
    shape: Tuple[int, ...]       # (lines, samples) or (bands, lines, samples)
    dtype: np.dtype              # numpy dtype
    offset_bytes: int            # byte offset in file
    axis_names: Tuple[str, ...]  # e.g., ('Line', 'Sample') or ('BAND', 'LINE', 'SAMPLE')

@dataclass(frozen=True)
class PDS4Label:
    """Complete parsed PDS4 label."""
    instrument_name: str
    start_time: str
    pixel_resolution_m: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    area: str
    corners: CornerCoordinates
    array: ArrayDescriptor
    bands: List[BandInfo]        # empty for 2D images
    file_name: str
    file_size_bytes: int


class PDS4Parser:
    """Dynamically parses Chandrayaan-2 PDS4 XML labels."""
    
    PDS_TYPE_MAP: Dict[str, np.dtype] = {
        'UnsignedByte': np.dtype('uint8'),
        'UnsignedLSB2': np.dtype('<u2'),  # little-endian uint16
        'SignedLSB2': np.dtype('<i2'),
        'IEEE754MSBSingle': np.dtype('>f4'),
    }
    
    NAMESPACES = {
        'pds': 'http://pds.nasa.gov/pds4/pds/v1',
        'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'
    }
    
    def __init__(self, xml_path: str | Path):
        """
        Initialize the parser with the path to the PDS4 XML file.
        
        Args:
            xml_path: Path to the XML file.
        """
        self.xml_path = Path(xml_path)
        self._tree = ET.parse(str(self.xml_path))
        self._root = self._tree.getroot()
    
    def parse(self) -> PDS4Label:
        """
        Parse the complete PDS4 label into a PDS4Label dataclass.
        
        Returns:
            A populated PDS4Label instance.
        """
        obs_area = self._root.find('.//pds:Observation_Area', self.NAMESPACES)
        if obs_area is None:
            raise ValueError("Observation_Area not found in XML")
            
        time_coords = obs_area.find('.//pds:Time_Coordinates', self.NAMESPACES)
        start_time = self._find_text(time_coords, 'start_date_time') if time_coords is not None else ""
        
        instrument_name = "Unknown"
        obs_system = self._root.find('.//pds:Observing_System', self.NAMESPACES)
        if obs_system is not None:
            for comp in obs_system.findall('.//pds:Observing_System_Component', self.NAMESPACES):
                type_val = self._find_text(comp, 'type')
                if type_val == 'Instrument':
                    name_val = self._find_text(comp, 'name')
                    if name_val:
                        instrument_name = name_val
                        break
        
        prod_params = self._parse_product_params()
        corners = self._parse_corners()
        array = self._parse_array()
        bands = self._parse_bands()
        
        file_area = self._root.find('.//pds:File_Area_Observational', self.NAMESPACES)
        if file_area is None:
            raise ValueError("File_Area_Observational not found in XML")
            
        file_el = file_area.find('pds:File', self.NAMESPACES)
        file_name = self._find_text(file_el, 'file_name')
        file_size_bytes = int(self._find_text(file_el, 'file_size') or 0)
        
        return PDS4Label(
            instrument_name=instrument_name,
            start_time=start_time or "",
            pixel_resolution_m=prod_params['pixel_resolution'],
            sun_azimuth_deg=prod_params['sun_azimuth'],
            sun_elevation_deg=prod_params['sun_elevation'],
            area=prod_params['area'],
            corners=corners,
            array=array,
            bands=bands,
            file_name=file_name or "",
            file_size_bytes=file_size_bytes
        )
    
    def _parse_array(self) -> ArrayDescriptor:
        """
        Parses the binary array layout (either 2D Image or 3D Spectrum).
        
        Returns:
            ArrayDescriptor containing the dimensions and data type.
        """
        file_area = self._root.find('.//pds:File_Area_Observational', self.NAMESPACES)
        if file_area is None:
            raise ValueError("File_Area_Observational not found")
            
        array = file_area.find('pds:Array_2D_Image', self.NAMESPACES)
        if array is None:
            array = file_area.find('pds:Array_3D_Spectrum', self.NAMESPACES)
            
        if array is None:
            raise ValueError("No Array_2D_Image or Array_3D_Spectrum found")
            
        axes_count = int(self._find_text(array, 'axes') or 0)
        offset_bytes = int(self._find_text(array, 'offset') or 0)
        
        elem_array = array.find('pds:Element_Array', self.NAMESPACES)
        dt_str = self._find_text(elem_array, 'data_type') if elem_array is not None else None
        
        if dt_str not in self.PDS_TYPE_MAP:
            raise ValueError(f"Unknown data type: {dt_str}")
        dtype = self.PDS_TYPE_MAP[dt_str]
        
        axis_arrays = array.findall('pds:Axis_Array', self.NAMESPACES)
        
        axis_arrays_sorted = sorted(
            axis_arrays,
            key=lambda x: int(self._find_text(x, 'sequence_number') or 0)
        )
        
        shape = []
        axis_names = []
        for ax in axis_arrays_sorted:
            shape.append(int(self._find_text(ax, 'elements') or 0))
            axis_names.append(self._find_text(ax, 'axis_name') or "")
            
        return ArrayDescriptor(
            ndim=axes_count,
            shape=tuple(shape),
            dtype=dtype,
            offset_bytes=offset_bytes,
            axis_names=tuple(axis_names)
        )
    
    def _parse_bands(self) -> List[BandInfo]:
        """
        Parses the spectral band information for 3D cubes.
        
        Returns:
            A list of BandInfo, empty if no bands found.
        """
        bands = []
        file_area = self._root.find('.//pds:File_Area_Observational', self.NAMESPACES)
        if file_area is not None:
            array_3d = file_area.find('pds:Array_3D_Spectrum', self.NAMESPACES)
            if array_3d is not None:
                axis_arrays = array_3d.findall('pds:Axis_Array', self.NAMESPACES)
                for ax in axis_arrays:
                    band_bin_set = ax.find('pds:Band_Bin_Set', self.NAMESPACES)
                    if band_bin_set is not None:
                        for band_bin in band_bin_set.findall('pds:Band_Bin', self.NAMESPACES):
                            band_num = int(self._find_text(band_bin, 'band_number') or 0)
                            center_wvl = float(self._find_text(band_bin, 'center_wavelength') or 0.0)
                            band_width = float(self._find_text(band_bin, 'band_width') or 0.0)
                            
                            bands.append(BandInfo(
                                band_number=band_num,
                                center_wavelength_nm=center_wvl,
                                band_width_nm=band_width
                            ))
        return bands
    
    def _parse_corners(self) -> CornerCoordinates:
        """
        Parses corner coordinates, preferring refined over system-level.
        
        Returns:
            CornerCoordinates populated with latitudes and longitudes.
        """
        geom_params = self._root.find('.//isda:Geometry_Parameters', self.NAMESPACES)
        if geom_params is None:
            raise ValueError("Geometry_Parameters not found")
            
        coords = geom_params.find('.//isda:Refined_Corner_Coordinates', self.NAMESPACES)
        if coords is None:
            coords = geom_params.find('.//isda:System_Level_Coordinates', self.NAMESPACES)
            
        if coords is None:
            raise ValueError("No Corner Coordinates found (Refined or System_Level)")
            
        return CornerCoordinates(
            upper_left_lat=float(self._find_isda_text(coords, 'upper_left_latitude') or 0.0),
            upper_left_lon=float(self._find_isda_text(coords, 'upper_left_longitude') or 0.0),
            upper_right_lat=float(self._find_isda_text(coords, 'upper_right_latitude') or 0.0),
            upper_right_lon=float(self._find_isda_text(coords, 'upper_right_longitude') or 0.0),
            lower_left_lat=float(self._find_isda_text(coords, 'lower_left_latitude') or 0.0),
            lower_left_lon=float(self._find_isda_text(coords, 'lower_left_longitude') or 0.0),
            lower_right_lat=float(self._find_isda_text(coords, 'lower_right_latitude') or 0.0),
            lower_right_lon=float(self._find_isda_text(coords, 'lower_right_longitude') or 0.0),
        )
    
    def _parse_product_params(self) -> dict:
        """
        Parses general product parameters like resolution and sun position.
        
        Returns:
            A dictionary with pixel_resolution, sun_azimuth, sun_elevation, and area.
        """
        obs_area = self._root.find('.//pds:Observation_Area', self.NAMESPACES)
        if obs_area is None:
            raise ValueError("Observation_Area not found")
        mission_area = obs_area.find('.//pds:Mission_Area', self.NAMESPACES)
        if mission_area is None:
            raise ValueError("Mission_Area not found")
        prod_params = mission_area.find('.//isda:Product_Parameters', self.NAMESPACES)
        if prod_params is None:
            raise ValueError("Product_Parameters not found")
            
        return {
            'pixel_resolution': float(self._find_isda_text(prod_params, 'pixel_resolution') or 0.0),
            'sun_azimuth': float(self._find_isda_text(prod_params, 'sun_azimuth') or 0.0),
            'sun_elevation': float(self._find_isda_text(prod_params, 'sun_elevation') or 0.0),
            'area': self._find_isda_text(prod_params, 'area') or ""
        }
    
    def _find_text(self, element: ET.Element, tag: str, ns: str = 'pds') -> str | None:
        """
        Helper method to find text content of an element.
        """
        if element is None:
            return None
        el = element.find(f'{ns}:{tag}', self.NAMESPACES)
        if el is not None:
            return el.text
        return None
    
    def _find_isda_text(self, element: ET.Element, tag: str) -> str | None:
        """
        Helper method to find text content of an ISDA namespace element.
        """
        return self._find_text(element, tag, ns='isda')
