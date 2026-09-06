#!/usr/bin/env python3
"""
Inspect the candidate TMC-2 products overlapping OHRC Shiv Shakti Point.
Target OHRC: ch2_ohr_ncp_20211023T0027462822_d_img_d18
"""

import math
import ssl
import json
import urllib.request
import urllib.parse

COOKIE = "JSESSIONID=7B9FF192F1962FFF155D7618C6C34286; JSESSIONID=011D5553573001B7C9E15E2D47FD627E; introjs-dontShowAgain=true"
HEADERS = {
    "Cookie": COOKIE,
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Origin": "https://chmapbrowse.issdc.gov.in",
    "Referer": "https://chmapbrowse.issdc.gov.in/MapBrowse/",
}
CTX = ssl._create_unverified_context()
R_MOON = 1737400.0

def latlon_to_stere_sp(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    colat = math.radians(90.0 + lat_deg)
    rho = 2.0 * R_MOON * math.tan(colat / 2.0)
    lon_rad = math.radians(lon_deg)
    return rho * math.sin(lon_rad), rho * math.cos(lon_rad)

def search_box(min_lat, max_lat, min_lon, max_lon):
    corners = [
        latlon_to_stere_sp(min_lat, min_lon),
        latlon_to_stere_sp(min_lat, max_lon),
        latlon_to_stere_sp(max_lat, min_lon),
        latlon_to_stere_sp(max_lat, max_lon),
    ]
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)

    cql = f"BBOX(the_geom, {min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f})"
    url = f"https://chmapbrowse.issdc.gov.in/server/wfs?service=wfs&version=2.0.0&request=GetFeature&outputFormat=application/json&typeName=moon:ins:sp:ch2_tmc_cal_sp&cql_filter={urllib.parse.quote(cql)}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=CTX) as r:
        data = json.loads(r.read().decode("utf-8"))
        return data.get("features", [])

def main():
    # Let's test exact and slightly expanded bounding boxes
    print("--- Searching within +/- 0.5 deg ---")
    f1 = search_box(-69.75, -68.75, 31.5, 33.5)
    for f in f1:
        p = f["properties"]
        print(f"  [{p['PRODUCT_ID']}] Type: {p.get('PRODUCT_ID', '')[8:11]} | Date: {p['OBS_ST_TIME']} | Inc: {p.get('INC_ANGLE')}")

    print("\n--- Searching within +/- 1.5 deg ---")
    f2 = search_box(-70.75, -67.75, 30.5, 34.5)
    for f in f2:
        p = f["properties"]
        print(f"  [{p['PRODUCT_ID']}] Type: {p.get('PRODUCT_ID', '')[8:11]} | Date: {p['OBS_ST_TIME']} | Inc: {p.get('INC_ANGLE')}")

    # Now let's fetch the full PDS4 label XML for ch2_tmc_ncn_20230130T1900132182_d_img_d32
    target_tmc = "ch2_tmc_ncn_20230130T1900132182_d_img_d32"
    print(f"\n--- Fetching full PDS4 Label for {target_tmc} ---")
    label_url = "https://chmapbrowse.issdc.gov.in/server/DisplayLabel"
    req_label = urllib.request.Request(
        label_url,
        data=json.dumps({"productId": target_tmc}).encode("utf-8"),
        headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-Requested-With": "XMLHttpRequest"}
    )
    with urllib.request.urlopen(req_label, context=CTX) as r:
        xml_text = r.read().decode("utf-8")
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
        ns = {"pds": "http://pds.nasa.gov/pds4/pds/v1", "isda": "https://isda.issdc.gov.in/pds4/isda/v1"}
        print("Product Title:", root.findtext(".//pds:title", namespaces=ns))
        print("Start Time:", root.findtext(".//pds:start_date_time", namespaces=ns))
        print("Pixel Resolution:", root.findtext(".//isda:pixel_resolution", namespaces=ns))
        print("Sun Elevation:", root.findtext(".//isda:sun_elevation", namespaces=ns))
        print("Sun Azimuth:", root.findtext(".//isda:sun_azimuth", namespaces=ns))
        print("Solar Incidence:", root.findtext(".//isda:solar_incidence", namespaces=ns))
        print("Area:", root.findtext(".//isda:area", namespaces=ns))
        print("File Name:", root.findtext(".//pds:file_name", namespaces=ns))
        print("File Size (bytes):", root.findtext(".//pds:file_size", namespaces=ns))
        for tag in ["upper_left", "upper_right", "lower_left", "lower_right"]:
            lat = root.findtext(f".//isda:{tag}_latitude", namespaces=ns)
            lon = root.findtext(f".//isda:{tag}_longitude", namespaces=ns)
            print(f"  {tag}: Lat {lat}, Lon {lon}")

if __name__ == "__main__":
    main()
