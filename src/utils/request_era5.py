import cdsapi
import os

dataset = "reanalysis-era5-single-levels"

for i in range(1, 2):
    request = {
        "product_type": ["reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": ["2020"],
        "month": [str(i)],
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [28, -48, -38, 106],
    }

    # prepare output directory and filename as {YEAR}_{MONTH}.grib
    out_dir = "era5_gribs"
    os.makedirs(out_dir, exist_ok=True)
    # request['year'] and request['month'] are lists (strings); pick first element
    year_str = request.get("year", ["unknown"])[0]
    month_str = request.get("month", [str(i)])[0]
    try:
        month_num = int(month_str)
    except Exception:
        month_num = i
    out_name = f"{year_str}_{month_num:02d}.grib"
    out_path = os.path.join(out_dir, out_name)

    client = cdsapi.Client()
    client.retrieve(dataset, request).download(out_path)
