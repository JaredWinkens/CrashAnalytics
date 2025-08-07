import time
import mygeotab
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

# ─────── CONFIG ───────
USERNAME     = "karimpa@sunypoly.edu"
PASSWORD     = "Qwerty2025@"
DATABASE     = "nys_dot"
SERVER       = "altitudeserver.geotab.com"  # just the host
SERVICE_NAME = "dna-altitude-traffic"      # traffic module
# ──────────────────────

def main():
    api = mygeotab.API(
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        server=SERVER
    )
    api.authenticate()
    logging.info("✅ Authenticated")

    params = {
        "queryType": "getSpeedMapMetrics",
        "dateRange": {"DateFrom":"2025-07-01","DateTo":"2025-07-30"},
        "daysOfWeek": [3,4,5],
        "isMetric": False,
        "roadTypes": ["primary"],
        "timeRange": {"TimeFrom":"00:00:00.000","TimeTo":"23:59:59.999"},
        "vehicleClasses": [{"VehicleType":"Passenger","WeightClass":"*"}],
        "vehicleClassSchemeId": 2,
        "zones": [{
            "ZoneId":     "36065",     # Oneida County, NY
            "ISO_3166_2": "US-NY",
            "ZoneType":   "County"     # must be capitalized
        }]
    }

    create = api.call(
        method="GetAltitudeData",
        serviceName=SERVICE_NAME,
        functionName="createQueryJob",
        functionParameters=params
    )
    results = create["apiResult"]["results"]
    if not results:
        raise RuntimeError("Failed to create speedMapMetrics job (empty results).")
    job_id = results[0]["id"]
    logging.info("🆔 Job created: %s", job_id)

    params["jobId"] = job_id

    while True:
        status = api.call(
            method="GetAltitudeData",
            serviceName=SERVICE_NAME,
            functionName="getJobStatus",
            functionParameters=params
        )["apiResult"]["results"]
        state = status[0]["status"]["state"] if status else None
        logging.info("⏳ Status: %s", state)
        if state == "DONE":
            logging.info("✅ Job complete")
            break
        time.sleep(5)

    params["pageToken"] = None
    params["resultsLimit"] = 50000

    all_rows = []
    page_token = None
    while True:
        if page_token:
            print("pageToken: ", page_token)
            params["pageToken"] = page_token

        fetch = api.call(
            method="GetAltitudeData",
            serviceName=SERVICE_NAME,
            functionName="getQueryResults",
            functionParameters=params
        )["apiResult"]["results"]

        if not fetch:
            print("No pages left")
            break

        block = fetch[0]
        rows = block.get("rows", [])
        logging.info("📑 Retrieved %d rows", len(rows))
        all_rows.extend(rows)

        page_token = block.get("pageToken")
        if not page_token:
            break

    if not all_rows:
        print("⚠️ No speedMapMetrics data returned.")
    else:
        df = pd.DataFrame(all_rows)
        print(df.head())
        print(f"\n🏁 Total records: {len(df)}")

if __name__ == "__main__":
    main()
