-- Aggregates weather data by year and month
-- CTE for weather data aggregated at year-month level
WITH WeatherAgg AS (
    SELECT 
        YEAR([DATE]) AS Year,
        MONTH([DATE]) AS Month,
        MAX(AWND) AS AWND,
        MAX(PRCP) AS PRCP,
        MAX(SNOW) AS SNOW,
        MAX(SNWD) AS SNWD,
        MAX(TMAX) AS TMAX,
        MAX(TMIN) AS TMIN
    FROM Cleaned_Tbl_Weather
    GROUP BY YEAR([DATE]), MONTH([DATE])
),

-- CTE for GWBBS data safely cast to date and aggregated by year-month
GWBBS_Agg AS (
    SELECT 
        YEAR(TRY_CAST(Date AS DATE)) AS Year,
        MONTH(TRY_CAST(Date AS DATE)) AS Month,
        SUM(Buses_Count) AS Total_Buses,
        SUM(Passengers_Count) AS Total_Passengers
    FROM Join_GWBBS_NJT_NYC_Trips_Buses_Passengers
    WHERE TRY_CAST(Date AS DATE) IS NOT NULL
    GROUP BY YEAR(TRY_CAST(Date AS DATE)), MONTH(TRY_CAST(Date AS DATE))
)

-- Final query
SELECT 
    YEAR(bd.Date) AS Year,
    MONTH(bd.Date) AS Month,
    SUM(bd.[GWBBS Hudson Transit]) AS [Hudson Transit],
    SUM(bd.[GWBBS Rockland]) AS [Rockland],
    SUM(bd.[GWBBS NJ Transit]) AS [NJ Transit],
    SUM(bd.[GWBBS Spanish]) AS [Spanish],
    SUM(bd.[GWBBS Greyhound LD]) AS [Greyhound LD],
    SUM(bd.[GWBBS OurBus LD]) AS [OurBus LD],
    SUM(bd.[GWBBS Saddle River]) AS [Saddle River],
    SUM(bd.[GWBBS Saddle River LD]) AS [Saddle River LD],
    SUM(bd.[GWBBS Vanessa]) AS [Vanessa],
    SUM(bd.[GWBBS_Total]) AS [Total GWBBS],
    gw.Total_Buses,
    gw.Total_Passengers,
    w.AWND, w.PRCP, w.SNOW, w.SNWD, w.TMAX, w.TMIN
FROM 
    Cleaned_Bus_Carrier_Departure bd
LEFT JOIN GWBBS_Agg gw
    ON YEAR(bd.Date) = gw.Year AND MONTH(bd.Date) = gw.Month
LEFT JOIN WeatherAgg w
    ON YEAR(bd.Date) = w.Year AND MONTH(bd.Date) = w.Month
GROUP BY 
    YEAR(bd.Date), MONTH(bd.Date),
    gw.Total_Buses, gw.Total_Passengers,
    w.AWND, w.PRCP, w.SNOW, w.SNWD, w.TMAX, w.TMIN
ORDER BY 
    Year, Month;

-- Aggregates GWBBS bus and passenger data by month
DATASET JOINS - 3 DATASET WITH THE TRAFFIC FOR GETTING THE FACILITY MORE ROWS









WITH td_monthly AS (
    SELECT DISTINCT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        FAC_G2
    FROM Cleaned_Traffic_Data
)

SELECT 
    td.Year,
    td.Month,
    td.FAC_G2,
    TRY_CAST(gwbbs.Buses_Count AS BIGINT) AS gwbbs_Buses_Count,
    TRY_CAST(gwbbs.Passengers_Count AS BIGINT) AS gwbbs_Passengers_Count,
    TRY_CAST(mbt.Buses_Count AS BIGINT) AS mbt_Buses_Count,
    TRY_CAST(mbt.Passengers_Count AS BIGINT) AS mbt_Passengers_Count,
    TRY_CAST(mbtpd.total AS BIGINT) AS mbtpd_Total,
    TRY_CAST(mbtpd.bus_total AS BIGINT) AS mbtpd_Total_bus
FROM
    td_monthly td
RIGHT JOIN 
    Join_GWBBS_NJT_NYC_Trips_Buses_Passengers gwbbs
    ON td.Year = LEFT(CONVERT(VARCHAR, gwbbs.Date, 23), 4)
    AND td.Month = SUBSTRING(CONVERT(VARCHAR, gwbbs.Date, 23), 6, 2)
RIGHT JOIN 
    Join_MBT_NJT_NYC_Trips_Buses_Passengers mbt
    ON td.Year = LEFT(CONVERT(VARCHAR, mbt.Date, 23), 4)
    AND td.Month = SUBSTRING(CONVERT(VARCHAR, mbt.Date, 23), 6, 2)
RIGHT JOIN 
    Cleaned_MBT_Passenger_Departures mbtpd
    ON td.Year = LEFT(CONVERT(VARCHAR, mbtpd.Start_Date, 23), 4)
    AND td.Month = SUBSTRING(CONVERT(VARCHAR, mbtpd.Start_Date, 23), 6, 2)
ORDER BY 
    td.Year, 
    td.Month, 
    td.FAC_G2;

-- Aggregates GWBBS bus and passenger data by month
DATASET JOINS - 3 DATASET WITH THE TRAFFIC FOR GETTING THE FACILITY MORE ROWS weatehr





WITH td_monthly AS (
    SELECT DISTINCT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        FAC_G2,
        time
    FROM Cleaned_Traffic_Data
),

gwbbs_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(Passengers_Count AS BIGINT)) AS gwbbs_Passengers_Count
    FROM Join_GWBBS_NJT_NYC_Trips_Buses_Passengers
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2)
),

mbt_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(Passengers_Count AS BIGINT)) AS mbt_Passengers_Count
    FROM Join_MBT_NJT_NYC_Trips_Buses_Passengers
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2)
),

mbtpd_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Start_Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Start_Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(total AS BIGINT)) AS mbtpd_Passengers_Count
    FROM Cleaned_MBT_Passenger_Departures
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Start_Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Start_Date, 23), 6, 2)
),

weather_raw_with_rownum AS (
    SELECT
        LEFT(CONVERT(VARCHAR, date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, date, 23), 6, 2) AS Month,
        TRY_CAST(AWND AS FLOAT) AS AWND,
        TRY_CAST(PRCP AS FLOAT) AS PRCP,
        TRY_CAST(SNOW AS FLOAT) AS SNOW,
        TRY_CAST(SNWD AS FLOAT) AS SNWD,
        TRY_CAST(TMAX AS FLOAT) AS TMAX,
        TRY_CAST(TMIN AS FLOAT) AS TMIN,
        ROW_NUMBER() OVER (
            PARTITION BY 
                LEFT(CONVERT(VARCHAR, date, 23), 4),
                SUBSTRING(CONVERT(VARCHAR, date, 23), 6, 2)
            ORDER BY date
        ) AS rn
    FROM Cleaned_Tbl_Weather
),

weather_monthly AS (
    SELECT 
        Year,
        Month,
        AWND,
        PRCP,
        SNOW,
        SNWD,
        TMAX,
        TMIN
    FROM weather_raw_with_rownum
    WHERE rn = 1
)

-- Final result
SELECT DISTINCT
    td.Year,
    td.Month,
    td.FAC_G2,
    td.time,
    gwbbs.gwbbs_Passengers_Count,
    mbt.mbt_Passengers_Count,
    mbtpd.mbtpd_Passengers_Count,
    weather.AWND,
    weather.PRCP,
    weather.SNOW,
    weather.SNWD,
    weather.TMAX,
    weather.TMIN
FROM
    td_monthly td
LEFT JOIN gwbbs_monthly gwbbs
    ON td.Year = gwbbs.Year AND td.Month = gwbbs.Month
LEFT JOIN mbt_monthly mbt
    ON td.Year = mbt.Year AND td.Month = mbt.Month
LEFT JOIN mbtpd_monthly mbtpd
    ON td.Year = mbtpd.Year AND td.Month = mbtpd.Month
LEFT JOIN weather_monthly weather
    ON td.Year = weather.Year AND td.Month = weather.Month
ORDER BY 
    td.Year, 
    td.Month, 
    td.FAC_G2,
    td.time;

-- Joins MBT passenger and bus departure data and aggregates by month
SELECT
    YEAR(p.Start_Date) AS Year,
    MONTH(p.Start_Date) AS Month,
    SUM(p.Academy) AS passenger_academy,
    SUM(b.Academy) AS bus_academy,
    SUM(p.Greyhound) AS passenger_greyhound,
    SUM(b.Greyhound) AS bus_greyhound,
    SUM(p.[Coach USA]) AS passenger_coach_usa,
    SUM(b.[Coach USA]) AS bus_coach_usa,
    SUM(p.[TransBridge]) AS passenger_transbridge,
    SUM(b.[TransBridge]) AS bus_transbridge,
    SUM(p.[Peter Pan/Bonanza]) AS passenger_peterpan,
    SUM(b.[Peter Pan/Bonanza]) AS bus_peterpan,
    SUM(p.[C & J Bus Lines]) AS passenger_cj,
    SUM(b.[C & J Bus Lines]) AS bus_cj
FROM cleaned_mbt_passenger_departures p
JOIN cleaned_mbt_bus_departures b
  ON p.Start_Date = b.Start_Date
 AND p.End_Date = b.End_Date
GROUP BY
    YEAR(p.Start_Date),
    MONTH(p.Start_Date)
ORDER BY
    Year, Month;

-- Joins MBT passenger and bus departure data and aggregates by month
SELECT
    YEAR(p.Start_Date) AS Year,
    MONTH(p.Start_Date) AS Month,
    p.Academy AS passenger_academy,
    b.Academy AS bus_academy,
    p.Greyhound AS passenger_greyhound,
    b.Greyhound AS bus_greyhound,
    p.[Coach USA] AS passenger_coach_usa,
    b.[Coach USA] AS bus_coach_usa,
    p.[TransBridge] AS passenger_transbridge,
    b.[TransBridge] AS bus_transbridge,
    p.[Peter Pan/Bonanza] AS passenger_peterpan,
    b.[Peter Pan/Bonanza] AS bus_peterpan,
    p.[C & J Bus Lines] AS passenger_cj,
    b.[C & J Bus Lines] AS bus_cj
FROM cleaned_mbt_passenger_departures p
JOIN cleaned_mbt_bus_departures b
  ON p.Start_Date = b.Start_Date;

-- Aggregates GWBBS bus and passenger data by month
SELECT 
    YEAR(bd.Date) AS Year,
    MONTH(bd.Date) AS Month,
    SUM(bd.[GWBBS Hudson Transit]) AS [Hudson Transit],
    SUM(bd.[GWBBS Rockland]) AS [Rockland],
    SUM(bd.[GWBBS NJ Transit]) AS [NJ Transit],
    SUM(bd.[GWBBS Spanish]) AS [Spanish],
    SUM(bd.[GWBBS Greyhound LD]) AS [Greyhound LD],
    SUM(bd.[GWBBS OurBus LD]) AS [OurBus LD],
    SUM(bd.[GWBBS Saddle River]) AS [Saddle River],
    SUM(bd.[GWBBS Saddle River LD]) AS [Saddle River LD],
    SUM(bd.[GWBBS Vanessa]) AS [Vanessa],
    SUM(bd.[GWBBS_Total]) AS [Total GWBBS],
    SUM(gwbbs.Buses_Count) AS [Total Buses],
    SUM(gwbbs.Passengers_Count) AS [Total Passengers]
FROM 
    Cleaned_Bus_Carrier_Departure bd
JOIN 
    Join_GWBBS_NJT_NYC_Trips_Buses_Passengers gwbbs
    ON FORMAT(bd.Date, 'yyyy-MM') = gwbbs.Date
GROUP BY 
    YEAR(bd.Date),
    MONTH(bd.Date)
ORDER BY 
    YEAR(bd.Date),
    MONTH(bd.Date);

-- Aggregates GWBBS bus and passenger data by month
SELECT 
    YEAR(bd.Date) AS Year,
    MONTH(bd.Date) AS Month,
    bd.[GWBBS Hudson Transit],
    bd.[GWBBS Rockland],
    bd.[GWBBS NJ Transit],
    bd.[GWBBS Spanish],
    bd.[GWBBS Greyhound LD],
    bd.[GWBBS OurBus LD],
    bd.[GWBBS Saddle River],
    bd.[GWBBS Saddle River LD],
    bd.[GWBBS Vanessa],
    bd.[GWBBS_Total],
    gwbbs.Buses_Count,
    gwbbs.Passengers_Count
FROM 
    Cleaned_Bus_Carrier_Departure bd
JOIN 
    Join_GWBBS_NJT_NYC_Trips_Buses_Passengers gwbbs
    ON FORMAT(bd.Date, 'yyyy-MM') = gwbbs.Date;

-- Aggregates GWBBS bus and passenger data by month
SELECT 
    bcd.*, 
    mbt.*, 
    gwbbs.*,
	mbtpb.*
FROM 
    Cleaned_Bus_Carrier_Departure bcd
LEFT JOIN 
    Join_MBT_NJT_NYC_Trips_Buses_Passengers mbt
    ON mbt.Date = LEFT(bcd.Date, 7)
LEFT JOIN 
    Join_GWBBS_NJT_NYC_Trips_Buses_Passengers gwbbs
    ON gwbbs.Date = LEFT(bcd.Date, 7)
LEFT JOIN
    joined_mbt_passenger_bus_departures mbtpb
    ON FORMAT(mbtpb.Start_Date, 'yyyy-MM') = LEFT(bcd.Date, 7)








SELECT 
    bcd.date, bcd.[MBT Greyhound SH], bcd.[MBT Greyhound LD], bcd.[MBT Academy SH], bcd.[MBT NJ Transit SH], bcd.[MBT NJ Transit LD], bcd.[MBT Rockland SH], bcd.[MBT Lakeland SH], bcd.[MBT Martz Trailways SH], bcd.[MBT Martz Trailways LD], bcd.[GWBBS Rockland], bcd.[GWBBS NJ Transit], bcd.[GWBBS Spanish], bcd.[GWBBS Greyhound LD],
    mbt.Route, mbt.Buses_Count, mbt.Passengers_Count,
    gwbbs.Route, gwbbs.Buses_Count, gwbbs.Passengers_Count,
	mbtpb.Start_Date, mbtpb.Total, mbtpb.Bus_Total, mbtpb.Greyhound, mbtpb.Martz, mbtpb.[NJ Transit], mbtpb.Lakeland, mbtpb.Trailways, mbtpb.DeCamp, mbtpb.[Weekly Total], mbtpb.buss_Total, mbtpb.bus_Greyhound, mbtpb.bus_Martz, mbtpb.[bus_NJ Transit], mbtpb.bus_Lakeland, mbtpb.bus_Trailways, mbtpb.bus_DeCamp
FROM 
    Cleaned_Bus_Carrier_Departure bcd
LEFT JOIN 
    Join_MBT_NJT_NYC_Trips_Buses_Passengers mbt
    ON mbt.Date = LEFT(bcd.Date, 7)
LEFT JOIN 
    Join_GWBBS_NJT_NYC_Trips_Buses_Passengers gwbbs
    ON gwbbs.Date = LEFT(bcd.Date, 7)
LEFT JOIN
    joined_mbt_passenger_bus_departures mbtpb
    ON FORMAT(mbtpb.Start_Date, 'yyyy-MM') = LEFT(bcd.Date, 7)








DATASET JOINS - 3 DATASET WITH THE TRAFFIC FOR GETTING THE FACILITY







WITH td_monthly AS (
    SELECT DISTINCT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        FAC_G2
    FROM Cleaned_Traffic_Data
),

gwbbs_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(Buses_Count AS BIGINT)) AS gwbbs_Buses_Count,
        SUM(TRY_CAST(Passengers_Count AS BIGINT)) AS gwbbs_Passengers_Count
    FROM Join_GWBBS_NJT_NYC_Trips_Buses_Passengers
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2)
),

mbt_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(Buses_Count AS BIGINT)) AS mbt_Buses_Count,
        SUM(TRY_CAST(Passengers_Count AS BIGINT)) AS mbt_Passengers_Count
    FROM Join_MBT_NJT_NYC_Trips_Buses_Passengers
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Date, 23), 6, 2)
),

mbtpd_monthly AS (
    SELECT
        LEFT(CONVERT(VARCHAR, Start_Date, 23), 4) AS Year,
        SUBSTRING(CONVERT(VARCHAR, Start_Date, 23), 6, 2) AS Month,
        SUM(TRY_CAST(total AS BIGINT)) AS mbtpd_Total,
        SUM(TRY_CAST(bus_total AS BIGINT)) AS mbtpd_Total_bus
    FROM Cleaned_MBT_Passenger_Departures
    GROUP BY 
        LEFT(CONVERT(VARCHAR, Start_Date, 23), 4),
        SUBSTRING(CONVERT(VARCHAR, Start_Date, 23), 6, 2)
)

SELECT 
    td.Year,
    td.Month,
    td.FAC_G2,
    gwbbs.gwbbs_Buses_Count,
    gwbbs.gwbbs_Passengers_Count,
    mbt.mbt_Buses_Count,
    mbt.mbt_Passengers_Count,
    mbtpd.mbtpd_Total,
    mbtpd.mbtpd_Total_bus
FROM
    td_monthly td
LEFT JOIN gwbbs_monthly gwbbs
    ON td.Year = gwbbs.Year AND td.Month = gwbbs.Month
LEFT JOIN mbt_monthly mbt
    ON td.Year = mbt.Year AND td.Month = mbt.Month
LEFT JOIN mbtpd_monthly mbtpd
    ON td.Year = mbtpd.Year AND td.Month = mbtpd.Month
ORDER BY 
    td.Year, 
    td.Month, 
    td.FAC_G2;

-- General data aggregation or join query
SELECT c.*, w.AWND, w.PRCP, w.SNOW, w.SNWD, w.TMAX, w.TMAX, w.TMIN
FROM Cleaned_Traffic_Data c
left JOIN cleaned_tbl_weather w
ON c.Date = w.Date;

-- Joins traffic and weather data and summarizes by month
SELECT t.DATE, t.LANE, t.TIME, t.TOTAL, t.[CLASS 1], t.[CLASS 9], t.FAC_B, t.Buses, t.FAC_G, t.Day_Name, w.AWND, w.PRCP, w.SNOW, w.SNWD, w.TMAX, w.TMIN
FROM Traffic_Data T
left JOIN Cleaned_Tbl_Weather W
ON T.DATE = W.DATE
order by t.date asc




SELECT 
    YEAR(t.DATE) AS Yr,
    MONTH(t.DATE) AS Mo,
    COUNT(*) AS Record_Count,
    AVG(t.LANE) AS Avg_LANE,
    AVG(t.TIME) AS Avg_TIME,
    SUM(t.TOTAL) AS Total_Vehicles,
    SUM(t.[CLASS 1]) AS Total_Class_1,
    SUM(t.[CLASS 9]) AS Total_Class_9,
    MAX(t.FAC_B) AS Sample_FAC_B,  -- Use MAX or MIN since it's text
    SUM(t.Buses) AS Total_Buses,
    MAX(t.FAC_G) AS Sample_FAC_G,
    MAX(t.Day_Name) AS Sample_Day_Name,
    AVG(w.AWND) AS Avg_Wind_Speed,
    SUM(w.PRCP) AS Total_Precipitation,
    SUM(w.SNOW) AS Total_Snowfall,
    SUM(w.SNWD) AS Total_SnowDepth,
    AVG(w.TMAX) AS Avg_TMAX,
    AVG(w.TMIN) AS Avg_TMIN
FROM Traffic_Data t
LEFT JOIN Cleaned_Tbl_Weather w ON t.DATE = w.DATE
GROUP BY YEAR(t.DATE), MONTH(t.DATE)
ORDER BY Yr, Mo;

-- Aggregates weather data by year and month
WITH WeatherAgg AS (
    SELECT 
        YEAR([DATE]) AS Year,
        MONTH([DATE]) AS Month,
        MAX(AWND) AS AWND,
        MAX(PRCP) AS PRCP,
        MAX(SNOW) AS SNOW,
        MAX(SNWD) AS SNWD,
        MAX(TMAX) AS TMAX,
        MAX(TMIN) AS TMIN
    FROM Cleaned_Tbl_Weather
    GROUP BY YEAR([DATE]), MONTH([DATE])
)

SELECT
    YEAR(p.Start_Date) AS Year,
    MONTH(p.Start_Date) AS Month,
    
    -- Passenger counts
    SUM(p.Academy) AS passenger_academy,
    SUM(b.Academy) AS bus_academy,
    SUM(p.Greyhound) AS passenger_greyhound,
    SUM(b.Greyhound) AS bus_greyhound,
    SUM(p.[Coach USA]) AS passenger_coach_usa,
    SUM(b.[Coach USA]) AS bus_coach_usa,
    SUM(p.[TransBridge]) AS passenger_transbridge,
    SUM(b.[TransBridge]) AS bus_transbridge,
    SUM(p.[Peter Pan/Bonanza]) AS passenger_peterpan,
    SUM(b.[Peter Pan/Bonanza]) AS bus_peterpan,
    SUM(p.[C & J Bus Lines]) AS passenger_cj,
    SUM(b.[C & J Bus Lines]) AS bus_cj,

    -- Weather (no aggregation here)
    w.AWND,
    w.PRCP,
    w.SNOW,
    w.SNWD,
    w.TMAX,
    w.TMIN

FROM 
    cleaned_mbt_passenger_departures p
JOIN 
    cleaned_mbt_bus_departures b
    ON p.Start_Date = b.Start_Date
    AND p.End_Date = b.End_Date

LEFT JOIN 
    WeatherAgg w
    ON YEAR(p.Start_Date) = w.Year AND MONTH(p.Start_Date) = w.Month

GROUP BY
    YEAR(p.Start_Date),
    MONTH(p.Start_Date),
    w.AWND,
    w.PRCP,
    w.SNOW,
    w.SNWD,
    w.TMAX,
    w.TMIN

ORDER BY
    Year, Month;;