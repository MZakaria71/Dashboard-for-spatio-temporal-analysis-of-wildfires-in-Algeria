/**
 * ═══════════════════════════════════════════════════════════════════════════
 *  Algeria Wildfire Dashboard — GEE Data Export Script
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *  Products:
 *    MODIS/061/MCD64A1     — Monthly burned area, 500 m (burn date DOY)
 *    MODIS/061/MCD12Q1     — Annual land cover IGBP LC_Type1, 500 m
 *    FAO/GAUL/2015/level2  — Algeria commune boundaries (includes wilaya info)
 *
 *  Strategy: export at ADM2 (commune) level only — ADM1 (wilaya) totals are
 *  derived by groupby aggregation in prepare_dataset.py, so GEE only runs
 *  240 reduceRegions calls instead of 480.
 *
 *  Output — Google Drive folder: "Algeria_Wildfire_Dashboard"
 *  ┌─────────────────────────────────────────────────────────────────────┐
 *  │  burned_area_adm2_month.csv                                         │
 *  │    ADM1_CODE | ADM1_NAME | ADM2_CODE | ADM2_NAME | year | month    │
 *  │    burned_forest_km2 | burned_shrubland_km2 | burned_cropland_km2  │
 *  │    burned_other_km2  | burned_total_km2                            │
 *  │                                                                     │
 *  │  landcover_adm2_year.csv                                            │
 *  │    ADM1_CODE | ADM1_NAME | ADM2_CODE | ADM2_NAME | year            │
 *  │    forest_km2 | shrubland_km2 | cropland_km2                       │
 *  │    other_km2  | total_km2                                          │
 *  └─────────────────────────────────────────────────────────────────────┘
 *
 *  Land cover reclassification (IGBP LC_Type1 → 4 classes):
 *    Forest    : 1-5 (forest types) + 8 (woody savannas)
 *    Shrubland : 6-7 (shrublands) + 9 (savannas) + 10 (grasslands)
 *    Cropland  : 12 (croplands) + 14 (cropland/natural veg mosaic)
 *    Other     : 11, 13, 15, 16, 17 (wetlands, urban, ice, barren, water)
 *
 *  Period : see START_YEAR/END_YEAR below  |  Scale: 500 m (MODIS native)
 *
 *  HOW TO USE:
 *    1. Paste into GEE Code Editor (code.earthengine.google.com)
 *    2. Click "Run" — map loads, Tasks panel shows 2 tasks
 *    3. Click "Run" on each task in the Tasks panel
 *    4. Download both CSVs from Drive → "Algeria_Wildfire_Dashboard/"
 *    5. Run prepare_dataset.py to produce all Parquet files
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ─────────────────────────────────────────────────────────────────────────
// 0.  CONFIGURATION
// ─────────────────────────────────────────────────────────────────────────
var START_YEAR   = 2001;
var END_YEAR     = 2026;

// Two caveats when extending past 2020:
//
//  1. MCD64A1 burned area is available to 2026-06-01, so 2026 is a PARTIAL
//     year (roughly January-May). Its annual total is not comparable with a
//     full year and should be excluded from any trend fit.
//
//  2. MCD12Q1 land cover ends in 2023. Section 3 already falls back to the most
//     recent available year, so 2024-2026 burned area is split by cover type
//     using the 2023 land-cover map. Fine for a stable landscape, but it will
//     not reflect land-use change in those years.
//
// The burned-area export now runs END_YEAR-START_YEAR+1 = 26 years x 12 months
// = 312 reduceRegions calls. If a task fails with an out-of-memory or timeout
// error, raise TILE_SCALE to 8 or 16 and re-run.
var DRIVE_FOLDER = 'Algeria_Wildfire_Dashboard';
var SCALE        = 500;   // metres — MODIS 500 m native resolution
var TILE_SCALE   = 4;     // increase to 8 if a task fails with memory error

// ─────────────────────────────────────────────────────────────────────────
// 1.  BOUNDARIES  (FAO GAUL 2015, Admin Level 2 — communes / daïras)
//     Level-2 features carry ADM1_CODE/ADM1_NAME from the parent wilaya,
//     so we get both admin levels in a single reduceRegions pass.
// ─────────────────────────────────────────────────────────────────────────
var communes = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'Algeria'))
  .select(['ADM1_CODE', 'ADM1_NAME', 'ADM2_CODE', 'ADM2_NAME']);

print('ADM1 (wilayas) count — cross-check via level1:',
  ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM0_NAME', 'Algeria')).size()
);
print('ADM2 (communes) count:', communes.size());  // expect ~553

Map.centerObject(communes, 5);
Map.addLayer(communes, {color: '888888', fillColor: '00000000'}, 'Algeria Communes (ADM2)');
Map.addLayer(
  ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM0_NAME', 'Algeria')),
  {color: 'cc0000', fillColor: '00000000'},
  'Algeria Wilayas (ADM1)'
);

// ─────────────────────────────────────────────────────────────────────────
// 2.  HELPER FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────

// Pixel area in km² — geographically correct per-pixel area
var pixelAreaKm2 = ee.Image.pixelArea().divide(1e6);

/**
 * Reclassify MCD12Q1 LC_Type1 (IGBP, 1-17) → 4 simplified classes:
 *   1 = Forest   2 = Shrubland   3 = Cropland   4 = Other
 * Unclassified / masked pixels → 4 (Other)
 */
function reclassifyLC(lcBand) {
  var igbpFrom = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17];
  var classTo  = [1,  1,  1,  1,  1,  2,  2,  1,  2,  2,  4,  3,  4,  3,  4,  4,  4];
  return lcBand.remap(igbpFrom, classTo, 4).rename('lc_class');
}

/**
 * Fetch one MCD64A1 burn-date image for a given year+month.
 * Returns an all-zero image when no scene is available (no export gaps).
 */
function getBurnImage(year, month) {
  var m2  = month < 10 ? '0' + month : String(month);
  var nm  = (month % 12) + 1;
  var ny  = month === 12 ? year + 1 : year;
  var nm2 = nm < 10 ? '0' + nm : String(nm);

  var col = ee.ImageCollection('MODIS/061/MCD64A1')
    .filterDate(year + '-' + m2 + '-01', ny + '-' + nm2 + '-01')
    .select('BurnDate');

  return ee.Image(ee.Algorithms.If(
    col.size().gt(0),
    col.first(),
    ee.Image.constant(0).rename('BurnDate').selfMask().unmask(0)
  ));
}

/**
 * Return a 5-band image: burned area in km² per LC class + total.
 * Non-burned or wrong-class pixels contribute 0 to the zonal sum.
 */
function burnedByClass(burnImg, lcReclass) {
  var burned = burnImg.gt(0);
  return burned.and(lcReclass.eq(1)).multiply(pixelAreaKm2).rename('burned_forest_km2')
    .addBands(burned.and(lcReclass.eq(2)).multiply(pixelAreaKm2).rename('burned_shrubland_km2'))
    .addBands(burned.and(lcReclass.eq(3)).multiply(pixelAreaKm2).rename('burned_cropland_km2'))
    .addBands(burned.and(lcReclass.eq(4)).multiply(pixelAreaKm2).rename('burned_other_km2'))
    .addBands(burned.multiply(pixelAreaKm2).rename('burned_total_km2'));
}

// ─────────────────────────────────────────────────────────────────────────
// 3.  BURNED AREA TABLE  —  commune × year × month × LC class
//     Rows: N_communes × 20 years × 12 months  (e.g. 553 × 240 = 132 720)
// ─────────────────────────────────────────────────────────────────────────
print('Building burned area computation graph (20 years × 12 months)…');

var burnedFeatures = [];

for (var year = START_YEAR; year <= END_YEAR; year++) {

  // Annual land cover — fall back to most-recent available year if missing
  var lcCol = ee.ImageCollection('MODIS/061/MCD12Q1')
    .filterDate(year + '-01-01', (year + 1) + '-01-01')
    .select('LC_Type1');

  var lcImg = ee.Image(ee.Algorithms.If(
    lcCol.size().gt(0),
    lcCol.first(),
    ee.ImageCollection('MODIS/061/MCD12Q1')
      .sort('system:time_start', false)
      .first()
      .select('LC_Type1')
  ));

  var lcReclass = reclassifyLC(lcImg);

  for (var month = 1; month <= 12; month++) {

    var burnImg   = getBurnImage(year, month);
    var multiBand = burnedByClass(burnImg, lcReclass);

    var zonal = multiBand.reduceRegions({
      collection : communes,
      reducer    : ee.Reducer.sum(),
      scale      : SCALE,
      tileScale  : TILE_SCALE
    });

    // IIFE to capture loop variables (avoids JS closure trap)
    burnedFeatures.push(zonal.map((function(y, m) {
      return function(f) { return f.set('year', y, 'month', m); };
    })(year, month)));
  }
}

var burnedFC = burnedFeatures.reduce(function(acc, fc) { return acc.merge(fc); });

print('Burned area features (expected N_communes × 240):', burnedFC.size());

Export.table.toDrive({
  collection     : burnedFC,
  description    : 'burned_area_adm2_month',
  folder         : DRIVE_FOLDER,
  fileNamePrefix : 'burned_area_adm2_month',
  fileFormat     : 'CSV',
  selectors      : [
    'ADM1_CODE', 'ADM1_NAME', 'ADM2_CODE', 'ADM2_NAME',
    'year', 'month',
    'burned_forest_km2', 'burned_shrubland_km2',
    'burned_cropland_km2', 'burned_other_km2', 'burned_total_km2'
  ]
});

// ─────────────────────────────────────────────────────────────────────────
// 4.  LAND COVER TABLE  —  commune × year
//     Rows: N_communes × 20 years  (e.g. 553 × 20 = 11 060)
// ─────────────────────────────────────────────────────────────────────────
print('Building land cover computation graph (20 years)…');

var lcFeatures = [];

for (var lcYear = START_YEAR; lcYear <= END_YEAR; lcYear++) {

  var lcYearCol = ee.ImageCollection('MODIS/061/MCD12Q1')
    .filterDate(lcYear + '-01-01', (lcYear + 1) + '-01-01')
    .select('LC_Type1');

  var annualLC = ee.Image(ee.Algorithms.If(
    lcYearCol.size().gt(0),
    lcYearCol.first(),
    ee.ImageCollection('MODIS/061/MCD12Q1')
      .sort('system:time_start', false)
      .first()
      .select('LC_Type1')
  ));

  var reclass = reclassifyLC(annualLC);

  var lcBands = reclass.eq(1).multiply(pixelAreaKm2).rename('forest_km2')
    .addBands(reclass.eq(2).multiply(pixelAreaKm2).rename('shrubland_km2'))
    .addBands(reclass.eq(3).multiply(pixelAreaKm2).rename('cropland_km2'))
    .addBands(reclass.eq(4).multiply(pixelAreaKm2).rename('other_km2'))
    .addBands(pixelAreaKm2.rename('total_km2'));

  var lcZonal = lcBands.reduceRegions({
    collection : communes,
    reducer    : ee.Reducer.sum(),
    scale      : SCALE,
    tileScale  : TILE_SCALE
  });

  lcFeatures.push(lcZonal.map((function(y) {
    return function(f) { return f.set('year', y); };
  })(lcYear)));
}

var lcFC = lcFeatures.reduce(function(acc, fc) { return acc.merge(fc); });

print('Land cover features (expected N_communes × 20):', lcFC.size());

Export.table.toDrive({
  collection     : lcFC,
  description    : 'landcover_adm2_year',
  folder         : DRIVE_FOLDER,
  fileNamePrefix : 'landcover_adm2_year',
  fileFormat     : 'CSV',
  selectors      : [
    'ADM1_CODE', 'ADM1_NAME', 'ADM2_CODE', 'ADM2_NAME', 'year',
    'forest_km2', 'shrubland_km2', 'cropland_km2', 'other_km2', 'total_km2'
  ]
});

// ─────────────────────────────────────────────────────────────────────────
// 5.  SANITY CHECK — visualise 2019 burn dates on the map
// ─────────────────────────────────────────────────────────────────────────
var burnCheck2019 = ee.ImageCollection('MODIS/061/MCD64A1')
  .filterDate('2019-01-01', '2020-01-01')
  .filterBounds(communes)
  .select('BurnDate')
  .max()
  .selfMask();

Map.addLayer(
  burnCheck2019,
  {min: 1, max: 366, palette: ['ffffb2', 'fecc5c', 'fd8d3c', 'f03b20', 'bd0026']},
  '2019 Burn Date DOY (sanity check)'
);

var nationalBurn2019 = burnCheck2019.gt(0)
  .multiply(pixelAreaKm2)
  .reduceRegion({
    reducer   : ee.Reducer.sum(),
    geometry  : communes.geometry(),
    scale     : SCALE,
    tileScale : TILE_SCALE,
    maxPixels : 1e10
  });
print('~2019 national burned area (km²):', nationalBurn2019);

// ─────────────────────────────────────────────────────────────────────────
// 6.  BOUNDARY EXPORTS  —  for the ignition pipeline and the dashboard map
//
//     The repo currently ships Dz_adm1.shp, which uses the CURRENT 58-wilaya
//     scheme. The burned-area tables above are aggregated on FAO GAUL 2015,
//     which predates the 2019 reorganisation and has 48 wilayas. Exporting the
//     boundaries from GAUL guarantees the map and the data agree, and gives
//     prepare_ignitions.py the polygons it needs to place ignition points.
//
//     gaul_adm2.geojson  full resolution — point-in-polygon join, offline only
//     gaul_adm1.geojson  simplified      — the dashboard choropleth
// ─────────────────────────────────────────────────────────────────────────
var wilayas = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Algeria'))
  .select(['ADM1_CODE', 'ADM1_NAME']);

Export.table.toDrive({
  collection     : communes,
  description    : 'gaul_adm2_geojson',
  folder         : DRIVE_FOLDER,
  fileNamePrefix : 'gaul_adm2',
  fileFormat     : 'GeoJSON'
});

// ~500 m simplification: invisible at national zoom, but roughly an order of
// magnitude smaller to ship to the browser on every map render.
Export.table.toDrive({
  collection     : wilayas.map(function (f) {
                     return f.setGeometry(f.geometry().simplify(500));
                   }),
  description    : 'gaul_adm1_geojson',
  folder         : DRIVE_FOLDER,
  fileNamePrefix : 'gaul_adm1',
  fileFormat     : 'GeoJSON'
});
