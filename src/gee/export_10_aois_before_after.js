// ======================================================
// Urban Growth Detection – Export Script for 10 AOIs (USA)
// Exports "before" and "after" Sentinel-2 composites
// ======================================================

// IMPORTANT:
// You have already drawn these polygons in GEE Map:
// aoi1 = Austin (Pflugerville)
// aoi2 = Dallas (Frisco)
// aoi3 = Houston (Birnham Woods / Benders Landing)
// aoi4 = Phoenix (Gilbert / Seville)
// aoi5 = Denver (Meridian Village / Stonegate)
// aoi6 = Las Vegas (Inspirada / West Henderson)
// aoi7 = Charlotte (Steele Creek / Shopton–Youngblood)
// aoi8 = Apex (Haddon Hall / Villages of Apex)
// aoi9 = Brier Creek (Raleigh–Durham)
// aoi10 = North Hills (Raleigh Midtown)

// ====== Time Windows ======
var BEFORE_START = '2017-01-01';
var BEFORE_END   = '2017-12-31';

var AFTER_START  = '2023-01-01';
var AFTER_END    = '2024-12-31';

// ====== Sentinel-2 SR and Cloud Mask ======
var s2 = ee.ImageCollection('COPERNICUS/S2_SR');

// Keep only the optical bands we actually need
var opticalBands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'];

function maskS2clouds(image) {
  // Use SCL from the original image
  var scl = image.select('SCL');
  
  var mask = scl.neq(3)   // CLOUD_SHADOWS
              .and(scl.neq(8))   // CLOUD_MEDIUM_PROBABILITY
              .and(scl.neq(9))   // CLOUD_HIGH_PROBABILITY
              .and(scl.neq(10))  // THIN_CIRRUS
              .and(scl.neq(11)); // SNOW_ICE

  // 1) Select ONLY the optical bands so all images have same bands
  // 2) Apply the cloud mask
  // 3) Scale to surface reflectance
  return image
    .select(opticalBands)
    .updateMask(mask)
    .divide(10000);
}

function compositeForAOI(aoi, start, end, label) {
  var comp = s2
    .filterBounds(aoi)
    .filterDate(start, end)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    .map(maskS2clouds)
    .median()
    .clip(aoi);

  Map.addLayer(comp, 
    {bands: ['B4','B3','B2'], min:0.02, max:0.30}, 
    label, 
    false);
    
  return comp;
}

// ====== AOI List (Your Actual 10 AOIs) ======
var aois = [
  {name: 'Austin_Pflugerville',  geom: aoi1},
  {name: 'Dallas_Frisco',        geom: aoi2},
  {name: 'Houston_Spring',       geom: aoi3},
  {name: 'Phoenix_Gilbert',      geom: aoi4},
  {name: 'Denver_Stonegate',     geom: aoi5},
  {name: 'LasVegas_Inspirada',   geom: aoi6},
  {name: 'Charlotte_SteeleCreek',geom: aoi7},
  {name: 'Apex_HaddonHall',      geom: aoi8},
  {name: 'Raleigh_BrierCreek',   geom: aoi9},
  {name: 'Raleigh_NorthHills',   geom: aoi10}
];

// ====== Export Loop ======
aois.forEach(function(entry) {
  var name = entry.name;
  var geom = entry.geom;

  print('Processing AOI:', name);
  Map.centerObject(geom, 12);

  var before = compositeForAOI(geom, BEFORE_START, BEFORE_END, name + '_Before');
  var after  = compositeForAOI(geom, AFTER_START, AFTER_END, name + '_After');

  // Export Before
  Export.image.toDrive({
    image: before,
    description: name + '_BEFORE',
    folder: 'GEE_UrbanGrowth_10AOI',
    fileNamePrefix: name + '_before',
    region: geom,
    scale: 10,
    maxPixels: 1e13
  });

  // Export After
  Export.image.toDrive({
    image: after,
    description: name + '_AFTER',
    folder: 'GEE_UrbanGrowth_10AOI',
    fileNamePrefix: name + '_after',
    region: geom,
    scale: 10,
    maxPixels: 1e13
  });
});