#![doc(
    html_logo_url = "https://raw.githubusercontent.com/virtualritz/libh3/master/h3Logo-color.svg"
)]
//! H3 is a geospatial indexing system that partitions the world into hexagonal
//! cells.
//!
//! The crate wraps the C-API for the H3 grid system. It includes functions for
//! converting from latitude and longitude coordinates to the containing H3
//! cell, finding the center of H3 cells, finding the boundary geometry of H3
//! cells, finding neighbors of H3 cells, and more.
//!
//! * The H3 Core Library is written in *C*. [Bindings for many other
//!   languages](/docs/community/bindings) are available.
//!
//! ## Highlights
//!
//! * H3 is a hierarchical [geospatial index](/docs/highlights/indexing).
//! * H3 was developed to address the [challenges of Uber's data science
//!   needs](/docs/highlights/aggregation).
//! * H3 can be used to [join disparate data sets](/docs/highlights/joining).
//! * In addition to the benefits of the hexagonal grid shape, H3 includes
//!   features for [modeling flow](/docs/highlights/flowmodel).
//! * H3 is well suited to apply [ML to geospatial data](/docs/highlights/ml).
//!
//! ## Comparisons
//!
//! * [S2](/docs/comparisons/s2), an open source, hierarchical, discrete, and
//!   global grid system using square cells.
//! * [Geohash](/docs/comparisons/geohash), a system for encoding locations
//!   using a string of characters, creating a hierarchical, square grid system
//!   (a quadtree).
//! * [Hexbin](/docs/comparisons/hexbin), the process of taking coordinates and
//!   binning them into hexagonal cells in analytics or mapping software.
//! * [Admin Boundaries](/docs/comparisons/admin), officially designated areas
//!   used for aggregating and analyzing data.
//! * [Placekey](/docs/comparisons/placekey), a system for encoding points of
//!   interest (POIs) which incorporates H3 in its POI identifier.
use derive_more::{Deref, From, Into};
use std::mem::MaybeUninit;

/// A coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeoCoord {
    // The latitute of the coordinate, typcially this should be specified using
    // radians but it is easy to convert using [degs_to_rads](degs_to_rads)
    pub lat: f64,
    // The longitude of the coordinate, typcially this should be specified using
    // radians but it is easy to convert using [degs_to_rads](degs_to_rads)
    pub lon: f64,
}

impl GeoCoord {
    /// Create a new GeoCoord representing a coordinate
    ///
    /// # Arguments
    ///
    /// * `lat` - The latitude of the coordinate
    /// * `long` - The longitude of the coordinate
    pub fn new(lat: f64, lon: f64) -> GeoCoord {
        GeoCoord { lat, lon }
    }
}

impl From<&GeoCoord> for libh3_sys::GeoCoord {
    fn from(coord: &GeoCoord) -> Self {
        libh3_sys::GeoCoord {
            lat: coord.lat,
            lon: coord.lon,
        }
    }
}

/// A H3 index value a unique address of a hexagon or, less likely,
/// a pentagon.
#[derive(PartialEq, Eq, Copy, Clone, Debug, From, Into)]
pub struct H3Index(libh3_sys::H3Index);

/// A resolution that ranges from 0 to 15.
///
/// Below is a table lisiting the number of hexagons at each H3 resolution. There are always [exactly *twelve* pentagons](https://h3geo.org/docs/core-library/overview) at every resolution.
///
/// Resolution | Total Number of Cells | Number of Hexagons
/// --: | --: | --:
/// 0 | 122 | 110
/// 1 | 842 | 830
/// 2 | 5,882 | 5,870
/// 3 | 41,162 | 41,150
/// 4 | 288,122 | 288,110
/// 5 | 2,016,842 | 2,016,830
/// 6 | 14,117,882 | 14,117,870
/// 7 | 98,825,162 | 98,825,150
/// 8 | 691,776,122 | 691,776,110
/// 9 | 4,842,432,842 | 4,842,432,830
/// 10 | 33,897,029,882 | 33,897,029,870
/// 11 | 237,279,209,162 | 237,279,209,150
/// 12 | 1,660,954,464,122 | 1,660,954,464,110
/// 13 | 11,626,681,248,842 | 11,626,681,248,830
/// 14 | 81,386,768,741,882 | 81,386,768,741,870
/// 15 | 569,707,381,193,162 | 569,707,381,193,150
#[derive(PartialEq, Eq, Copy, Clone, Debug, Into)]
pub struct Resolution(u8);

impl TryFrom<u8> for Resolution {
    type Error = &'static str;

    fn try_from(r: u8) -> Result<Self, Self::Error> {
        if r < 16 {
            Ok(Resolution(r))
        } else {
            Err("Resolution must be in 0..15.")
        }
    }
}

/// Return the edge length of a hexagon at a particular resolution in
/// kilometers.
///
/// ```
/// use libh3::edge_length_km;
/// assert_eq!(edge_length_km(5), 8.544408276);
/// ```
pub fn hex_edge_length_km(resolution: Resolution) -> f64 {
    unsafe { libh3_sys::edgeLengthKm(resolution.0 as _) }
}

/// Return the number of hexagons at a particular resolution.
///
/// ```
/// use libh3::num_hexagons;
/// assert_eq!(num_hexagons(5), 2016842);
/// ```
pub fn hex_count_at_res(resolution: Resolution) -> usize {
    unsafe { libh3_sys::numHexagons(resolution.0 as _) as _ }
}

/// Return the edge length of a hexagon at a particular resolution in meters.
///
/// ```
/// use libh3::edge_length_m;
/// assert_eq!(edge_length_m(5), 8544.408276);
/// ```
pub fn hex_edge_length_m(resolution: Resolution) -> f64 {
    unsafe { libh3_sys::edgeLengthM(resolution.0 as _) }
}

/// Determine the area of a hexagon at a particular resolution.
///
/// ```
/// use libh3::hex_area_km_2;
/// assert_eq!(hex_area_km_2(10), 0.0150475);
/// ```
pub fn hex_area_sq_km(resolution: i32) -> f64 {
    unsafe { libh3_sys::hexAreaKm2(resolution) }
}

/// Number of resolution 0 H3 indexes.
///
/// ```
/// use libh3::res_zero_index_len;
///
/// assert_eq!(res_zero_index_len(), 122);
/// ```
pub fn res_zero_index_len() -> u32 {
    (unsafe { libh3_sys::res0IndexCount() }) as _
}

/// All the resolution 0 H3 indexes.
///
/// ```
/// use libh3::res_zero_indexes;
///
/// assert_eq!(res_zero_indexes().len(), 122);
/// ```
pub fn res_zero_indexes() -> Vec<H3Index> {
    let max = res_zero_index_len() as _;
    let mut result = Vec::<H3Index>::with_capacity(max);

    unsafe {
        libh3_sys::getRes0Indexes(result.as_mut_ptr() as _);
        result.set_len(max);
        result
    }
}

/// Get all hexagons with centers contained in a given polygon. The polygon
/// is specified with GeoJson semantics as an array of loops. The first loop
/// is the perimeter of the polygon, and subsequent loops are
/// expected to be holes.
///
/// # Arguments
///
/// * `polygon` - The vector of polygons.
/// * `resolution` - The resolution of the generated hexagons
///
/// ```
/// use libh3::{degs_to_rads, polyfill, GeoCoord};
/// /// Some vertexes around San Francisco
/// let sf_verts = vec![
///     (0.659966917655, -2.1364398519396),
///     (0.6595011102219, -2.1359434279405),
///     (0.6583348114025, -2.1354884206045),
///     (0.6581220034068, -2.1382437718946),
///     (0.6594479998527, -2.1384597563896),
///     (0.6599990002976, -2.1376771158464),
/// ]
/// .iter()
/// .map(|v| GeoCoord::new(v.0, v.1))
/// .collect();
///
/// let h = polyfill(&vec![sf_verts], 9);
/// assert_eq!(h.len(), 1253);
///
/// /// Fill a polygon around Wellington, NZ
/// /// Coordinates are in GeoJSON lon, lat order.
/// let wellington_verts: Vec<GeoCoord> = vec![
///     (174.800937866947, -41.22501356278325),
///     (174.8079721211159, -41.226732341115365),
///     (174.82262997231396, -41.231639803277986),
///     (174.83561105648377, -41.23873115217201),
///     (174.84634815587896, -41.24769587535717),
///     (174.85069634833735, -41.252194466801384),
///     (174.8587207192276, -41.26264015566857),
///     (174.8636809159909, -41.27410982273725),
///     (174.86536017144866, -41.28610182569948),
///     (174.8653611411562, -41.29165993179072),
///     (174.8636858034186, -41.30364998147521),
///     (174.85872883206798, -41.31511423541933),
///     (174.8507068877321, -41.32555201329627),
///     (174.846359294586, -41.33004662498431),
///     (174.8356222320399, -41.33900220579377),
///     (174.82263983163466, -41.346084796073406),
///     (174.807979604528, -41.35098543989819),
///     (174.80094378133927, -41.35270175860709),
///     (174.78524618901284, -41.35520670405109),
///     (174.76919781098724, -41.35520670405109),
///     (174.75350021866086, -41.35270175860712),
///     (174.7464643954721, -41.35098543989822),
///     (174.7318041683653, -41.346084796073406),
///     (174.71882176795995, -41.33900220579369),
///     (174.7080847054138, -41.330046624984135),
///     (174.70373711226773, -41.32555201329609),
///     (174.69571516793187, -41.31511423541913),
///     (174.69075819658127, -41.303649981474955),
///     (174.68908285884382, -41.29165993179046),
///     (174.68908382855136, -41.2861018256992),
///     (174.69076308400918, -41.274109822737074),
///     (174.69572328077246, -41.26264015566849),
///     (174.70374765166264, -41.252194466801384),
///     (174.70809584412103, -41.24769587535717),
///     (174.71883294351622, -41.23873115217201),
///     (174.731814027686, -41.231639803278014),
///     (174.746471878884, -41.22673234111539),
///     (174.75350613305287, -41.22501356278328),
///     (174.7691998725514, -41.222504896122565),
///     (174.78524412744844, -41.222504896122565),
///     (174.800937866947, -41.22501356278325),
/// ]
/// .iter()
/// .map(|v| GeoCoord::new(v.1.to_radians(), v.0.to_radians()))
/// .collect();
///
/// let mut h = polyfill(&vec![wellington_verts], 6);
/// assert_eq!(h.len(), 5);
/// h.sort_unstable();
/// assert_eq!(
///     h,
///     vec![
///         606774924341673983,
///         606774925281198079,
///         606774925549633535,
///         606774929307729919,
///         606774929441947647
///     ]
/// );
/// ```
pub fn polyfill(polygon: &[Vec<GeoCoord>], resolution: Resolution) -> Vec<H3Index> {
    let real_polygon: Vec<Vec<libh3_sys::GeoCoord>> = polygon
        .iter()
        .map(|p| p.iter().map(libh3_sys::GeoCoord::from).collect())
        .collect();

    unsafe {
        let fence = libh3_sys::Geofence {
            numVerts: real_polygon[0].len() as _,
            verts: real_polygon[0].as_ptr(),
        };

        let holes: Vec<libh3_sys::Geofence> = real_polygon
            .iter()
            .skip(1)
            .map(|p| libh3_sys::Geofence {
                numVerts: p.len() as i32,
                verts: p.as_ptr(),
            })
            .collect();

        let p = libh3_sys::GeoPolygon {
            geofence: fence,
            numHoles: (real_polygon.len() - 1) as i32,
            holes: holes.as_ptr(),
        };

        let max = libh3_sys::maxPolyfillSize(&p, resolution.0 as _);
        let mut r = Vec::<H3Index>::with_capacity(max as usize);
        libh3_sys::polyfill(&p, resolution.0 as _, r.as_mut_ptr() as _);
        r.set_len(max as usize);
        r.retain(|&v| v.0 != 0);
        r
    }
}

impl From<H3Index> for GeoCoord {
    /// Construct a `GeoCoord` from a [`H3Index`].
    ///
    /// ```
    /// # use libh3::GeoCoord;
    /// let r = GeoCoord::from(0x8a2a1072b59ffff);
    /// assert_eq!(r.lat, 0.7101643819054542);
    /// assert_eq!(r.lon, -1.2923191206954798);
    /// ```
    fn from(h3: H3Index) -> Self {
        let result = unsafe {
            let mut result: MaybeUninit<libh3_sys::GeoCoord> = MaybeUninit::uninit();
            libh3_sys::h3ToGeo(h3.0, result.as_mut_ptr());
            result.assume_init()
        };

        Self::new(result.lat, result.lon)
    }
}

#[derive(Deref)]
struct GeoBoundary(Vec<GeoCoord>);

impl From<H3Index> for GeoBoundary {
    /// Construct a `GeoBoundary` from a [`H3Index`].
    ///
    /// ```
    /// # use libh3::GeoBoundary;
    /// let foo = GeoBoundary::from(0x8a2a1072b59ffff);
    /// assert_eq!(foo.len(), 6);
    /// ```
    fn from(h3: H3Index) -> Self {
        let boundary = unsafe {
            let mut boundary_result: MaybeUninit<libh3_sys::GeoBoundary> = MaybeUninit::uninit();
            libh3_sys::h3ToGeoBoundary(h3.0, boundary_result.as_mut_ptr());
            boundary_result.assume_init()
        };

        Self(
            boundary
                .verts
                .iter()
                .map(|vertex| GeoCoord::new(vertex.lat, vertex.lon))
                .collect(),
        )
    }
}

impl H3Index {
    /// Try creating an index from a [`GeoCoord`] and a [`Resolution`].
    ///
    /// ```
    /// # use libh3::{GeoCoord, H3Index};
    /// let coords = GeoCoord {
    ///     lat: 40.689167.to_radians(),
    ///     lon: -74.044444.to_radians(),
    /// };
    ///
    /// let v = H3Index::try_from_geo_coord(&coords, 10)?;
    /// assert_eq!(v, 0x8a2a1072b59ffff);
    /// ```
    pub fn try_from(coord: &GeoCoord, resolution: Resolution) -> Option<H3Index> {
        match unsafe { libh3_sys::geoToH3(&libh3_sys::GeoCoord::from(coord), resolution.0 as _) } {
            0 => None,
            x => Some(H3Index(x)),
        }
    }

    /// Return the resolution this index.
    ///
    /// ```
    /// # use libh3::{GeoCoord, H3Index};
    /// let coords = GeoCoord::new(
    ///    lat: 40.689167.to_radians(),
    ///    lon: -74.044444.to_radians(),
    /// );
    ///
    /// let v = H3Index::try_from_geo_coord(&coords, 10)?;
    /// assert_eq!(v.resolution(), 10);
    /// ````
    pub fn resolution(self) -> Resolution {
        Resolution(unsafe { libh3_sys::h3GetResolution(self.0) as _ })
    }

    /// Check if the index is valid.
    ///
    /// ```
    /// # use libh3::{GeoCoord, H3Index};
    /// let coords = GeoCoord::new(
    ///    lat: 40.689167.to_radians(),
    ///    lon: -74.044444.to_radians(),
    /// );
    ///
    /// let v = H3Index::try_from_geo_coord(&coords, 10)?;
    /// assert_eq!(v.is_valid(), true);
    /// ```
    pub fn is_valid(self) -> bool {
        !matches!(unsafe { libh3_sys::h3IsValid(self.0) }, 0)
    }

    /// Check if the given index is a neighbor.
    ///
    /// ```
    /// # use libh3::{GeoCoord, H3Index};
    /// let coords = GeoCoord::new(
    ///    lat: 40.689167.to_radians(),
    ///    lon: -74.044444.to_radians(),
    /// );
    ///
    /// let v = H3Index::try_from_geo_coord(&coords, 10)?;
    /// assert_eq!(v.is_neighbor(v), false);
    /// ```
    pub fn is_neighbor(self, destination: H3Index) -> bool {
        !matches!(
            unsafe { libh3_sys::h3IndexesAreNeighbors(self.0, destination.0) },
            0
        )
    }

    /// Determine if the index is a pentagon.
    /// ```
    /// # use libh3::H3Index;
    /// assert_eq!(H3Index::from(0x8a2a1072b59ffff).is_pentagon(), false);
    /// ```
    pub fn is_pentagon(self) -> bool {
        !matches!(unsafe { libh3_sys::h3IsPentagon(self.0) }, 0)
    }

    /// Get the number of the base cell for a given H3 index
    ///
    /// ```
    /// # use libh3::H3Index;
    /// assert_eq!(H3Index::from(0x8a2a1072b59ffff).base_cell(), 21);
    /// ```
    pub fn base_cell(self) -> u32 {
        (unsafe { libh3_sys::h3GetBaseCell(self.0) }) as _
    }

    /// Get all hexagons in a k-ring around a given center. The order of the
    /// hexagons is undefined.
    ///
    /// # Arguments
    ///
    /// * `origin` - The center of the ring.
    /// * `radius` - The radis of the ring in hexagons, which is the same
    ///   resolution as the origin.
    ///
    /// ```
    /// # use libh3::H3Index;
    /// let expected_k_ring = vec![
    ///     0x8a2a1072b59ffff,
    ///     0x8a2a1072b597fff,
    ///     0x8a2a1070c96ffff,
    ///     0x8a2a1072b4b7fff,
    ///     0x8a2a1072b4a7fff,
    ///     0x8a2a1072b58ffff,
    ///     0x8a2a1072b587fff,
    /// ]
    /// .iter()
    /// .map(|v| v.into())
    /// .collect();
    ///
    /// let r = H3Index::from(0x8a2a1072b59ffff).k_ring(1);
    /// assert_eq!(r, expected_kring);
    /// ```
    pub fn k_ring(self, radius: u32) -> Vec<H3Index> {
        assert!(radius <= i32::MAX as _);
        unsafe {
            let max = libh3_sys::maxKringSize(radius as _) as _;
            let mut result = Vec::<H3Index>::with_capacity(max);
            libh3_sys::kRing(self.0, radius as _, result.as_mut_ptr() as _);
            result.set_len(max);
            result
        }
        .into_iter()
        .filter(|&v| v.0 != 0)
        .collect()
    }

    /// Get all hexagons in a k-ring around a given center, in an array of
    /// arrays ordered by distance from the origin. The order of the
    /// hexagons within each ring is undefined.
    ///
    /// # Arguments
    ///
    /// * `origin` - The center of the ring.
    /// * `radius` - The radis of the ring in hexagons, which is the same
    ///   resolution as the origin.
    ///
    /// ```
    /// # use libh3::H3Index;
    /// let expected_k_ring_distances = vec![
    ///     (0x8a2a1072b59ffff, 0),
    ///     (0x8a2a1072b597fff, 1),
    ///     (0x8a2a1070c96ffff, 1),
    ///     (0x8a2a1072b4b7fff, 1),
    ///     (0x8a2a1072b4a7fff, 1),
    ///     (0x8a2a1072b58ffff, 1),
    ///     (0x8a2a1072b587fff, 1),
    /// ]
    /// .iter()
    /// .map(|v| v.into())
    /// .collect();
    ///
    /// let r = H3Index::from(0x8a2a1072b59ffff).k_ring_distances(1);
    /// assert_eq!(r, expected_k_ring_distances);
    /// ```
    pub fn k_ring_distances(self, radius: u32) -> Vec<(H3Index, u32)> {
        assert!(radius <= i32::MAX as _);
        unsafe {
            let max = libh3_sys::maxKringSize(radius as _) as _;
            let mut indexes = Vec::<H3Index>::with_capacity(max);
            let mut distances = Vec::<u32>::with_capacity(max);

            libh3_sys::kRingDistances(
                self.0,
                radius as _,
                indexes.as_mut_ptr() as _,
                distances.as_mut_ptr() as _,
            );
            indexes.set_len(max);
            distances.set_len(max);

            indexes
                .into_iter()
                .zip(distances.into_iter())
                .filter(|&v| v.0 .0 != 0)
                .collect::<Vec<(H3Index, u32)>>()
        }
    }

    pub fn hex_range(self, k: u32) -> (bool, Vec<H3Index>) {
        unsafe {
            let max = libh3_sys::maxKringSize(k as _) as _;
            let mut r = Vec::<H3Index>::with_capacity(max);
            let distortion = libh3_sys::hexRange(self.0, k as _, r.as_mut_ptr() as _);
            r.set_len(max);
            r.retain(|v| v.0 != 0);
            (distortion == 0, r)
        }
    }

    pub fn hex_range_distances(self, k: u32) -> (bool, Vec<(H3Index, u32)>) {
        unsafe {
            let max = libh3_sys::maxKringSize(k as _) as _;
            let mut indexes = Vec::<H3Index>::with_capacity(max);
            let mut distances = Vec::<u32>::with_capacity(max);
            let distortion = libh3_sys::hexRangeDistances(
                self.0,
                k as _,
                indexes.as_mut_ptr() as _,
                distances.as_mut_ptr() as _,
            );
            indexes.set_len(max);
            distances.set_len(max);

            (
                distortion == 0,
                indexes
                    .into_iter()
                    .zip(distances.into_iter())
                    .filter(|v| v.0 .0 != 0)
                    .collect::<Vec<(H3Index, u32)>>(),
            )
        }
    }

    /// Get the grid distance between two hex indexes.
    ///
    /// This function may fail to find the distance between two indexes if they
    /// are very far apart or on opposite sides of a pentagon.
    ///
    /// # Arguments
    ///
    /// * `other` - The other index.
    ///
    /// ```
    /// assert_eq!(
    ///     H3Index::from(0x8a2a1072b4a7fff).dinstance(H3Index::from(0x8a2a1072b58ffff)),
    ///     Some(1)
    /// );
    /// ```
    pub fn distance(self, other: H3Index) -> Option<u32> {
        let r = unsafe { libh3_sys::h3Distance(self.0, other.0) };

        if r < 0 {
            None
        } else {
            Some(r as _)
        }
    }

    /// Returns the parent (coarser) index containing h3.
    ///
    /// # Arguments
    ///
    /// * `h` - The index of the child resolution.
    /// * `resolution` - The resolution of the desired level.
    ///
    /// ```
    /// use libh3::h3_to_parent;
    /// assert_eq!(h3_to_parent(0x8a2a1072b4a7fff, 5), 0x852a1073fffffff);
    /// ```
    pub fn parent(self, resolution: Resolution) -> H3Index {
        H3Index(unsafe { libh3_sys::h3ToParent(self.0, resolution.0 as _) })
    }

    /// Returns children indexes contained by the given index at the given
    /// resolution.
    ///
    /// # Arguments
    ///
    /// * `h` - The index of the child resolution.
    /// * `resolution` - The resolution of the desired level.
    ///
    /// ```
    /// use libh3::h3_to_children;
    /// assert_eq!(
    ///     h3_to_children(0x852a1073fffffff, 6),
    ///     vec![
    ///         0x862a10707ffffff,
    ///         0x862a1070fffffff,
    ///         0x862a10717ffffff,
    ///         0x862a1071fffffff,
    ///         0x862a10727ffffff,
    ///         0x862a1072fffffff,
    ///         0x862a10737ffffff
    ///     ]
    /// );
    /// ```
    pub fn children(self, resolution: Resolution) -> Vec<H3Index> {
        unsafe {
            let max = libh3_sys::maxH3ToChildrenSize(self.0, resolution.0 as _) as _;
            let mut result = Vec::<H3Index>::with_capacity(max);
            libh3_sys::h3ToChildren(self.0, resolution.0 as _, result.as_mut_ptr() as _);
            result.set_len(max);
            result
        }
    }
}
