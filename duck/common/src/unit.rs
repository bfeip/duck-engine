//! Real-world length units, and the scale relating them to world space.
//!
//! Three distinct ideas live here:
//!
//! - [`LengthUnit`] is a *named* unit — millimeter, inch — and knows its size
//!   in meters.
//! - [`WorldUnits`] is a *physical fact* about a coordinate space: how many
//!   meters one world unit spans. A scene carries one.
//! - [`LengthDisplay`] is a *presentation preference*: what a person reads and
//!   types. It converts between world space and a chosen unit.
//!
//! World space is meters by default ([`WorldUnits::METER`]).

/// A named unit of length, defined by its size in meters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LengthUnit {
    Nanometer,
    Micrometer,
    Millimeter,
    Centimeter,
    Meter,
    Kilometer,
    Inch,
    Foot,
    Yard,
    Mile,
}

impl LengthUnit {
    /// Every unit, in ascending order of size. Metric first, then imperial.
    pub const ALL: [LengthUnit; 10] = [
        LengthUnit::Nanometer,
        LengthUnit::Micrometer,
        LengthUnit::Millimeter,
        LengthUnit::Centimeter,
        LengthUnit::Meter,
        LengthUnit::Kilometer,
        LengthUnit::Inch,
        LengthUnit::Foot,
        LengthUnit::Yard,
        LengthUnit::Mile,
    ];

    /// How many meters one of this unit spans. Imperial values are exact by
    /// definition (one inch is exactly 25.4 mm).
    pub const fn meters(self) -> f64 {
        match self {
            LengthUnit::Nanometer => 1e-9,
            LengthUnit::Micrometer => 1e-6,
            LengthUnit::Millimeter => 1e-3,
            LengthUnit::Centimeter => 1e-2,
            LengthUnit::Meter => 1.0,
            LengthUnit::Kilometer => 1e3,
            LengthUnit::Inch => 0.0254,
            LengthUnit::Foot => 0.3048,
            LengthUnit::Yard => 0.9144,
            LengthUnit::Mile => 1609.344,
        }
    }

    /// Short suffix used when displaying a value.
    pub const fn suffix(self) -> &'static str {
        match self {
            LengthUnit::Nanometer => "nm",
            LengthUnit::Micrometer => "µm",
            LengthUnit::Millimeter => "mm",
            LengthUnit::Centimeter => "cm",
            LengthUnit::Meter => "m",
            LengthUnit::Kilometer => "km",
            LengthUnit::Inch => "in",
            LengthUnit::Foot => "ft",
            LengthUnit::Yard => "yd",
            LengthUnit::Mile => "mi",
        }
    }

    /// Full name, for menus and labels.
    pub const fn name(self) -> &'static str {
        match self {
            LengthUnit::Nanometer => "Nanometer",
            LengthUnit::Micrometer => "Micrometer",
            LengthUnit::Millimeter => "Millimeter",
            LengthUnit::Centimeter => "Centimeter",
            LengthUnit::Meter => "Meter",
            LengthUnit::Kilometer => "Kilometer",
            LengthUnit::Inch => "Inch",
            LengthUnit::Foot => "Foot",
            LengthUnit::Yard => "Yard",
            LengthUnit::Mile => "Mile",
        }
    }

    /// The unit written as `suffix`, if any. Case-insensitive, and `"um"` is
    /// accepted for micrometers alongside `"µm"`.
    pub fn from_suffix(suffix: &str) -> Option<Self> {
        let suffix = suffix.trim();
        if suffix.eq_ignore_ascii_case("um") {
            return Some(LengthUnit::Micrometer);
        }
        LengthUnit::ALL.into_iter().find(|u| u.suffix().eq_ignore_ascii_case(suffix))
    }

    /// Converts `value` from this unit into `target`.
    pub fn convert(self, value: f64, target: LengthUnit) -> f64 {
        value * (self.meters() / target.meters())
    }
}

/// How large one world unit is, in real-world terms.
///
/// Stored as meters per world unit rather than a [`LengthUnit`] because some
/// formats declare an arbitrary factor that matches no named unit — USD's
/// `metersPerUnit` among them.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldUnits {
    meters_per_unit: f64,
}

impl WorldUnits {
    /// One world unit is one meter. The engine default.
    pub const METER: Self = Self { meters_per_unit: 1.0 };
    /// One world unit is one millimeter.
    pub const MILLIMETER: Self = Self { meters_per_unit: 1e-3 };

    /// One world unit is one `unit`.
    pub const fn from_unit(unit: LengthUnit) -> Self {
        Self { meters_per_unit: unit.meters() }
    }

    /// Builds a scale from a raw factor, rejecting anything non-finite or not
    /// strictly positive.
    pub fn from_meters_per_unit(meters_per_unit: f64) -> Option<Self> {
        (meters_per_unit.is_finite() && meters_per_unit > 0.0).then_some(Self { meters_per_unit })
    }

    /// Meters spanned by one world unit.
    pub const fn meters_per_unit(self) -> f64 {
        self.meters_per_unit
    }

    /// The named unit this scale represents, if it matches one to within
    /// floating-point tolerance.
    pub fn nearest_unit(self) -> Option<LengthUnit> {
        LengthUnit::ALL.into_iter().find(|u| {
            let m = u.meters();
            (m - self.meters_per_unit).abs() <= m * 1e-9
        })
    }

    /// Factor converting a length expressed in `self` into one expressed in
    /// `other`.
    pub const fn factor_to(self, other: WorldUnits) -> f64 {
        self.meters_per_unit / other.meters_per_unit
    }

    /// Converts a world-space length into `unit`.
    pub fn to_unit(self, world_value: f64, unit: LengthUnit) -> f64 {
        world_value * (self.meters_per_unit / unit.meters())
    }

    /// Converts a length in `unit` into world space.
    pub fn from_unit_value(self, value: f64, unit: LengthUnit) -> f64 {
        value * (unit.meters() / self.meters_per_unit)
    }
}

impl Default for WorldUnits {
    fn default() -> Self {
        Self::METER
    }
}

/// How lengths are shown to and accepted from a person: a unit and a decimal
/// precision.
///
/// This is a presentation preference, deliberately separate from the
/// [`WorldUnits`] a scene is authored in — one scene can be read in
/// millimeters or inches without being modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LengthDisplay {
    pub unit: LengthUnit,
    pub decimals: usize,
}

impl LengthDisplay {
    pub const fn new(unit: LengthUnit, decimals: usize) -> Self {
        Self { unit, decimals }
    }

    /// Formats a world-space length, with the unit suffix.
    pub fn format(&self, world_value: f64, world: WorldUnits) -> String {
        format!(
            "{:.*} {}",
            self.decimals,
            world.to_unit(world_value, self.unit),
            self.unit.suffix()
        )
    }

    /// Formats a world-space length without the unit suffix, for widgets that
    /// show the suffix themselves.
    pub fn format_bare(&self, world_value: f64, world: WorldUnits) -> String {
        format!("{:.*}", self.decimals, world.to_unit(world_value, self.unit))
    }

    /// Parses text into a world-space length. A unit suffix in the text wins;
    /// a bare number is read as [`Self::unit`].
    pub fn parse(&self, text: &str, world: WorldUnits) -> Option<f64> {
        let text = text.trim();
        // Split the numeric head from a trailing suffix.
        let split = text
            .char_indices()
            .rfind(|(_, c)| c.is_ascii_digit() || *c == '.')
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        let (number, suffix) = text.split_at(split);
        let value: f64 = number.trim().parse().ok()?;
        let unit = if suffix.trim().is_empty() {
            self.unit
        } else {
            LengthUnit::from_suffix(suffix)?
        };
        Some(world.from_unit_value(value, unit))
    }

    /// A sensible drag increment for this display, in world units: one step of
    /// the last shown decimal place, scaled up so coarse displays still move
    /// usefully.
    pub fn step(&self, world: WorldUnits) -> f64 {
        let in_unit = 10f64.powi(-(self.decimals as i32)) * 10.0;
        world.from_unit_value(in_unit, self.unit)
    }
}

impl Default for LengthDisplay {
    fn default() -> Self {
        Self { unit: LengthUnit::Meter, decimals: 3 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-9;

    #[test]
    fn imperial_factors_are_exact() {
        assert_eq!(LengthUnit::Inch.meters(), 0.0254);
        assert_eq!(LengthUnit::Foot.meters(), 0.3048);
        // A foot is exactly twelve inches.
        assert!((LengthUnit::Foot.meters() - 12.0 * LengthUnit::Inch.meters()).abs() < EPSILON);
    }

    #[test]
    fn convert_round_trips() {
        for from in LengthUnit::ALL {
            for to in LengthUnit::ALL {
                let there = from.convert(2.5, to);
                let back = to.convert(there, from);
                assert!((back - 2.5).abs() < 2.5 * 1e-12, "{from:?} -> {to:?} -> {back}");
            }
        }
    }

    #[test]
    fn convert_known_values() {
        assert!((LengthUnit::Meter.convert(1.0, LengthUnit::Millimeter) - 1000.0).abs() < EPSILON);
        assert!((LengthUnit::Inch.convert(1.0, LengthUnit::Millimeter) - 25.4).abs() < EPSILON);
    }

    #[test]
    fn suffix_round_trips() {
        for unit in LengthUnit::ALL {
            assert_eq!(LengthUnit::from_suffix(unit.suffix()), Some(unit));
        }
        assert_eq!(LengthUnit::from_suffix("MM"), Some(LengthUnit::Millimeter));
        assert_eq!(LengthUnit::from_suffix("um"), Some(LengthUnit::Micrometer));
        assert_eq!(LengthUnit::from_suffix("furlong"), None);
    }

    #[test]
    fn world_units_rejects_degenerate_factors() {
        assert!(WorldUnits::from_meters_per_unit(0.0).is_none());
        assert!(WorldUnits::from_meters_per_unit(-1.0).is_none());
        assert!(WorldUnits::from_meters_per_unit(f64::NAN).is_none());
        assert!(WorldUnits::from_meters_per_unit(f64::INFINITY).is_none());
        assert!(WorldUnits::from_meters_per_unit(0.01).is_some());
    }

    #[test]
    fn default_world_is_meters() {
        assert_eq!(WorldUnits::default(), WorldUnits::METER);
        assert_eq!(WorldUnits::METER.nearest_unit(), Some(LengthUnit::Meter));
        assert_eq!(WorldUnits::MILLIMETER.nearest_unit(), Some(LengthUnit::Millimeter));
    }

    #[test]
    fn nearest_unit_is_none_for_arbitrary_scale() {
        // USD stages may declare a factor matching no named unit.
        let odd = WorldUnits::from_meters_per_unit(0.0142857).unwrap();
        assert_eq!(odd.nearest_unit(), None);
    }

    #[test]
    fn factor_between_scales() {
        // One millimeter-unit is 0.001 meter-units.
        let f = WorldUnits::MILLIMETER.factor_to(WorldUnits::METER);
        assert!((f - 0.001).abs() < EPSILON);
    }

    #[test]
    fn meter_world_reads_as_millimeters() {
        // 0.0125 world meters is 12.5 mm.
        let d = LengthDisplay::new(LengthUnit::Millimeter, 2);
        assert_eq!(d.format(0.0125, WorldUnits::METER), "12.50 mm");
    }

    #[test]
    fn parse_uses_display_unit_for_bare_numbers() {
        let d = LengthDisplay::new(LengthUnit::Millimeter, 2);
        let world = d.parse("12.5", WorldUnits::METER).unwrap();
        assert!((world - 0.0125).abs() < EPSILON);
    }

    #[test]
    fn parse_honors_an_explicit_suffix() {
        let d = LengthDisplay::new(LengthUnit::Millimeter, 2);
        // An inch is 0.0254 m regardless of the display unit.
        let world = d.parse("1 in", WorldUnits::METER).unwrap();
        assert!((world - 0.0254).abs() < EPSILON);
    }

    #[test]
    fn format_parse_round_trips() {
        let d = LengthDisplay::new(LengthUnit::Millimeter, 4);
        let world = 0.012_345_6;
        let back = d.parse(&d.format(world, WorldUnits::METER), WorldUnits::METER).unwrap();
        assert!((back - world).abs() < 1e-7);
    }

    #[test]
    fn parse_rejects_nonsense() {
        let d = LengthDisplay::default();
        assert!(d.parse("", WorldUnits::METER).is_none());
        assert!(d.parse("abc", WorldUnits::METER).is_none());
        assert!(d.parse("12 furlongs", WorldUnits::METER).is_none());
    }

    #[test]
    fn step_is_in_world_units() {
        // Two decimals of mm -> 0.1 mm -> 1e-4 m.
        let d = LengthDisplay::new(LengthUnit::Millimeter, 2);
        assert!((d.step(WorldUnits::METER) - 1e-4).abs() < 1e-12);
    }
}
