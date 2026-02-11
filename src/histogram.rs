use std::{fmt::Debug, marker::PhantomData, sync::Arc};
use std::{collections::HashMap, hash::Hash};

/*
    Histogram class to record statistics
    Most values of interest are physical properties evolving over time
Therefore the unit stored in the histogram is the base unit multiplied by time.
    I.e. one electron spending one second in a certain state is equivalent to two electrons spending half a second each

    A histogram is parameterized by a binner, representing how to discretize the states.
    If there are multiple axis to make a histogram over, a Binner2D maps pairs of values with two binners
    For a Binner2D, a histogram slice over a specific value for either the first or second axis can be obtained

    If you want to split a histogram over many threads, the method `.get_worker()` gives an identical but blank histogram
    to send to the clone, and `.merge_worker()` joins the data from the now filled worker with the main histogram
    (A histogram forms a monoid with `.get_worker()` being the identity and `.merge_worker()` is the operator)
*/


#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinnerError {
    OutOfRangeBelow,
    OutOfRangeAbove,
    NotInSpec,
}

pub type Result<X> = std::result::Result<X, BinnerError>;


/// A binner implements mapping physical values into discrete bins
/// `bin` and `unbin` turns a physical quantity into an index and vice versa
/// Unbin must satisfy that `bin(unbin(idx)) == Some(idx)`
/// However, which value of `T` `unbin` selects is implementation-defined
pub trait Binner: Debug + Clone + Send {
    /// The physical quantity to map from
    // TODO: Ideally we'd want to be able to separate what values we consume in bin
    // (typically want to consume a reference) and what values to produce in unbin
    type T;

    fn bin(&self, value: Self::T) -> Result<usize>;
    fn unbin(&self, idx: usize) -> Result<Self::T>;

    /// Number of values this binner can bin into
    /// Used to know size of backing storage to allocate
    fn count(&self) -> usize;
}


pub mod units {
   #![allow(non_camel_case_types)]

    /// Type-level definitions of units
    pub trait Unit: Copy + Send + std::fmt::Debug {
        /// Latex-formattable
        const NAME: &'static str;
        /// Value of the unit in SI
        const IN_SI: f64;

        fn from_si(val: f64) -> f64 {
            val / Self::IN_SI
        }

        fn to_si(val: f64) -> f64 {
            val * Self::IN_SI
        }
    }

    use crate::consts::EV_TO_J;

    #[derive(Clone, Copy, Debug)]
    pub struct EV;
    impl Unit for EV {
        const NAME: &'static str = "eV";

        const IN_SI: f64 = EV_TO_J;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct MEV;
    impl Unit for MEV {
        const NAME: &'static str = "meV";

        const IN_SI: f64 = EV_TO_J * 1e-3;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct KV_PER_CM;
    impl Unit for KV_PER_CM {
        const NAME: &'static str = "kV/cm";

        const IN_SI: f64 = 1e5;
    }

    // why is this used so much in litterature?
    #[derive(Clone, Copy, Debug)]
    pub struct MILLION_CM_PER_SECOND;
    impl Unit for MILLION_CM_PER_SECOND {
        const NAME: &'static str = "$10^6$ cm/s";

        const IN_SI: f64 = 1e4;
    }

    #[derive(Clone, Copy, Debug)]
    pub struct PS;
    impl Unit for PS {
        const NAME: &'static str = "ps";

        const IN_SI: f64 = 1e-12;
    }
}
pub use units::Unit;


#[allow(unused)]
#[derive(Debug, Clone)]
/// Bins SI values in terms of a unit, start, end and count
/// Start and end are both inclusive. Note that values are rounded to nearest index, meaning the range of start and end extend a bit outside the proper range
pub struct UnitBinner<U: Unit> {
    _marker: PhantomData<U>,

    pub quantity_name: &'static str,
    pub start_si: f64,
    pub end_si: f64,
    pub count: usize,
}

impl<U: Unit> UnitBinner<U> {
    pub fn new_si(quantity_name: &'static str, start_si: f64, end_si: f64, count: usize) -> Self {
        UnitBinner {
            _marker: PhantomData,
            quantity_name,
            start_si,
            end_si,
            count,
        }
    }

    pub fn new(quantity_name: &'static str, start_unit: f64, end_unit: f64, count: usize) -> Self {
        UnitBinner {
            _marker: PhantomData,
            quantity_name,
            start_si: U::to_si(start_unit),
            end_si: U::to_si(end_unit),
            count,
        }
    }

    pub fn bin_size(&self) -> f64 {
        (self.end_si - self.start_si) / self.count as f64
    }

    pub fn bin_size_unit(&self) -> f64 {
        U::from_si(self.bin_size())
    }

    /// Convenience method for Self::U::to_si
    pub fn to_si(&self, val: f64) -> f64 {
        U::to_si(val)
    }

    /// Convenience method for Self::U::from_si
    pub fn from_si(&self, val: f64) -> f64 {
        U::from_si(val)
    }
}

impl<U: Unit + Debug + Clone> UnitBinner<U> {
    /// in SI
    pub fn steps(&self) -> impl Iterator<Item=f64> {
        (0..self.count).map(|idx| {
            self.unbin(idx).expect("Couldn't un-index value in range (should never happen)")
        })
    }
}

impl<U: Unit + Debug + Clone> Binner for UnitBinner<U> {
    type T = f64;

    fn bin(&self, value: f64) -> Result<usize> {
        let idx = (value - self.start_si) / (self.end_si - self.start_si) * (self.count-1) as f64;
        let idx = idx.round();

        if idx < 0. {
            return Err(BinnerError::OutOfRangeBelow);
        }
        if idx >= self.count as f64 {
            return Err(BinnerError::OutOfRangeAbove);
        }
        Ok(idx as usize)
    }

    fn unbin(&self, idx: usize) -> Result<Self::T> {
        // TODO: Check if idx is in range?
        let val = (idx as f64 / (self.count-1) as f64) * (self.end_si - self.start_si) + self.start_si;
        Ok(val)
    }

    fn count(&self) -> usize {
        self.count
    }
}

// TODO: Make this parametric over key type
#[derive(Clone, Debug)]
pub struct DiscreteBinner<K: Hash + Eq> {
    indices: HashMap<K, usize>,
    count: usize,
}

impl<K: Hash + Eq> DiscreteBinner<K> {
    pub fn new(items: Vec<K>) -> DiscreteBinner<K> {
        let count = items.len();
        let indices = items.into_iter().enumerate().map(|(idx, item)| (item, idx)).collect();
        DiscreteBinner {
            indices,
            count,
        }
    }
}

impl<K: Hash + Eq + Clone + Debug + Send> Binner for DiscreteBinner<K> {
    type T = K;

    // TODO: Kinda awkward interface. Ideally would want to get a &str here, not a &String
    // Setting T = str would make unbin hard to implement...
    fn bin(&self, value: K) -> Result<usize> {
        self.indices.get(&value).cloned().ok_or(BinnerError::NotInSpec)
    }

    fn unbin(&self, idx: usize) -> Result<K> {
        self.indices.iter()
            .find_map(|(item, &item_idx)| (idx == item_idx).then(|| item))
            .cloned()
            .ok_or(BinnerError::NotInSpec)

    }

    fn count(&self) -> usize {
        self.count
    }
}

impl<K: Hash + Eq + Clone + Debug + Send> DiscreteBinner<K> {
    pub fn steps(&self) -> impl Iterator<Item=K> {
        (0..self.count())
            .map(|idx| self.unbin(idx).expect("Couldn't unbin value in range (this should not happen)"))
    }
}

/// Bins into major_idx * minor_count + minor_idx
#[derive(Debug, Clone)]
pub struct Binner2D<Major: Binner, Minor: Binner> {
    pub major: Major,
    pub minor: Minor,
}

impl<Major: Binner, Minor: Binner> Binner for Binner2D<Major, Minor> {
    type T = (Major::T, Minor::T);

    fn bin(&self, (major_value, minor_value): (Major::T, Minor::T)) -> Result<usize> {
        let (major_idx, minor_idx) = (self.major.bin(major_value)?, self.minor.bin(minor_value)?);
        let minor_count = self.minor.count();

        Ok(major_idx * minor_count + minor_idx)
    }

    fn unbin(&self, idx: usize) -> Result<Self::T> {
        let minor_count = self.minor.count();
        let (major_idx, minor_idx) = (idx / minor_count, idx % minor_count);
        let (major_value, minor_value) = (self.major.unbin(major_idx)?, self.minor.unbin(minor_idx)?);
        Ok((major_value, minor_value))
    }

    fn count(&self) -> usize {
        self.major.count() * self.minor.count()
    }
}

pub struct Histogram<B: Binner> {
    pub name: Arc<str>,
    pub binner: B,

    // Invariant: `storage.len() == binner.count()`
    storage: Vec<f64>,
    // All values `add`ed to the histogram
    pub total: f64,

    // TODO: record out-of-bounds writes
    // TODO: Represent adds and gets with ranges
}

impl<B: Binner> Histogram<B> {
    pub fn new(name: String, binner: B) -> Self {
        let count = binner.count();
        Histogram {
            name: name.into(),
            binner,
            storage: std::iter::repeat(0.).take(count).collect(),
            total: 0.,
        }
    }

    pub fn get_worker(&self) -> Self {
        let count = self.storage.len();
        Histogram {
            name: self.name.clone(),
            binner: self.binner.clone(),
            storage: std::iter::repeat(0.).take(count).collect(),
            total: 0.,
        }
    }

    pub fn merge_worker(&mut self, worker: Self) {
        // Sanity check: is the name of the worker the same?
        assert_eq!(self.name, worker.name, "Worker must come from self");

        for (x, y) in self.storage.iter_mut().zip(worker.storage) {
            *x += y;
        }
        self.total += worker.total;
    }

    pub fn as_ref<'h>(&'h self) -> HistogramRef<'h, B> {
        HistogramRef {
            binner: &self.binner,
            storage: &self.storage,
            stride: 1,
            total: &self.total,
        }
    }

    pub fn as_ref_mut<'h>(&'h mut self) -> HistogramRefMut<'h, B> {
        HistogramRefMut {
            binner: &self.binner,
            storage: &mut self.storage,
            stride: 1,
            total: &mut self.total,
        }
    }

    // Wrapper methods around HistogramRef(Mut)
    pub fn get(&self, at: B::T) -> f64 {
        self.as_ref().get(at)
    }

    // (collected into a Vec to avoid lifetime issues)
    pub fn all_values<'s>(&'s self) -> Vec<(B::T, f64)> {
        self.as_ref().all_values().collect()
    }

    pub fn add(&mut self, at: B::T, amount: f64) {
        self.as_ref_mut().add(at, amount);
    }
}

impl<Major: Binner, Minor: Binner> Histogram<Binner2D<Major, Minor>> {
    pub fn as_ref_at_major<'h>(&'h self, major_value: Major::T) -> Result<HistogramRef<'h, Minor>> {
        let major_idx = self.binner.major.bin(major_value)?;
        let minor_count = self.binner.minor.count();

        Ok(HistogramRef {
            binner: &self.binner.minor,
            storage: &self.storage[major_idx * minor_count..(major_idx+1) * minor_count],
            stride: 1,
            // TODO: this is wrong... how do we do this...
            total: &self.total,
        })
    }

    pub fn as_ref_at_minor<'h>(&'h self, minor_value: Minor::T) -> Result<HistogramRef<'h, Major>> {
        let minor_idx = self.binner.minor.bin(minor_value)?;
        let minor_count = self.binner.minor.count();

        Ok(HistogramRef {
            binner: &self.binner.major,
            storage: &self.storage[minor_idx..],
            stride: minor_count,
            // TODO: This is wrong too...
            total: &self.total,
        })
    }

    pub fn as_ref_mut_at_major<'h>(&'h mut self, major_value: Major::T) -> Result<HistogramRefMut<'h, Minor>> {
        let major_idx = self.binner.major.bin(major_value)?;
        let minor_count = self.binner.minor.count();
        Ok(HistogramRefMut {
            binner: &self.binner.minor,
            storage: &mut self.storage[major_idx * minor_count..(major_idx+1) * minor_count],
            stride: 1,
            total: &mut self.total,
        })
    }

    pub fn as_ref_mut_at_minor<'h>(&'h mut self, minor_value: Minor::T) -> Result<HistogramRefMut<'h, Major>> {
        let minor_idx = self.binner.minor.bin(minor_value)?;
        let minor_count = self.binner.minor.count();

        Ok(HistogramRefMut {
            binner: &self.binner.major,
            storage: &mut self.storage[minor_idx..],
            stride: minor_count,
            total: &mut self.total,
        })
    }
}

// TODO: The histogramrefmut etc interface kinda sucks
// we need explicit conversion methods everywhere and it's just bad and yucky

pub struct HistogramRef<'h, B: Binner> {
    pub binner: &'h B,
    storage: &'h [f64],
    pub total: &'h f64,
    stride: usize,
}

impl<'h, B: Binner> HistogramRef<'h, B> {
    pub fn get(&self, at: B::T) -> f64 {
        let Ok(idx) = self.binner.bin(at) else {
            return 0.;
        };
        self.storage[self.stride * idx]
    }

    pub fn all_values<'l>(&'l self) -> impl Iterator<Item=(B::T, f64)> + 'l {
        (0..self.binner.count())
            .map(|idx| (
                self.binner.unbin(idx).expect("Unbin failed on a value 0 <= _ < count (this should never happen)"),
                self.storage[self.stride * idx],
            ))
    }

    /// Total time for this slice
    /// TODO: Check this
    pub fn subtotal(&self) -> f64 {
        self.storage.iter().sum()
    }
}

impl<'h, U: Unit + Debug> HistogramRef<'h, UnitBinner<U>> {
    pub fn mean(&self) -> f64 {
        let integral = self.storage.iter().enumerate()
            .map(|(idx, amount)| {
                let value = self.binner.unbin(idx).expect("Failed to unbin value in range");
                value * amount
            })
            .sum::<f64>();

        integral / self.subtotal()
    }

    pub fn stddev(&self) -> f64 {
        let mean = self.mean();

        let integral = self.storage.iter().enumerate()
            .map(|(idx, amount)| {
                let value = self.binner.unbin(idx).expect("Failed to unbin value in range");
                (value-mean).powi(2) * amount
            })
            .sum::<f64>();

        (integral / self.subtotal()).sqrt()
    }
}

pub struct HistogramRefMut<'h, B: Binner> {
    pub binner: &'h B,
    storage: &'h mut [f64],
    stride: usize,
    pub total: &'h mut f64,
}

impl<'h, B: Binner> HistogramRefMut<'h, B> {
    pub fn add(&mut self, at: B::T, amount: f64) {
        *self.total += amount;
        let Ok(idx) = self.binner.bin(at) else {
            // TODO: Record OOB?
            return;
        };
        self.storage[self.stride * idx] += amount;
    }

    pub fn as_ref<'s: 'h>(&'s self) -> HistogramRef<'h, B> {
        HistogramRef {
            binner: &self.binner,
            storage: &self.storage,
            stride: self.stride,
            total: &*self.total,
        }
    }

    // Wrapper methods around HistogramRef
    pub fn get(&self, at: B::T) -> f64 {
        self.as_ref().get(at)
    }

    // (collected into a Vec to avoid lifetime issues)
    pub fn all_values<'s>(&'s self) -> Vec<(B::T, f64)> {
        self.as_ref().all_values().collect()
    }

    pub fn subtotal(&self) -> f64 {
        self.as_ref().subtotal()
    }
    // TODO: mean method
}

#[macro_export]
macro_rules! generate_histogram_collection_struct {
    { $(struct $name:ident { $($field:ident: $ty:ty),* $(,)? })* } => {
        $(
            struct $name {
                $($field: $ty,)+
            }
            impl $name {
                #[allow(dead_code)]
                fn get_worker(&self) -> $name {
                    $name {
                        $($field: self.$field.get_worker(),)*
                    }
                }

                #[allow(dead_code)]
                fn merge_worker(&mut self, worker: $name) {
                    $(self.$field.merge_worker(worker.$field);)*
                }
            }
        )*
    };
}

pub use generate_histogram_collection_struct;

#[cfg(test)]
mod tests {
    pub use super::*;

    fn pred_is_sync<T: Sync>() {}
    // This will fail to compile if Histogram ever turns !Sync
    #[allow(unused)]
    fn histo_is_sync<B: Binner + Sync>() {
        pred_is_sync::<Histogram<B>>();
    }


    macro_rules! assert_eq_float {
        ($lhs:expr, $rhs:expr) => {
            let lhs: f64 = $lhs;
            let rhs: f64 = $rhs;
            assert!((lhs - rhs).abs() / lhs.abs().max(rhs).max(1e-30).abs() < 1e-5, "lhs = {lhs:?}, rhs = {rhs:?}");
        };
    }

    macro_rules! assert_eq_option_float_tuple {
        ($a:expr, $b:expr) => {
            let a: Option<(f64, f64)> = $a;
            let b: Option<(f64, f64)> = $b;
            match (a, b) {
                (Some(lhs), Some(rhs)) => {
                    assert_eq_float!(lhs.0, rhs.0);
                    assert_eq_float!(lhs.1, rhs.1);
                }
                (a, b) => assert_eq!(a, b),
            }
        };
    }

    #[allow(unused)]
    macro_rules! assert_eq_option_float {
        ($a:expr, $b:expr) => {
            let a: Option<f64> = $a;
            let b: Option<f64> = $b;
            match (a, b) {
                (Some(lhs), Some(rhs)) => {
                    assert_eq_float!(lhs, rhs);
                }
                (a, b) => assert_eq!(a, b),
            }
        };
    }

    #[allow(unused)]
    macro_rules! assert_eq_result_float {
        ($a:expr, $b:expr) => {
            let a: Result<f64> = $a;
            let b: Result<f64> = $b;
            match (a, b) {
                (Ok(lhs), Ok(rhs)) => {
                    assert_eq_float!(lhs, rhs);
                }
                (a, b) => assert_eq!(a, b),
            }
        };
    }

    #[test]
    fn test_unit_binner() {
        use units::Unit;

        let binner: UnitBinner<units::EV> = UnitBinner::new("E", 5., 10., 6);
        let mut steps = binner.steps();

        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(5.)));
        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(6.)));
        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(7.)));
        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(8.)));
        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(9.)));
        assert_eq_option_float!(steps.next(), Some(units::EV::to_si(10.)));
        assert_eq_option_float!(steps.next(), None);

        assert_eq!(binner.bin(units::EV::to_si(5.0)), Ok(0));
        assert_eq!(binner.bin(units::EV::to_si(4.6)), Ok(0));
        assert_eq!(binner.bin(units::EV::to_si(4.4)), Err(BinnerError::OutOfRangeBelow));
        assert_eq!(binner.bin(units::EV::to_si(7.0)), Ok(2));

        assert_eq!(binner.bin(units::EV::to_si(10.4)), Ok(5));
        assert_eq!(binner.bin(units::EV::to_si(10.6)), Err(BinnerError::OutOfRangeAbove));

        // this is kinda undefined, could be any value between 6.5 and 7.5
        assert_eq_result_float!(binner.unbin(2), Ok(units::EV::to_si(7.0)));
    }

    #[test]
    fn test_discrete_binner() {
        let binner = DiscreteBinner::new(vec!["first", "second", "third"]);
        assert_eq!(binner.bin("first"), Ok(0));
        assert_eq!(binner.bin("third"), Ok(2));
        let mut iter = binner.steps();
        assert_eq!(iter.next(), Some("first"));
        assert_eq!(iter.next(), Some("second"));
        assert_eq!(iter.next(), Some("third"));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_binner2d() {
        let major: UnitBinner<units::KV_PER_CM> = UnitBinner::new("E_x", 0., 5., 6);
        let minor: UnitBinner<units::MILLION_CM_PER_SECOND> = UnitBinner::new("v", -20., 20., 3);
        let binner = Binner2D { major, minor };

        assert_eq!(binner.count(), 18);

        assert_eq!(binner.bin((binner.major.to_si(0.), binner.minor.to_si(-20.))), Ok(0));
        assert_eq!(binner.bin((binner.major.to_si(3.), binner.minor.to_si(0.))), Ok(10));

        assert_eq_option_float_tuple!(binner.unbin(10).ok(), Some((binner.major.to_si(3.), binner.minor.to_si(0.))));
    }

    #[test]
    fn test_histogram() {
        let binner: UnitBinner<units::EV> = UnitBinner::new("E", 5., 10., 6);
        let mut histo = Histogram::new("Test histo 1D".into(), binner);

        histo.add(histo.binner.to_si(6.), 2.);
        histo.add(histo.binner.to_si(7.), 1.);
        // some out of range items
        histo.add(histo.binner.to_si(4.), 3.);
        histo.add(histo.binner.to_si(11.), 7.);

        assert_eq_float!(histo.get(histo.binner.to_si(6.)), 2.);

        let mut items = histo.all_values().into_iter();
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(5.), 0.)));
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(6.), 2.)));
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(7.), 1.)));
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(8.), 0.)));
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(9.), 0.)));
        assert_eq_option_float_tuple!(items.next(), Some((histo.binner.to_si(10.), 0.)));
        assert_eq_option_float_tuple!(items.next(), None);
    }

    #[test]
    fn test_histogram_2d() {
        let major: UnitBinner<units::KV_PER_CM> = UnitBinner::new("E_x", 0., 5., 6);
        let minor: UnitBinner<units::MILLION_CM_PER_SECOND> = UnitBinner::new("v", -20., 20., 3);
        let binner = Binner2D { major: major.clone(), minor: minor.clone(), };

        let mut histo = Histogram::new("Test histo 2D".into(), binner);

        histo.add((major.to_si(1.), minor.to_si(20.)), 6.);
        histo.add((major.to_si(3.), minor.to_si(-20.)), 7.);

        assert_eq_float!(histo.get((major.to_si(1.), minor.to_si(20.))), 6.);
        assert_eq_float!(histo.get((major.to_si(2.), minor.to_si(20.))), 0.);
        assert_eq_float!(histo.get((major.to_si(3.), minor.to_si(-20.))), 7.);
        // out of range values
        assert_eq_float!(histo.get((major.to_si(2.), minor.to_si(40.))), 0.);
    }

    #[test]
    fn test_histogram_2d_slices() {
        let major: UnitBinner<units::KV_PER_CM> = UnitBinner::new("E_x", 0., 5., 6);
        let minor: UnitBinner<units::MILLION_CM_PER_SECOND> = UnitBinner::new("v", -20., 20., 3);
        let binner = Binner2D { major: major.clone(), minor: minor.clone(), };

        let mut histo = Histogram::new("Test histo 2D slicing".into(), binner);

        // slice at a major
        {
            let Ok(mut at_one_ev) = histo.as_ref_mut_at_major(major.to_si(1.)) else {
                assert!(false, "histo.as_ref_mut_at_major faild at valid value");
                return;
            };
            // add some stuff
            at_one_ev.add(minor.to_si(-20.), 6.);
            at_one_ev.add(minor.to_si(20.), 7.);

            // some out of range stuff that could overflow and interfere
            at_one_ev.add(minor.to_si(20. + 40. * 3.), 1.);
        }
        // slice at a minor
        {
            let Ok(mut at_twenty_cm) = histo.as_ref_mut_at_minor(minor.to_si(20.)) else {
                assert!(false, "histo.as_ref_mut_at_major faild at valid value");
                return;
            };
            // add some stuff
            at_twenty_cm.add(major.to_si(3.), 6.);
            at_twenty_cm.add(major.to_si(1.), 7.); // overlaps with the othe 7!!
        }

        // check values
        assert_eq_float!(histo.get((major.to_si(1.), minor.to_si(-20.))), 6.);
        assert_eq_float!(histo.get((major.to_si(3.), minor.to_si(20.))), 6.);
        assert_eq_float!(histo.get((major.to_si(1.), minor.to_si(20.))), 14.);
    }

    #[test]
    fn test_histogram_workers() {
        let binner: UnitBinner<units::EV> = UnitBinner::new("E", 0., 5., 6);
        let mut histo = Histogram::new("Test histo workers".into(), binner.clone());

        let mut worker_1 = histo.get_worker();
        let mut worker_2 = histo.get_worker();
        let mut worker_3 = histo.get_worker();

        // Overlapping values
        worker_1.add(binner.to_si(1.), 2.);
        worker_2.add(binner.to_si(1.), 1.);
        worker_3.add(binner.to_si(2.), 7.);

        histo.merge_worker(worker_1);
        histo.merge_worker(worker_2);
        histo.merge_worker(worker_3);

        assert_eq_float!(histo.get(binner.to_si(1.)), 3.);
        assert_eq_float!(histo.get(binner.to_si(2.)), 7.);
        assert_eq_float!(histo.get(binner.to_si(3.)), 0.);

    }

    // Test generate_histogram_collection_struct
    mod autostructs {
        use super::*;

        // TODO: We should really try generating multiple histogram structs
        generate_histogram_collection_struct! {
            struct Histograms {
                histo_1: Histogram<DiscreteBinner<usize>>,
                histo_2: Histogram<UnitBinner<units::EV>>,
            }
        }

        #[test]
        fn test_generated_thingy() {
            let mut h = Histograms {
                histo_1: Histogram::new("awa".to_string(), DiscreteBinner::new(vec![1, 2, 3])),
                histo_2: Histogram::new("bwa".to_string(), UnitBinner::new("E", 0., 1., 10)),
            };
            // add some stuff to h
            h.histo_1.add(1, 6.);
            h.histo_2.add(units::EV::to_si(0.7), 10.);

            // add some stuff through workers
            let mut worker1 = h.get_worker();
            let mut worker2 = h.get_worker();
            worker1.histo_1.add(1, 7.);
            worker1.histo_1.add(2, 8.);
            worker1.histo_2.add(units::EV::to_si(0.5), 10.);

            worker2.histo_1.add(1, 6.);
            worker2.histo_1.add(3, 7.);
            worker1.histo_2.add(units::EV::to_si(0.5), 10.);

            // merge everything
            h.merge_worker(worker1);
            h.merge_worker(worker2);

            // check
            assert_eq_float!(h.histo_1.get(1), 19.);
            assert_eq_float!(h.histo_1.get(2), 8.);
            assert_eq_float!(h.histo_1.get(3), 7.);

            assert_eq_float!(h.histo_2.get(units::EV::to_si(0.5)), 20.);
            assert_eq_float!(h.histo_2.get(units::EV::to_si(0.7)), 10.);
            assert_eq_float!(h.histo_2.get(units::EV::to_si(0.9)), 0.);
        }
    }
}

