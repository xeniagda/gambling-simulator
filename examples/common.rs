use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use gambling_simulator::consts::EV_TO_J;
use plotly::Plot;
use build_html::{Html, HtmlContainer, HtmlPage};

#[allow(unused)]
pub const VALLEY_COLORS: [&str; 3] = ["black", "blue", "green"];

#[allow(dead_code)]
pub fn linspace(a: f64, b: f64, steps: u64) -> impl Iterator<Item = f64> {
    (0..steps).map(move |x| (x as f64) / (steps as f64) * (b - a) + a)
}

#[allow(dead_code)]
pub fn linspace_incl(a: f64, b: f64, steps: u64) -> impl Iterator<Item = f64> {
    (0..steps).map(move |x| (x as f64) / ((steps-1) as f64) * (b - a) + a)
}

#[allow(dead_code)]
pub fn write_plots(
    cat: impl AsRef<str>,
    name: impl AsRef<str>,
    plots: impl IntoIterator<Item=Plot>,
) {
    let (cat, name) = (cat.as_ref(), name.as_ref());
    let mut html = HtmlPage::new()
        .with_title(name)
        .with_header(1, name)
        .with_raw(Plot::offline_js_sources())
        .with_head_link("https://images.emojiterra.com/google/noto-emoji/unicode-16.0/color/svg/303d.svg", "icon")
        .with_script_link("https://cdn.plot.ly/plotly-3.3.1.min.js");
    for plot in plots {
        html = html.with_raw(plot.to_inline_html(None))
    }
    let text = html.to_html_string();

    let mut path = std::env::current_dir().expect("Couldn't get pwd");
    path.push("plots");
    path.push(cat);
    path.push(format!("{name}.html"));

    if let Err(e) = std::fs::write(&path, text) {
        eprintln!("Couldn't write {cat}/{name}: {e:?}");
    } else {
        eprintln!("Wrote file://{}", path.display());
    }
}

#[allow(unused)]
pub enum XAxisStyle {
    Discrete(Vec<String>),
    Continuous { label: Option<String>, },
}

pub trait Binner: Debug + Clone + Send {
    type T;

    #[allow(unused)]
    fn count(&self) -> usize;

    #[allow(unused)]
    fn bin(&self, value: Self::T) -> isize; // negative = out of range (too small), >= self.count() = out of range (too large)

    #[allow(unused)]
    fn unbin(&self, idx: usize) -> Self::T;

    #[allow(unused)]
    fn x_axis_style(&self) -> XAxisStyle;

    #[allow(unused)]
    fn items(&self) -> Vec<Self::T>;
}

pub type Unit = (&'static str, f64); // name, unit value in SI
#[allow(unused)]
pub const UNIT_EV: Unit = ("eV", EV_TO_J);
#[allow(unused)]
pub const UNIT_KV_CM: Unit = ("kV/cm", 1e5);
#[allow(unused)]
pub const UNIT_MEV: Unit = ("meV", EV_TO_J * 1e-3);
#[allow(unused)]
pub const UNIT_10_7_CM_S: Unit = ("$10^7$ cm/s", 1e5);

#[allow(unused)]
#[derive(Debug, Clone)]
/// Bins SI values in terms of a unit, start, end and count
pub struct UnitBinner {
    pub unit: Unit,
    pub start_unit: f64, // in terms of unit
    pub end_unit: f64, // in terms of unit
    pub count: usize,
}

impl UnitBinner {
    #[allow(unused)]
    pub fn start_end_si(&self) -> (f64, f64) {
        let start_si = self.start_unit * self.unit.1;
        let end_si = self.end_unit * self.unit.1;
        (start_si, end_si)
    }

}

impl Binner for UnitBinner {
    type T = f64;

    fn bin(&self, value_si: f64) -> isize {
        let value = value_si / self.unit.1;
        let idx = (value - self.start_unit) / (self.end_unit - self.start_unit) * self.count as f64;
        idx.round() as isize
    }

    fn unbin(&self, idx: usize) -> f64 {
        let val_unit = (idx as f64 / self.count as f64) * (self.end_unit - self.start_unit) + self.start_unit;
        val_unit * self.unit.1
    }

    fn x_axis_style(&self) -> XAxisStyle {
        XAxisStyle::Continuous { label: Some(format!("[{}]", self.unit.0)) }
    }

    fn count(&self) -> usize {
        self.count
    }

    /// in SI
    fn items(&self) -> Vec<Self::T> {
        let start_si = self.start_unit * self.unit.1;
        let end_si = self.end_unit * self.unit.1;
        (0..self.count)
            .map(|idx| (idx as f64 / self.count as f64) * (end_si - start_si) + start_si)
            .collect()
    }
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct DiscreteBinner {
    pub categories: Vec<String>
}

impl Binner for DiscreteBinner {
    type T = String;

    fn count(&self) -> usize {
        self.categories.len()
    }

    fn bin(&self, value: String) -> isize {
        self.categories.iter()
            .position(|x| x == &value)
            .map(|x| x as isize)
            .expect(&format!("Could not find value {value:?}"))
    }

    fn unbin(&self, idx: usize) -> Self::T {
        self.categories[idx].clone()
    }

    fn x_axis_style(&self) -> XAxisStyle {
        XAxisStyle::Discrete(self.categories.clone())
    }

    fn items(&self) -> Vec<Self::T> {
        self.categories.clone()
    }
}

/// Bins into (y * xcount + x)
#[allow(unused)]
#[derive(Debug, Clone)]
pub struct Binner2D<Bx: Binner, By: Binner> {
    pub bx: Bx,
    pub by: By,
}

impl<Bx: Binner, By: Binner> Binner for Binner2D<Bx, By> {
    type T = (Bx::T, By::T);

    fn count(&self) -> usize {
        self.bx.count() * self.by.count()
    }

    fn bin(&self, (vx, vy): Self::T) -> isize {
        let nx = self.bx.bin(vx);
        let ny = self.by.bin(vy);
        // check OOB
        // TODO: OOB should be a better interface than returning an out of bounds value
        if nx < 0 || ny < 0 {
            return -1;
        }
        if nx >= self.bx.count() as isize || ny >= self.by.count() as isize {
            return self.count() as isize;
        }

        ny * self.bx.count() as isize + nx
    }

    fn unbin(&self, idx: usize) -> Self::T {
        let (ny, nx) = (idx / self.bx.count(), idx % self.bx.count());
        (self.bx.unbin(nx), self.by.unbin(ny))
    }

    fn x_axis_style(&self) -> XAxisStyle {
        unimplemented!("Can't produce a 1D plot with a Binner2D")
    }

    fn items(&self) -> Vec<Self::T> {
        todo!("Implement cartesian product")
    }
}

/// Convenience data structure to manage histogram/discretized probability distributions
/// Keeps track of bins and merging multiple sets of data
#[allow(unused)]
pub struct Histogram<B: Binner> {
    pub name: Arc<str>,

    pub binner: B,

    // of length `self.binner.count()`
    storage: Vec<f64>,

    n_oob: (usize, usize), // below, above
    warn_oob_when_dropped: bool,
}

impl<B: Binner> Histogram<B> {
    #[allow(dead_code)]
    pub fn new(name: String, binner: B) -> Self {
        let name = name.into();
        let count = binner.count();

        Histogram {
            name,
            binner,
            storage: std::iter::repeat_n(0.0, count).collect(),
            n_oob: (0, 0),
            warn_oob_when_dropped: true,
        }
    }

    #[allow(dead_code)]
    pub fn no_warn_oob(mut self) -> Self {
        self.warn_oob_when_dropped = false;
        self
    }

    #[allow(dead_code)]
    pub fn add_value(&mut self, value: B::T, amount: f64) {
        let idx = self.binner.bin(value);

        if idx < 0 {
            self.n_oob.0 += 1;
            return;
        }

        let idx = idx as usize;
        if idx >= self.storage.len() {
            self.n_oob.1 += 1;
            return;
        }
        self.storage[idx] += amount;
    }

    #[allow(dead_code)]
    pub fn get_count_safe(&self, value: B::T) -> Option<f64> {
        let idx = self.binner.bin(value);
        if idx < 0 {
            return None;
        }
        let idx = idx as usize;
        if idx >= self.storage.len() {
            return None;
        }
        Some(self.storage[idx])
    }

    #[allow(dead_code)]
    pub fn get_count(&self, value: B::T) -> f64 {
        let idx = self.binner.bin(value);
        if idx < 0 {
            panic!("Tried to get out-of-range value (too small, idx = {})", idx);
        }
        let idx = idx as usize;
        if idx >= self.storage.len() {
            panic!("Tried to get out-of-range value (too large, idx = {})", idx);
        }
        self.storage[idx]
    }

    /// Create a new identical but empty histogram
    /// This should later be consumed using Self::merge_worker
    #[allow(dead_code)]
    pub fn new_worker(&self) -> Histogram<B> {
        Histogram {
            binner: self.binner.clone(),
            n_oob: (0, 0),
            storage: std::iter::repeat_n(0.0, self.storage.len()).collect(),
            name: self.name.clone(),
            warn_oob_when_dropped: false, // don't want the worker to print
        }
    }

    // Merge a worker histogram created using Self::new_worker
    // Panics if an attempted merge is made with an unrelated histogram
    #[allow(dead_code)]
    pub fn merge_worker(&mut self, worker: Histogram<B>) {
        let mut identical = true;
        identical &= self.name == worker.name;
        // We can't check binner because it doesn't guarantee PartialEq but we'll assume it's correct
        identical &= self.storage.len() == worker.storage.len();
        if !identical {
            panic!("Tried to merge histogram {:?} into {:?}", self, worker);
        }
        for (my_count, worker_count) in self.storage.iter_mut().zip(worker.storage.iter()) {
            *my_count += worker_count;
        }

        // Note: worker is dropped here, however Self::new_worker sets warn_oob_when_dropped to false, so it won't print any warnings
    }
}

impl<B: Binner> Debug for Histogram<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // TODO: List start and end? Probably not in SI...
        f.debug_struct("Histogram")
            .field("name", &self.name)
            .field("binner", &self.binner)
            .finish_non_exhaustive()
    }
}

impl<B: Binner> Drop for Histogram<B> {
    fn drop(&mut self) {
        if self.warn_oob_when_dropped && (self.n_oob.0 > 0 || self.n_oob.1 > 0) {
            eprintln!(
                "[Histogram {name}] Out of bounds warning: {below} below bounds, {above} above bounds ({total} items recorded in total, {percent:.4}% out of bounds)",
                name=self.name,
                below=self.n_oob.0, above=self.n_oob.1,
                total=self.storage.len(), percent=(self.n_oob.0+self.n_oob.1) as f64 / self.storage.len() as f64 * 100.,
            );
        }
    }
}

pub enum OKLabInterpolationMethod {
    Linear,
    LCh,
}

#[allow(unused)]
pub struct ColorGradient {
    pub start: [u8; 3],
    pub end: [u8; 3],
    pub method: OKLabInterpolationMethod,
}

#[allow(unused)]
pub const COLOR_GRADIENT_STANDARD:  ColorGradient = ColorGradient {
    start: [0, 0, 255],
    end: [50, 200, 0],
    method: OKLabInterpolationMethod::LCh,
};

impl ColorGradient {
    #[allow(unused)]
    pub fn new_linear(start: [u8; 3], end: [u8; 3]) -> ColorGradient {
        ColorGradient {
            start,
            end,
            method: OKLabInterpolationMethod::Linear,
        }
    }

    #[allow(unused)]
    pub fn new_lch(start: [u8; 3], end: [u8; 3]) -> ColorGradient {
        ColorGradient {
            start,
            end,
            method: OKLabInterpolationMethod::LCh,
        }
    }

    pub fn get(&self, at: f64) -> plotly::color::Rgb {
        let at = (at as f32).clamp(0., 1.,);
        let start_lab = oklab::srgb_to_oklab(oklab::Rgb::new(self.start[0], self.start[1], self.start[2]));
        let end_lab = oklab::srgb_to_oklab(oklab::Rgb::new(self.end[0], self.end[1], self.end[2]));
        let res_lab = match self.method {
            OKLabInterpolationMethod::Linear => {
                oklab::Oklab {
                    l: start_lab.l * (1. - at) + end_lab.l * at,
                    a: start_lab.a * (1. - at) + end_lab.a * at,
                    b: start_lab.b * (1. - at) + end_lab.b * at,
                }
            }
            OKLabInterpolationMethod::LCh => {
                use std::f32::consts::PI;

                let start_c = (start_lab.a.powi(2) + start_lab.b.powi(2)).sqrt();
                let end_c = (end_lab.a.powi(2) + end_lab.b.powi(2)).sqrt();

                let mut start_h = f32::atan2(start_lab.b, start_lab.a);
                let mut end_h = f32::atan2(end_lab.b, end_lab.a);
                if start_h > end_h + PI {
                    start_h -= 2. * PI;
                }
                if end_h > start_h + PI {
                    end_h -= 2. * PI;
                }

                let l = start_lab.l * (1. - at) + end_lab.l * at;
                let c = start_c * (1. - at) + end_c * at;
                let h = start_h * (1. - at) + end_h * at;
                let a = c * h.cos();
                let b = c * h.sin();
                oklab::Oklab { l, a, b }
            }
        };
        let res = oklab::oklab_to_srgb(res_lab);
        plotly::color::Rgb::new(res.r, res.g, res.b)
    }
}


#[allow(dead_code)] // main is required for LSP not to error since common is technically an example, however when using it as a module main is considered dead
fn main() {}
