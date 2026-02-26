use plotly::Plot;
use build_html::{Html, HtmlContainer, HtmlPage};

#[allow(unused)]
pub const VALLEY_COLORS: [&str; 3] = ["black", "blue", "green"];

#[allow(dead_code)]
pub fn linspace(a: f64, b: f64, steps: usize) -> impl Iterator<Item = f64> {
    (0..steps).map(move |x| (x as f64) / (steps as f64) * (b - a) + a)
}

#[allow(dead_code)]
pub fn linspace_incl(a: f64, b: f64, steps: usize) -> impl Iterator<Item = f64> {
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

    #[allow(unused)]
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

pub mod plot_histogram {
    use plotly::{Layout, Plot};

    #[allow(unused)]
    pub fn set_default_layout(plot: &mut Plot, title: &str) {
        plot.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(title)
        )
    }

    pub mod quantity {
        use plotly::layout::Axis;
        use plotly::{Plot, Scatter};
        use gambling_simulator::histogram::{HistogramRef, UnitBinner};
        use gambling_simulator::units::Unit;

        #[allow(unused)]
        pub fn set_layout<X: Unit, Y: Unit>(
            plot: &mut Plot,
            histo: HistogramRef<UnitBinner<X>>,
            y_quantity_name: &str,
        ) {
            let layout = plot.layout().clone();
            plot.set_layout(
                layout
                    .x_axis(Axis::new().title(format!(r"${}\ [\text{{{}}}]$", histo.binner.quantity_name, X::NAME)))
                    .y_axis(Axis::new().title(format!(r"${}\ [\text{{{}}}]$", y_quantity_name, Y::NAME)))
            );
        }

        #[allow(unused)]
        pub fn set_layout_log<X: Unit, Y: Unit>(
            plot: &mut Plot,
            histo: HistogramRef<UnitBinner<X>>,
            y_quantity_name: &str,
        ) {
            let layout = plot.layout().clone();
            plot.set_layout(
                layout
                    .x_axis(Axis::new().title(format!(r"${}\ [\text{{{}}}]$", histo.binner.quantity_name, X::NAME)))
                    .y_axis(Axis::new().title(format!(r"${}\ [\text{{{}, log10}}]$", y_quantity_name, Y::NAME)))
            );
        }

        #[allow(unused)]
        pub fn plot<X: Unit, Y: Unit>(
            histo: HistogramRef<UnitBinner<X>>,
        ) -> Box<Scatter<f64, f64>> {
            Scatter::new(
                histo.all_values().map(|(x, _y)| X::from_si(x)).collect(),
                histo.all_values().map(|(_x, y)| Y::from_si(y)).collect(),
            )
        }

        #[allow(unused)]
        pub fn plot_log<X: Unit, Y: Unit>(
            histo: HistogramRef<UnitBinner<X>>,
        ) -> Box<Scatter<f64, f64>> {
            Scatter::new(
                histo.all_values().map(|(x, _y)| X::from_si(x)).collect(),
                histo.all_values().map(|(_x, y)| Y::from_si(y).log10()).collect(),
            )
        }
    }
}

#[allow(dead_code)] // main is required for LSP not to error since common is technically an example, however when using it as a module main is considered dead
fn main() {}
