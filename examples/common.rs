use plotly::Plot;
use build_html::{Html, HtmlContainer, HtmlPage};

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

#[allow(dead_code)] // main is required for LSP not to error since common is technically an example, however when using it as a module main is considered dead
fn main() {}
