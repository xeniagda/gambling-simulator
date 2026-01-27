#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::consts::J_TO_EV;
use gambling_simulator::semiconductor::{Electron, Semiconductor};

use plotly::common::{Line, Mode};
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};

mod common;
use common::{linspace, write_plots};

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);
    // Plot E vs k

    let mut plot = Plot::new();
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");
    let L_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "L").expect("No L valley in GaAs");
    let X_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "X").expect("No X valley in GaAs");

    let valleys = [(L_valley_idx, "red"), (Γ_valley_idx, "green"), (X_valley_idx, "blue")];
    let line_names = ["Λ", "Δ"];

    let rel_extent = 0.2;
    let n = 40;

    let mut cum_x = 0.0;
    let (mut tick_values, mut tick_labels) = (Vec::new(), Vec::new());

    for (i, line_name) in line_names.into_iter().enumerate() {
        let before = valleys[i];
        let next = valleys[(i+1) % valleys.len()];

        #[derive(PartialEq)]
        enum Which {
            Before, Next
        }
        use Which::*;
        for which in [Before, Next] {
            let ((this_idx, col), (other_idx, _)) = if which == Before { (before, next) } else { (next, before) };
            let (this, other) = (&sample_sc.valleys[this_idx], &sample_sc.valleys[other_idx]);
            let (kt, ko) = (this.k_center[0], other.k_center[0]);
            let delta = [ko[0] - kt[0], ko[1] - kt[1], ko[2] - kt[2]];
            let dist = delta.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

            let ts = linspace(0., rel_extent, n);
            let ks = ts.map(|t| [delta[0] * t, delta[1] * t, delta[2] * t]);

            let electrons = ks.map(|k| Electron {
                sc: &sample_sc,
                valley_idx: this_idx,
                k,
            }).collect::<Vec<_>>();

            let dir_mul = if which == Before { 1. } else { -1. };
            let scatter = Scatter::new(
                    electrons.iter().map(|el| (dir_mul * el.k_mag() + cum_x) * sample_sc.lattice_constant).collect(),
                    electrons.iter().map(|el| (el.energy() + this.energy) * J_TO_EV).collect(),
                )
                .mode(Mode::Lines)
                .line(Line::new().color(col))
                .name(this.name)
                .legend_group(this.name);
            plot.add_trace(scatter);

            if which == Which::Next || i == 0 {
                tick_values.push(cum_x * sample_sc.lattice_constant);
                tick_labels.push(this.name);
            }
            if which == Which::Before {
                tick_values.push((cum_x + dist / 2.0) * sample_sc.lattice_constant);
                tick_labels.push(line_name);
            }

            if which == Before {
                cum_x += dist;
            }
        }
    }
    plot.set_layout(
        Layout::new()
            .width(400).height(400)
            .title("E vs k")
            .x_axis(
                Axis::new().title("$k$")
                    .range(vec![0.0, cum_x * sample_sc.lattice_constant])
                    .tick_values(tick_values)
                    .tick_text(tick_labels)
            )
            .y_axis(
                Axis::new()
                    .title(r"$E [\text{eV}]$")
                    .range(vec![0.0, 4.0])
            )
    );

    write_plots("gaas-rt", "E-vs-k", [plot]);

}
