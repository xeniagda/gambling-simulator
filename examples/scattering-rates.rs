#![allow(non_snake_case, mixed_script_confusables)]

use gambling_simulator::semiconductor::{Electron, PhononType, Semiconductor};

mod common;
use common::{linspace, write_plots};
use plotly::common::DashType;
use plotly::common::{Line, Mode};
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter};

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");
    let L_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "L").expect("No L valley in GaAs");
    let X_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "X").expect("No X valley in GaAs");

    let n = 100;

    let kxs = linspace(0., 1.0 / sample_sc.lattice_constant, n).collect::<Vec<_>>();

    let mut rate_plots = Vec::new();

    for valley_idx in [Γ_valley_idx, L_valley_idx, X_valley_idx] {
        let valley = &sample_sc.valleys[valley_idx];
        let electrons = kxs.iter().map(|&kx| Electron {
            sc: &sample_sc,
            valley_idx,
            k: [kx, 0., 0.],
        }).collect::<Vec<_>>();

        let mut plot_rate_valley = Plot::new();
        plot_rate_valley.add_trace(Scatter::new (
                electrons.iter().map(|e| e.energy_eV()).collect(),
                electrons.iter().map(
                    |e| e.scattering_rate_intravalley_acoustic_phonon(PhononType::Emission).max(1.0).log10()
                ).collect(),
            )
            .mode(Mode::Lines)
            .line(Line::new().color("black"))
            .name("Intra ac. phonon em./abs.")
        );

        for (ty, name, col) in [(PhononType::Emission, "em", "blue"), (PhononType::Absorption, "abs", "green")] {
            plot_rate_valley.add_trace(Scatter::new (
                    electrons.iter().map(|e| e.energy_eV()).collect(),
                    electrons.iter().map(
                        |e| e.scattering_rate_intravalley_optical_phonon(ty).max(1.0).log10()
                    ).collect(),
                )
                .mode(Mode::Lines)
                .line(Line::new().color(col))
                .name(format!("Intra opt. phonon {}.", name))
            );

            for (destination_valley_idx, dashty) in [(Γ_valley_idx, DashType::Dash), (L_valley_idx, DashType::Dot), (X_valley_idx, DashType::DashDot)] {
                let dest_valley = &sample_sc.valleys[destination_valley_idx];

                plot_rate_valley.add_trace(Scatter::new (
                        electrons.iter().map(|e| e.energy_eV()).collect(),
                        electrons.iter().map(
                            |e| e.scattering_rate_intervalley_optical_phonon(ty, destination_valley_idx).max(1.0).log10()
                        ).collect(),
                    )
                    .mode(Mode::Lines)
                    .line(Line::new().color(col).dash(dashty))
                    .name(format!("Inter opt. phonon {}. to {}", name, dest_valley.name))
                );
            }
        }

        let p_log10_range = 10..=15;

        plot_rate_valley.set_layout(
            Layout::new()
                .width(700).height(500)
                .title(format!("Rates for {}", valley.name))
                .x_axis(
                    Axis::new()
                        .title(r"$E [\text{eV}]$")
                )
                .y_axis(
                    Axis::new()
                        .title(r"$\text{P} [\text{s}^{-1}]$")
                        .tick_values(p_log10_range.clone().map(|x| x as f64).collect())
                        .tick_text(p_log10_range.clone().map(|x| format!("$10^{{{x}}}$")).collect())
                        .range(vec![*p_log10_range.start() as f64, *p_log10_range.end() as f64])
                )
        );
        rate_plots.push(plot_rate_valley);
    }

    write_plots("scattering-rates", "scattering-rates", rate_plots);
}
