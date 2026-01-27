#![allow(non_snake_case, mixed_script_confusables)]

use gambling_simulator::consts::EV_TO_J;
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

    let n = 500;

    // outside this and the bands stop being correct
    let e_init = 3. * EV_TO_J;
    let k_mag = sample_sc.valleys[Γ_valley_idx].kmag_for_e(e_init);
    let kxs = linspace(0., k_mag, n).collect::<Vec<_>>();

    let mut rate_plots = Vec::new();

    for valley_idx in [Γ_valley_idx, L_valley_idx, X_valley_idx] {
        let valley = &sample_sc.valleys[valley_idx];
        eprintln!("For {}:", valley.name);
        let electrons = kxs.iter().map(|&kx| Electron {
            sc: &sample_sc,
            valley_idx,
            k: [kx, 0., 0.],
        }).collect::<Vec<_>>();

        if let Some(max_energy) = electrons.iter().map(|e| e.energy_eV()).max_by(f64::total_cmp) {
            eprintln!("  Maximum energy: {:.4} meV", 1e3 * max_energy);
        }

        let mut plot_rate_valley = Plot::new();

        let mut plot_rate_funtion = |name: &str, f: Box<dyn Fn(&Electron) -> f64>, (mode, dash, color): (Mode, DashType, &str)| {
            plot_rate_valley.add_trace(Scatter::new (
                    electrons.iter().map(|e| e.energy_eV()).collect(),
                    electrons.iter().map(|e| f(e).max(1.0).log10()).collect(),
                )
                .mode(mode)
                .line(Line::new().color(color.to_owned()).dash(dash))
                .name(name.to_owned())
            );
            eprint!("  {name:>40}: ");



            // Find maximum
            let Some((max_e, max_rate)) = electrons
                .iter()
                .map(|e| (e, f(e)))
                .max_by(|(_e1, f1), (_e2, f2)| f1.total_cmp(f2))
            else {
                eprintln!("nothing!");
                return;
            };
            eprint!("rate = {: >8.4} ps⁻¹ for E = {: >8.4} meV", max_rate / 1.0e12, 1e3 * max_e.energy_eV());
            if let Some(true) = electrons.iter().map(|e| e.energy_eV()).max_by(|e1, e2| e1.total_cmp(e2)).map(|e| e == max_e.energy_eV()) {
                eprint!(" (maximum energy)");
            }
            // Check for NaN's
            if let Some(nan_energy) = electrons
                .iter()
                .find_map(|e| f(e).is_nan().then(|| e.energy_eV())) {
                eprint!(" [NaN at {: >8.4} meV]", 1e3 * nan_energy);
            }
            eprintln!();
        };

        plot_rate_funtion(
            "Intra ac. phonon em./abs.",
            Box::new(|e| e.rate_intra_ac_phonon(None)),
            (Mode::Lines, DashType::Solid, "blue"),
        );

        for (ty, name, col) in [(PhononType::Emission, "em", "red"), (PhononType::Absorption, "abs", "purple")] {
            plot_rate_funtion(
                &format!("Intra opt. phonon {}.", name),
                Box::new(move |e| e.rate_intra_opt_phonon(ty, None)),
                (Mode::Lines, DashType::Solid, col),
            );

            for (destination_valley_idx, dashty) in [(Γ_valley_idx, DashType::Dash), (L_valley_idx, DashType::Dot), (X_valley_idx, DashType::DashDot)] {
                let dest_valley = &sample_sc.valleys[destination_valley_idx];

                plot_rate_funtion(
                    &format!("Inter opt. phonon {}. to {}", name, dest_valley.name),
                    Box::new(move |e| e.rate_inter_opt_phonon(ty, destination_valley_idx, None)),
                    (Mode::Lines, dashty, col),
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
