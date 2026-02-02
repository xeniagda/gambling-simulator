#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::{EV_TO_J, PLANCK_BAR_SI}, semiconductor::{Electron, Semiconductor, StepInfo}};

use plotly::{common::{Marker, MarkerSymbol, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use common::{write_plots, VALLEY_COLORS};

fn main() {
    let mut rng = ChaCha8Rng::from_os_rng();

    let sample_sc = Semiconductor::GaAs(300.0);
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");

    let energy_max = 2. * EV_TO_J;
    let e_x = 0.0; // kV/cm

    let step_info = StepInfo {
        applied_field: [e_x * 1e5, 0., 0.],
        maximum_assumed_energy: energy_max,
    };

    let mut plot_traces = Plot::new();

    let mut electron = Electron {
        sc: &sample_sc,
        valley_idx: Γ_valley_idx,
        k: [0., 0., 0.,],
        pos: [0., 0., 0.,],
    };

    let mut next_point_t = 0.;
    let points_every = 0.05e-12;
    let mut time = 0.;

    let mut total_velocity_x = 0.;

    for step in 0..2000 {
        let el_before = electron.clone();
        let res = electron.free_flight(&step_info, &mut rng);
        time += res.free_flight_time;

        total_velocity_x += electron.velocity()[0] * res.free_flight_time;

        println!(
            "(t={:.4} ps) (#{step: <3}) Moved to r = ({:.4}, {:.4}, {:.4}) a, v = ({:.4}, {:.4}, {:.4}) a/ps, k = ({:.4}, {:.4}, {:.4}) a⁻¹, E = {:.4} meV (valley {})",
            time*1e12,
            electron.pos[0] / sample_sc.lattice_constant, electron.pos[1] / sample_sc.lattice_constant, electron.pos[2] / sample_sc.lattice_constant,
            electron.velocity()[0] / sample_sc.lattice_constant * 1e-12, electron.velocity()[1] / sample_sc.lattice_constant * 1e-12, electron.velocity()[2] / sample_sc.lattice_constant * 1e-12,
            electron.k[0] * sample_sc.lattice_constant, electron.k[1] * sample_sc.lattice_constant, electron.k[2] * sample_sc.lattice_constant,
            electron.energy_eV() * 1e3, electron.valley().name,
        );

        // draw continuous trace
        // let dts = linspace_incl(0., res.free_flight_time, 5);
        let v = el_before.velocity();
        let a = [
            PLANCK_BAR_SI / electron.valley().effective_mass() * res.k_acceleration[0],
            PLANCK_BAR_SI / electron.valley().effective_mass() * res.k_acceleration[1],
            PLANCK_BAR_SI / electron.valley().effective_mass() * res.k_acceleration[2],
        ];

        let color = VALLEY_COLORS[electron.valley_idx];
        // let intermediate_positions = dts
        //     .map(|dt| [
        //         el_before.pos[0] + v[0] * dt + a[0] * dt.powi(2) / 2.,
        //         el_before.pos[1] + v[1] * dt + a[1] * dt.powi(2) / 2.,
        //         el_before.pos[2] + v[2] * dt + a[2] * dt.powi(2) / 2.,
        //     ])
        //     .collect::<Vec<_>>();
        // let trace = Scatter::new(
        //         intermediate_positions.iter().map(|pos| pos[0] / sample_sc.lattice_constant).collect(),
        //         intermediate_positions.iter().map(|pos| pos[1] / sample_sc.lattice_constant).collect(),
        //     )
        //     .mode(Mode::Lines)
        //     .line(Line::new().color(color))
        //     .name(format!("#{step} flight (in {}, E = {:.3} meV at start, t = {:.4} ps)", electron.valley().name, electron.energy_eV() * 1e3, res.free_flight_time*1e12));
        // plot_traces.add_trace(trace);
        while next_point_t < time {
            let dt = next_point_t - (time - res.free_flight_time);
            let xyz = [
                el_before.pos[0] + v[0] * dt + a[0] * dt.powi(2) / 2.,
                el_before.pos[1] + v[1] * dt + a[1] * dt.powi(2) / 2.,
                el_before.pos[2] + v[2] * dt + a[2] * dt.powi(2) / 2.,
            ];
            let trace = Scatter::new(
                    vec![xyz[0] / sample_sc.lattice_constant],
                    vec![xyz[1] / sample_sc.lattice_constant],
                )
                .mode(Mode::Markers)
                .name(format!("t={:.4} ps", next_point_t*1e12))
                .marker(Marker::new().color(color).size(4));
            plot_traces.add_trace(trace);

            next_point_t += points_every;
        }

        if let Some(mech) = electron.scatter(&step_info, &mut rng) {
            println!(
                "    BOING! {}: New state: k = ({:.4}, {:.4}, {:.4}) a⁻¹, E = {:.4} meV (valley {})",
                mech.name_short,
                electron.k[0] * sample_sc.lattice_constant, electron.k[1] * sample_sc.lattice_constant, electron.k[2] * sample_sc.lattice_constant,
                electron.energy_eV() * 1e3, electron.valley().name,
            );

            let scatter = Scatter::new(
                    vec![electron.pos[0] / sample_sc.lattice_constant],
                    vec![electron.pos[1] / sample_sc.lattice_constant],
                )
                .mode(Mode::Markers)
                .marker(Marker::new().color("red").symbol(MarkerSymbol::X))
                .name(format!("{} ({:.4} meV before)", mech.name_short, el_before.energy_eV() * 1e3));
            plot_traces.add_trace(scatter);
        }
    }

    println!("Info:");
    println!("  Mean velocity in x: {:.4} Mm/s", total_velocity_x / time / 1e6);

    plot_traces.set_layout(
        Layout::new()
            .width(1200).height(800)
            .title(format!(r"$\text{{Path}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
            .x_axis(
                Axis::new().title("$x [a]$")
            )
            .y_axis(
                Axis::new().title(r"$y [a]$")
                    .scale_anchor("x")
                    .scale_ratio(1.)
            )
    );

    write_plots("monte-carlo", "trace-flight", [plot_traces]);
}
