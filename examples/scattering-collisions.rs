#![allow(non_snake_case, mixed_script_confusables)]

use gambling_simulator::consts::EV_TO_J;
use gambling_simulator::semiconductor::{Electron, Semiconductor};

mod common;
use common::write_plots;
use plotly::common::Marker;
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter3D};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn main() {
    let mut r = ChaCha8Rng::from_os_rng();

    let sample_sc = Semiconductor::GaAs(300.0);
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");
    let Γ_valley = &sample_sc.valleys[Γ_valley_idx];


    let e_init = 247e-3 * EV_TO_J;
    let k_mag = Γ_valley.kmag_for_e(e_init);

    let n = 500;
    let e = Electron {
        sc: &sample_sc,
        valley_idx: Γ_valley_idx,
        k: [k_mag, 0., 0.],
    };

    let mechs = Electron::all_mechanisms();
    let mut plots = Vec::new();
    for mech in mechs {
        eprintln!("Mechanism {} ({})", mech.name_short, mech.name_full);

        let resulting_electrons: Vec<_> = (0..n).map(|_| (mech.resulting_state)(&e, &mut r)).collect();

        let mut plot = Plot::new();
        let point= Scatter3D::new(
            vec![e.k[0] * sample_sc.lattice_constant], vec![e.k[1] * sample_sc.lattice_constant], vec![e.k[2] * sample_sc.lattice_constant],
        )
            .mode(Mode::Markers)
            .marker(Marker::new().size(5).color("red"));
        plot.add_trace(point);

        let cloud = Scatter3D::new(
            resulting_electrons.iter().map(|e| e.k[0] * sample_sc.lattice_constant).collect(),
            resulting_electrons.iter().map(|e| e.k[1] * sample_sc.lattice_constant).collect(),
            resulting_electrons.iter().map(|e| e.k[2] * sample_sc.lattice_constant).collect(),
        )
            .mode(Mode::Markers)
            .marker(Marker::new().size(1).color("gray"));
        plot.add_trace(cloud);

        plot.set_layout(
            Layout::new()
                .width(600).height(600)
                .title(mech.name_full)
                .x_axis(
                    Axis::new().title("$k_x [a^{-1}]$")
                        .range(vec![-k_mag*2., k_mag*2.])
                )
                .y_axis(
                    Axis::new().title("$k_y [a^{-1}]$")
                        .range(vec![-k_mag*2., k_mag*2.])
                )
                .z_axis(
                    Axis::new().title("$k_z [a^{-1}]$")
                        .range(vec![-k_mag*2., k_mag*2.])
                )
        );
        plots.push(plot);
    }

    write_plots("scattering-collisions", "collisions", plots);
}
