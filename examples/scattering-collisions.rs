#![allow(non_snake_case, mixed_script_confusables)]

use std::f64::consts::PI;

use gambling_simulator::consts::EV_TO_J;
use gambling_simulator::semiconductor::{Electron, Semiconductor};

mod common;
use common::write_plots;
use plotly::common::{Line, Marker};
use plotly::common::Mode;
use plotly::layout::Axis;
use plotly::{Layout, Plot, Scatter3D};

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::common::linspace_incl;

fn main() {
    let mut r = ChaCha8Rng::from_os_rng();

    let sample_sc = Semiconductor::GaAs(300.0);
    let Γ_valley_idx = sample_sc.valleys.iter().position(|x| x.name == "Γ").expect("No Γ valley in GaAs");
    let Γ_valley = &sample_sc.valleys[Γ_valley_idx];

    let e_init = 700e-3 * EV_TO_J;
    let k_mag = Γ_valley.kmag_for_e(e_init);

    let n = 500;
    let e = Electron {
        sc: &sample_sc,
        valley_idx: Γ_valley_idx,
        k: [k_mag, 0., 0.],
        pos: [0., 0., 0.,],
    };

    let mechs = Electron::all_mechanisms();
    let mut plots = Vec::new();
    for mech in mechs {
        let mut plot = Plot::new();

        eprintln!("Mechanism {} ({})", mech.name_short, mech.name_full);

        // outline the sphere
        for (xhat, yhat, zhat) in [
            ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.]),
            ([0., 0., 1.], [1., 0., 0.], [0., 1., 0.]),
            ([0., 1., 0.], [0., 0., 1.], [1., 0., 0.]),
        ] {
            let phi = linspace_incl(0., 2.*PI, 100);
            let points = phi.map(|phi| {
                let r = k_mag * sample_sc.lattice_constant;
                let v = [r * phi.cos(), r * phi.sin(), 0.];
                [v[0] * xhat[0] + v[1] * yhat[0] + v[2] * zhat[0], v[0] * xhat[1] + v[1] * yhat[1] + v[2] * zhat[1], v[0] * xhat[2] + v[1] * yhat[2] + v[2] * zhat[2]]
            }).collect::<Vec<_>>();
            let trace = Scatter3D::new(
                    points.iter().map(|[x, _y, _z]| *x).collect(),
                    points.iter().map(|[_x, y, _z]| *y).collect(),
                    points.iter().map(|[_x, _y, z]| *z).collect(),
                )
                .mode(Mode::Lines)
                .line(Line::new().color("gray"));

            plot.add_trace(trace);
        }

        let resulting_electrons: Vec<_> = (0..n).map(|_| (mech.resulting_state)(&e, &mut r)).collect();

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
