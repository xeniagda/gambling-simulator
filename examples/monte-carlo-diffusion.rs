#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use std::sync::Arc;

use gambling_simulator::{semiconductor::{Electron, Semiconductor, StepInfo}, units, units::Unit};
use gambling_simulator::histogram::{generate_histogram_collection_struct, Histogram, Binner2D, UnitBinner};

use plotly::Plot;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
mod common;
use tqdm::tqdm;

use crate::common::{write_plots, plot_histogram};

generate_histogram_collection_struct! {
    struct Histograms {
        positions: Histogram<Binner2D<UnitBinner<units::PS>, UnitBinner<units::NM>>>,
    }
}

fn generate_histogram<R: Rng + SeedableRng>(
    thread_idx: usize,
    sc: Arc<Semiconductor>,

    mut histo: Histograms,
    step_info: StepInfo<R>, // applied_field overwritten

    n_electrons: usize,
) -> Histograms {
    let mut rng = R::from_os_rng();

    let t_stop = histo.positions.binner.major.end_si;

    for _run in tqdm(0..n_electrons).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        let mut electron = Electron::thermalized_in_field(&mut rng, sc.clone(), [0., 0., 0.], step_info.applied_field);

        let mut t = 0.;

        while t < t_stop {
            // Step electron
            let dt = electron.free_flight_time(&mut rng, &step_info);
            electron.free_flight(dt, &step_info);
            t += dt;

            // Scatter electron
            electron.scatter(&step_info, &mut rng);

            // Set histogram
            let x_now = electron.pos[0];
            histo.positions.add((t, x_now), dt);
        }
    }
    histo
}

fn main() {
    let sample_sc = Arc::new(Semiconductor::GaAs(300.0));

    let energy_max = units::EV::to_si(2.);

    let ex = units::KV_PER_CM::to_si(10.);
    let step_info = StepInfo {
        applied_field: [ex, 0., 0.], // will be overwritten
        maximum_assumed_energy: energy_max,
        scattering_mechanisms: Semiconductor::all_mechanisms::<ChaCha8Rng>(),
    };

    let binner_time = UnitBinner::<units::PS>::new(
        "t", 0., 20., 10,
    );
    let binner_pos = UnitBinner::<units::NM>::new(
        "x", -20000., 2000., 200,
    );

    let position_histo = Histogram::new(
        "position".to_string(),
        Binner2D {
            major: binner_time.clone(),
            minor: binner_pos.clone(),
        },
    );


    let n_electrons = 500;
    let n_threads = 8; //num_cpus::get();

    let histo: Histograms = std::thread::scope(|scope| {
        let mut histo = Histograms {
            positions: position_histo,
        };

        let mut handles = (0..n_threads).map(|thread_idx| {
            let (sample_sc, worker, step_info) = (sample_sc.clone(), histo.get_worker(), step_info.clone());
            let handle = scope.spawn(move || {
                generate_histogram(thread_idx, sample_sc, worker, step_info, n_electrons)
            });
            (handle, thread_idx)
        }).collect::<Vec<_>>();

        while !handles.is_empty() {
            let Some(finished_idx) = handles.iter().position(|(h, _idx)| h.is_finished()) else {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            };
            let (handle, thread_idx) = handles.remove(finished_idx);
            let Ok(worker) = handle.join() else {
                eprintln!("thread {thread_idx} panicked :(");
                continue;
            };

            histo.merge_worker(worker);
        }
        histo
    });

    let plot_histo_x = {
        let mut plot_histo_x = Plot::new();
        type Y = units::ELECTRONS_PER_CM_CUBED;

        plot_histogram::set_default_layout(&mut plot_histo_x, "Position");
        plot_histogram::quantity::set_layout::<_, Y>(&mut plot_histo_x, histo.positions.as_ref_at_major(0.).unwrap(), "n");
        for t in binner_time.steps() {
            let trace = plot_histogram::quantity::plot::<_, Y>(histo.positions.as_ref_at_major(t).unwrap())
                .name(format!("t = {}", units::PS::format(t)));
            plot_histo_x.add_trace(trace);
        }

        plot_histo_x
    };

    let mut histo_mean_position_over_time = Histogram::new(
        "Mean position".to_string(),
        binner_time.clone(),
    );
    let mut histo_int_diff = Histogram::new(
        "Integral of diffusion coefficient".to_string(),
        binner_time.clone(),
    );
    let mut histo_diff_divide = Histogram::new(
        "Diffusion coefficient".to_string(),
        binner_time.clone(),
    );
    for t in binner_time.steps() {
        let slice = histo.positions.as_ref_at_major(t).unwrap();
        histo_mean_position_over_time.add(t, slice.mean());

        let var = slice.stddev().powi(2);
        histo_int_diff.add(t, 0.5 * var);
        histo_diff_divide.add(t, 0.5 * var / t);
    }

    let histo_diff = histo_int_diff.derivative();

    let plot_histo_x_over_time = {
        let mut plot_histo_x_over_time = Plot::new();
        plot_histogram::set_default_layout(&mut plot_histo_x_over_time, "Position");
        type Y = units::NM;
        plot_histogram::quantity::set_layout::<_, Y>(&mut plot_histo_x_over_time, histo_mean_position_over_time.as_ref(), r"\langle x \rangle");

        let trace = plot_histogram::quantity::plot::<_, Y>(histo_mean_position_over_time.as_ref());
        plot_histo_x_over_time.add_trace(trace);

        plot_histo_x_over_time
    };

    let plot_histo_int_diff = {
        let mut plot_histo_int_diff = Plot::new();
        plot_histogram::set_default_layout(&mut plot_histo_int_diff, "Integral of diffusion coefficient");
        type Y = units::CM_SQUARED;
        plot_histogram::quantity::set_layout::<_, Y>(&mut plot_histo_int_diff, histo_int_diff.as_ref(), r"\int D dt");

        let trace = plot_histogram::quantity::plot::<_, Y>(histo_int_diff.as_ref());
        plot_histo_int_diff.add_trace(trace);

        plot_histo_int_diff
    };

    let plot_histo_diff = {
        let mut plot_histo_diff = Plot::new();
        plot_histogram::set_default_layout(&mut plot_histo_diff, "Diffusion coefficient");
        type Y = units::CM_SQUARED_PER_SECOND;
        plot_histogram::quantity::set_layout::<_, Y>(&mut plot_histo_diff, histo_diff.as_ref(), r"D");

        let trace = plot_histogram::quantity::plot::<_, Y>(histo_diff.as_ref())
            .name("Derivative");
        plot_histo_diff.add_trace(trace);

        let trace = plot_histogram::quantity::plot::<_, Y>(histo_diff_divide.as_ref())
            .name("Divide");
        plot_histo_diff.add_trace(trace);

        plot_histo_diff
    };

    write_plots("monte-carlo", "diffusion", [plot_histo_x, plot_histo_x_over_time, plot_histo_int_diff, plot_histo_diff]);
}
