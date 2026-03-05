#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use std::sync::Arc;

use gambling_simulator::{semiconductor::{Electron, Semiconductor, StepInfo}, units, units::Unit};
use gambling_simulator::histogram::{generate_histogram_collection_struct, Histogram, UnitBinner, DiscreteBinner, Binner2D};

use plotly::{common::{Line, Mode}, layout::Axis, Layout, Plot, Scatter};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
mod common;
use tqdm::tqdm;

use crate::common::write_plots;

type BinnerTime = UnitBinner<units::PS>;
type BinnerVelocity = UnitBinner<units::MILLION_CM_PER_SECOND>;
type BinnerMechanism = DiscreteBinner<&'static str>;
type BinnerValley = DiscreteBinner<&'static str>;

generate_histogram_collection_struct! {
    struct Histograms {
        velocity: Histogram<Binner2D<BinnerTime, BinnerVelocity>>,
        mechanisms: Histogram<Binner2D<BinnerTime, BinnerMechanism>>,
        valleys: Histogram<Binner2D<BinnerTime, BinnerValley>>,
    }
}

fn generate_histogram<R: Rng + SeedableRng>(
    thread_idx: usize,
    sc: Arc<Semiconductor>,

    mut histograms: Histograms,
    step_info: StepInfo<R>, // applied_field overwritten

    n_electrons: usize,
) -> Histograms {
    let mut rng = R::from_os_rng();

    let t_stop = histograms.velocity.binner.major.end_si;

    for _run in tqdm(0..n_electrons).desc(Some(format!("Thread #{thread_idx: <4}"))).style(tqdm::Style::ASCII) {
        let mut electron = Electron::thermalized_in_field(&mut rng, sc.clone(), [0., 0., 0.], step_info.applied_field);

        let mut t = 0.0;
        while t < t_stop {
            // Free flight
            let dt = electron.free_flight_time(&mut rng, &step_info);
            electron.free_flight(dt, &step_info);
            t += dt;

            if let Some(mech) = electron.scatter(&step_info, &mut rng) {
                histograms.mechanisms.add((t, mech.name_short), 1.);
            }

            // Record in histogram
            let vx = electron.velocity()[0];
            histograms.velocity.add((t, vx), dt);
            histograms.valleys.add((t, electron.valley().name), dt);
        }
    }
    histograms
}

fn main() {
    let sample_sc = Arc::new(Semiconductor::GaAs(300.0));

    let e_x = units::KV_PER_CM::to_si(20.);

    let binner_time: BinnerTime = UnitBinner::new(
        "t",
        0., 1., 20,
    );

    let binner_velocity: BinnerVelocity = UnitBinner::new(
        "v",
        -300., 100., 200
    );
    let histo_vel = Histogram::new(
        "Velocity".to_string(),
        Binner2D {
            major: binner_time.clone(),
            minor: binner_velocity,
        },
    );

    let binner_mechanisms: BinnerMechanism = {
        let mut mechanisms = vec!["self-scatter"];
        let all_mechs = Semiconductor::all_mechanisms::<ChaCha8Rng>();
        mechanisms.extend(all_mechs.into_iter().map(|x| x.name_short));
        DiscreteBinner::new(mechanisms)
    };

    let histo_mechs = Histogram::new(
        "Mechanisms".to_string(),
        Binner2D {
            major: binner_time.clone(),
            minor: binner_mechanisms.clone(),
        }
    );

    let binner_valleys = DiscreteBinner::new(sample_sc.valleys.iter().map(|v| v.name).collect());
    let histo_valleys = Histogram::new(
        "Valleys".to_string(),
        Binner2D {
            major: binner_time.clone(),
            minor: binner_valleys.clone(),
        },
    );

    let energy_max = units::EV::to_si(2.0);
    let step_info = StepInfo {
        applied_field: [e_x, 0., 0.],
        maximum_assumed_energy: energy_max,
        scattering_mechanisms: Semiconductor::all_mechanisms::<ChaCha8Rng>(),
    };

    let n_electrons = 10000;
    let n_threads = num_cpus::get();

    let histograms: Histograms = std::thread::scope(|scope| {
        let mut histograms: Histograms = Histograms {
            velocity: histo_vel,
            mechanisms: histo_mechs,
            valleys: histo_valleys,
        };

        let mut handles = (0..n_threads).map(|thread_idx| {
            let (sample_sc, worker, step_info) = (sample_sc.clone(), histograms.get_worker(), step_info.clone());
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

            histograms.merge_worker(worker);
        }
        histograms
    });

    let plot_histo_v_dist = {
        let mut plot_histo_v_dist = Plot::new();
        for (i, t) in binner_time.steps().enumerate() {
            let color = common::COLOR_GRADIENT_STANDARD.get(i as f64 / binner_time.count as f64);

            let histo_v = histograms.velocity.as_ref_at_major(t).unwrap();

            let trace = Scatter::new(
                    histo_v.all_values().map(|(v, _total)| units::MILLION_CM_PER_SECOND::from_si(v)).collect(),
                    histo_v.all_values().map(|(_v, total)| total / histograms.velocity.as_ref().subtotal()).collect(),
                )
                .mode(Mode::Lines)
                .line(Line::new().color(color))
                .name(format!("t = {:.2} ps", units::PS::from_si(t)));

            plot_histo_v_dist.add_trace(trace);
        }
        plot_histo_v_dist.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Velocities}}, E_x = {} kV/cm$", units::KV_PER_CM::from_si(step_info.applied_field[0])))
                .x_axis(
                    Axis::new().title("$v_x [10^6 cm/s]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Count (rel)}$")
                )
        );

        plot_histo_v_dist
    };

    let plot_histo_v_time = {
        let mut plot_histo_v_time = Plot::new();

        let points = binner_time.steps()
            .map(|t| {
                let histo_v = histograms.velocity.as_ref_at_major(t).unwrap();

                (t, histo_v.mean())
            })
            .collect::<Vec<_>>();

        let trace = Scatter::new(
                points.iter().map(|(t, _)| units::PS::from_si(*t)).collect(),
                points.iter().map(|(_, v)| units::MILLION_CM_PER_SECOND::from_si(*v)).collect(),
            )
            .mode(Mode::Lines);

        plot_histo_v_time.add_trace(trace);

        plot_histo_v_time.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Mean velocity}}, E_x = {} kV/cm$", units::KV_PER_CM::from_si(step_info.applied_field[0])))
                .x_axis(
                    Axis::new().title("p [ps]")
                )
                .y_axis(
                    Axis::new().title(r"$v_x [10^6 cm/s]$")
                )
        );

        plot_histo_v_time
    };

    let plot_mechs = {
        let mut plot_mechs = Plot::new();

        let total_at_time = binner_time.steps()
            .map(|t| histograms.mechanisms.as_ref_at_major(t).unwrap().subtotal())
            .collect::<Vec<_>>();

        for mech_name in binner_mechanisms.steps() {
            let histo_mech = histograms.mechanisms.as_ref_at_minor(mech_name).unwrap();

            let trace = Scatter::new(
                    binner_time.steps().map(|t| units::PS::from_si(t)).collect(),
                    histo_mech.all_values().zip(total_at_time.iter()).map(|((_t, count), total)| 100. * count / total).collect(),
                )
                .mode(Mode::Lines)
                .name(mech_name);
            plot_mechs.add_trace(trace);
        }
        plot_mechs.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Mechanisms}}, E_x = {} kV/cm$", units::KV_PER_CM::from_si(step_info.applied_field[0])))
                .x_axis(
                    Axis::new().title("$t [ps]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Count (rel, %)}$")
                )
        );

        plot_mechs
    };

    let plot_valleys = {
        let mut plot_valleys = Plot::new();

        let total_at_time = binner_time.steps()
            .map(|t| histograms.valleys.as_ref_at_major(t).unwrap().subtotal())
            .collect::<Vec<_>>();

        for valley_name in binner_valleys.steps() {
            let histo_valley = histograms.valleys.as_ref_at_minor(valley_name).unwrap();

            let trace = Scatter::new(
                    binner_time.steps().map(|t| units::PS::from_si(t)).collect(),
                    histo_valley.all_values().zip(total_at_time.iter()).map(|((_t, count), total)| 100. * count / total).collect(),
                )
                .mode(Mode::Lines)
                .name(valley_name);
            plot_valleys.add_trace(trace);
        }
        plot_valleys.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Valleys}}, E_x = {} kV/cm$", units::KV_PER_CM::from_si(step_info.applied_field[0])))
                .x_axis(
                    Axis::new().title("$t [ps]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Count (rel, %)}$")
                )
        );

        plot_valleys
    };

    write_plots("monte-carlo", "transient-statistics", [plot_histo_v_dist, plot_histo_v_time, plot_mechs, plot_valleys]);
}
