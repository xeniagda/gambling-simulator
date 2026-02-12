#![allow(non_snake_case, mixed_script_confusables)] // for band names such as Γ and L etc

use gambling_simulator::{consts::EV_TO_J, histogram::units::KV_PER_CM, semiconductor::{Electron, Semiconductor, StepInfo}};
use gambling_simulator::histogram::{generate_histogram_collection_struct, Histogram, Binner, DiscreteBinner, UnitBinner, Binner2D, units, units::Unit};

use plotly::{Layout, Plot, Scatter, common::{Line, Marker, Mode}, layout::Axis};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
mod common;
use common::{write_plots, VALLEY_COLORS};
use tqdm::tqdm;

type ValleyBinner = DiscreteBinner<&'static str>;
type MechanismBinner = DiscreteBinner<&'static str>;

generate_histogram_collection_struct! {
    struct Histograms {
        energy_histogram: Histogram<Binner2D<ValleyBinner, UnitBinner<units::MEV>>>,
        mechanism_histogram: Histogram<Binner2D<ValleyBinner, MechanismBinner>>,
        velocity_histogram: Histogram<UnitBinner<units::MILLION_CM_PER_SECOND>>,
    }
}

fn generate_histogram(
    thread_idx: usize,
    mut histograms: Histograms,

    sc: Semiconductor,
    step_info: StepInfo,

    n_electrons: usize,
    t_stop: f64,
) -> Histograms {
    let mut rng = ChaCha8Rng::from_os_rng();

    for _run in tqdm(0..n_electrons).desc(Some(format!("Thread #{thread_idx: <4}"))) {
        let mut electron = Electron::thermalized_in_field(&mut rng, &sc, [0., 0., 0.], step_info.applied_field);

        let mut t = 0.;
        while t < t_stop {
            // Set histogram
            let valley = electron.valley().name;
            let flight = electron.free_flight(&step_info, &mut rng);
            let dt = flight.free_flight_time;
            t += dt;

            histograms.energy_histogram.add((valley, electron.energy()), dt);
            histograms.velocity_histogram.add(electron.velocity()[0], dt);

            if let Some(mech) = electron.scatter(&step_info, &mut rng) {
                // Find index of mechanism
                histograms.mechanism_histogram.add((valley, mech.name_short), 1.);
            }  else {
                histograms.mechanism_histogram.add((valley, "self-scatter"), 1.);
            }
        }
    }
    histograms
}

fn main() {
    let sample_sc = Semiconductor::GaAs(300.0);

    let e_x = 10.; // kV/cm

    let e_x_si = KV_PER_CM::to_si(e_x);

    let step_info = StepInfo {
        applied_field: [e_x_si, 0., 0.],
        maximum_assumed_energy: 2.0 * EV_TO_J,
    };

    let mechanism_names: Vec<&'static str> = {
        let mut mechanism_names = vec!["self-scatter"];
        mechanism_names.extend(
            Electron::all_mechanisms::<ChaCha8Rng>().iter()
                .map(|x| x.name_short)
        );
        mechanism_names
    };
    let mechanism_binner = DiscreteBinner::new(mechanism_names);

    let valley_binner: ValleyBinner = DiscreteBinner::new(sample_sc.valleys.iter().map(|x| x.name).collect());

    let energy_histogram = Histogram::new(
        "Energy".into(),
        Binner2D {
            major: valley_binner.clone(),
            minor: UnitBinner::<units::MEV>::new(
                "E",
                0., 2000., 1000,
            ),
        }
    );

    let velocity_histogram = Histogram::new(
        "Velocity".into(),
        UnitBinner::<units::MILLION_CM_PER_SECOND>::new(
            "v_x",
            -100., 100., 1000,
        ),
    );

    let mechanism_histogram = Histogram::new(
        "Mechanism".into(),
        Binner2D {
            major: valley_binner.clone(),
            minor: mechanism_binner.clone(),
        }
    );

    let histograms = Histograms {
        energy_histogram,
        mechanism_histogram,
        velocity_histogram,
    };

    let n_electrons = 100;
    let t_stop = 40e-12; // ps
    let n_threads = num_cpus::get();

    let histograms: Histograms = std::thread::scope(|scope| {
        let mut histograms = histograms;

        let mut handles = (0..n_threads).map(|thread_idx| {
            let (sample_sc, worker) = (sample_sc.clone(), histograms.get_worker());
            let handle = scope.spawn(move || generate_histogram(thread_idx, worker, sample_sc, step_info,  n_electrons, t_stop));
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

    // Calculate mean velocity
    {
        let mean_v_si = histograms.velocity_histogram.as_ref().mean();
        let stddev_v_si = histograms.velocity_histogram.as_ref().stddev();

        eprintln!("Mean velocty = {:.4} ± {:.4} 10^6 cm/s", histograms.velocity_histogram.binner.from_si(mean_v_si), histograms.velocity_histogram.binner.from_si(stddev_v_si));

        let gaas_mobility = 0.9; // m/s / (V/m)
        let lin_mean_vel = -gaas_mobility * step_info.applied_field[0];
        eprintln!("  (should be {:.4} 10^6 cm/s)", histograms.velocity_histogram.binner.from_si(lin_mean_vel));
    }

    // Histogram of energies
    let plot_histo_e = {
        let mut plot_histo_e = Plot::new();
        for (idx, valley) in histograms.energy_histogram.binner.major.steps().enumerate() {
            let histo_e = histograms.energy_histogram.as_ref_at_major(valley).unwrap();
            let total = histograms.energy_histogram.total;
            let trace = Scatter::new(
                histo_e.all_values().map(|(val, _time)| histo_e.binner.from_si(val)).collect(),
                histo_e.all_values().map(|(_val, time)| (time / total / histo_e.binner.bin_size_unit()).log10()).collect(),
                )
                .name(valley)
                .line(Line::new().color(VALLEY_COLORS[idx]));
            plot_histo_e.add_trace(trace);
        }

        plot_histo_e.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Energies}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
                .x_axis(
                    Axis::new().title("$E [meV]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Time (rel)} / meV, \text{log10}$")
                )
        );
        plot_histo_e
    };
    let plot_histo_v = {
        let mut plot_histo_v = Plot::new();
        let histo_v = histograms.velocity_histogram;
        let total = histo_v.total;
        let trace = Scatter::new(
            histo_v.all_values().into_iter().map(|(val, _time)| histo_v.binner.from_si(val)).collect(),
            histo_v.all_values().into_iter().map(|(_val, time)| time / total / histo_v.binner.bin_size_unit()).collect(),
            )
            .line(Line::new());
        plot_histo_v.add_trace(trace);

        plot_histo_v.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Velocities}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
                .x_axis(
                    Axis::new().title("$v_x [10^6 cm/s]$")
                )
                .y_axis(
                    Axis::new().title(r"$\text{Time (rel)}$")
                )
        );
        plot_histo_v
    };
    let plot_histo_mechs = {
        let mut plot_histo_mechs = Plot::new();

        for (idx, valley) in valley_binner.steps().enumerate() {
            let histo_mechs = histograms.mechanism_histogram.as_ref_at_major(valley).unwrap();
            let total = histograms.mechanism_histogram.total;

            let trace = Scatter::new(
                    histo_mechs.all_values().enumerate().map(|(idx, _)| idx).collect(),
                    histo_mechs.all_values().enumerate().map(|(_, (_name, count))| (count / total).log10()).collect(),
                )
                .mode(Mode::Markers)
                .marker(Marker::new().color(VALLEY_COLORS[idx]))
                .name(valley);
            plot_histo_mechs.add_trace(trace);
        }

        let n_y_log10 = 5;

        plot_histo_mechs.set_layout(
            Layout::new()
                .width(1200).height(800)
                .title(format!(r"$\text{{Mechanisms}}, E_x = {} kV/cm$", step_info.applied_field[0] / 1e5))
                .x_axis(
                    Axis::new().title("Mechanism")
                        .tick_values((0..mechanism_binner.count()).map(|x| x as f64).collect())
                        .tick_text(mechanism_binner.steps().collect())
                )
                .y_axis(
                    Axis::new().title(r"$\text{Count (rel)}$")
                        .range(vec![-n_y_log10, 0])
                        .tick_values((-n_y_log10..=0).map(|x| x as f64).collect())
                        .tick_text((-n_y_log10..=0).map(|e| format!("$10^{{{e}}}$")).collect())
                )
        );
        plot_histo_mechs
    };

    write_plots("monte-carlo", "statistics", [plot_histo_e, plot_histo_v, plot_histo_mechs]);
}
