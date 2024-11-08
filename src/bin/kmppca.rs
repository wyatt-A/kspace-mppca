use std::path::PathBuf;
use kspace_mppca::{estimate_variance, flatten, prepare_input_from_kspace_dir, reduce_noise, unflatten, DenoiseInputs, FlattenSamplesInputs, KSpacePrepInputs, SampleOrderingNorm, UnflattenInputs};


fn main() {

    let inputs = KSpacePrepInputs {
        kspace_dir: PathBuf::from("/Users/Wyatt/scratch/S69964/object-data"),
        work_dir: PathBuf::from("/Users/Wyatt/scratch/S69964/kmppca_l2_var"),
        n_volumes: 67,
    };

    let prepped = prepare_input_from_kspace_dir(inputs);

    let inputs = FlattenSamplesInputs {
        sample_ordering_norm: SampleOrderingNorm::L2,
        file_prefix: "f".to_string(),
        work_dir: prepped.work_dir,
        n_volumes: prepped.n_volumes,
        nx: prepped.nx,
        phase_encoding_table_path: prepped.phase_encoding_table_path.clone(),
    };

    let flattened = flatten(inputs);

    let mut inputs = DenoiseInputs {
        sample_chunk_size: 2000,
        assumed_variance: None,
        assumed_global_rank: None,
        flattened_file_prefix: flattened.file_prefix,
        work_dir: flattened.work_dir,
        n_volumes: flattened.n_volumes,
    };

    let variance = estimate_variance(&inputs);

    inputs.assumed_variance = Some(variance);

    let denoised = reduce_noise(inputs);

    let inputs = UnflattenInputs {
        grid_size: [590,360,360],
        sample_ordering_file: flattened.sample_ordering_file,
        work_dir: denoised.work_dir,
        n_volumes: denoised.n_volumes,
        nx: flattened.nx,
        phase_encoding_table_path: prepped.phase_encoding_table_path,
        denoised_file_prefix: denoised.denoised_file_prefix,
        rank_filepath: denoised.rank_filepath,
        variance_filepath: denoised.variance_filepath,
        removed_energy_filepath: denoised.removed_energy_filepath,
        neighborhood_filepath: denoised.neighborhood_filepath,
    };

    unflatten(inputs);

    println!("done")
}
